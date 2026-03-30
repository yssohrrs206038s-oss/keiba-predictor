"""
週次外れ分析レポート

results_history.csv と predictions_cache.json / manual_results.json から
外れパターンを分類し Discord に送信する。

使い方:
    python -m keiba_predictor.analysis.loss_analysis
"""

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_PATH = DATA_DIR / "predictions_cache.json"
MANUAL_PATH = DATA_DIR / "manual_results.json"
HISTORY_PATH = DATA_DIR / "results_history.csv"


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


MODEL_ID = "claude-haiku-4-5-20251001"


def generate_loss_comment(race_name: str, pred: dict, actual_result: list) -> str:
    """
    外れたレースについてClaude Haikuが「なぜ外れたか」を150文字以内で分析する。
    ANTHROPIC_API_KEY 未設定時は空文字列を返す。
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""

    try:
        import anthropic
    except ImportError:
        return ""

    honmei = pred.get("honmei", {})
    hon_num = honmei.get("horse_number", "?")
    hon_name = honmei.get("horse_name", "?")
    hon_prob = honmei.get("prob", 0) * 100

    # SHAP要因
    shap_top = honmei.get("shap_top", [])
    shap_str = ""
    if shap_top:
        plus = [s["label"] for s in shap_top if s.get("value", 0) > 0]
        minus = [s["label"] for s in shap_top if s.get("value", 0) < 0]
        if plus:
            shap_str += f"プラス要因: {', '.join(plus)}"
        if minus:
            shap_str += f" マイナス要因: {', '.join(minus)}"

    # 危険馬
    dangerous = pred.get("dangerous_horses", [])
    danger_str = ", ".join(
        f"{d.get('horse_number')}番{d.get('horse_name', '')}({d.get('popularity')}人気)"
        for d in dangerous
    ) if dangerous else "なし"

    # モンテカルロ
    sim = pred.get("simulation", {})
    sim_str = ""
    if sim and str(hon_num) in sim:
        mc = sim[str(hon_num)]
        tag = "安定軸" if mc.get("is_stable") else "展開依存"
        sim_str = f"モンテカルロ判定: {tag}"

    # 結果
    result_str = " ".join(f"{i+1}着{n}番" for i, n in enumerate(actual_result[:3]))

    prompt = (
        f"あなたはKEIBA EDGEのAIアナリストです。\n"
        f"以下のレースで予想が外れました。データを分析して"
        f"「なぜ外れたか」を150文字以内で簡潔に説明してください。\n\n"
        f"レース: {race_name}\n"
        f"予想本命: {hon_num}番{hon_name}（AI確率{hon_prob:.1f}%）\n"
        f"実際の結果: {result_str}\n"
        f"危険馬指定: {danger_str}\n"
        f"{sim_str}\n"
        f"SHAP要因: {shap_str}\n\n"
        f"150文字以内でテキストのみ出力（JSONやコードブロック不要）"
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()[:200]
    except Exception as e:
        logger.warning(f"外れ分析AI生成失敗: {e}")
        return ""


def analyze_week() -> str:
    """今週のレース結果を分析してレポート文字列を返す。"""
    cache = _load_json(CACHE_PATH)
    manual = _load_json(MANUAL_PATH)

    if not cache and not manual:
        return ""

    sep = "━" * 18
    danger_hits: list[str] = []    # パターン1
    honmei_losses: list[str] = []  # パターン2
    dark_horses: list[str] = []    # パターン3
    wins: list[str] = []           # パターン4
    fukusho_miss_races: list[tuple[str, dict, list]] = []  # (race_name, pred, result)

    # manual_results.json をベースに分析（結果データがある）
    sources = manual if manual else {}
    # manual がない場合は cache + history から組み立て
    if not sources and HISTORY_PATH.exists():
        try:
            hist = pd.read_csv(HISTORY_PATH, encoding="utf-8-sig", dtype=str)
            for _, row in hist.iterrows():
                rid = str(row.get("race_id", ""))
                if rid and rid not in sources:
                    sources[rid] = {
                        "race_name": row.get("race_name", rid),
                        "result": [
                            int(row.get("actual1_num", 0) or 0),
                            int(row.get("actual2_num", 0) or 0),
                            int(row.get("actual3_num", 0) or 0),
                        ],
                        "fukusho_hit": row.get("fukusho_hit") == "True",
                        "umaren_hit": row.get("umaren_hit") == "True",
                        "sanrenpuku_hit": row.get("sanrenpuku_hit") == "True",
                    }
        except Exception:
            pass

    for race_id, m in sources.items():
        race_name = m.get("race_name", race_id)
        result_nums = m.get("result", [])
        pred = cache.get(race_id, {})
        if not result_nums:
            continue

        top3_actual = set(result_nums[:3])
        predicted_nums = set(pred.get("predicted_top5_nums", []))

        # パターン1: 危険馬が3着以内
        for d in pred.get("dangerous_horses", []):
            dnum = d.get("horse_number")
            dname = d.get("horse_name", "")
            if dnum and dnum in top3_actual:
                pos = result_nums.index(dnum) + 1 if dnum in result_nums else "?"
                danger_hits.append(f"{race_name} {dname}{pos}着")

        # パターン2: 本命大敗（5着以下）
        honmei_num = pred.get("predicted_top3_nums", [None])[0] if pred.get("predicted_top3_nums") else None
        if honmei_num and honmei_num not in top3_actual:
            honmei_name = pred.get("honmei", {}).get("horse_name", f"{honmei_num}番")
            # result に含まれていれば着順がわかる
            if honmei_num in result_nums:
                pos = result_nums.index(honmei_num) + 1
                if pos >= 5:
                    honmei_losses.append(f"{race_name} {honmei_name}{pos}着")
            else:
                honmei_losses.append(f"{race_name} {honmei_name}着外")

        # パターン3: 穴馬見逃し（predicted_top5外が3着以内）
        for num in result_nums[:3]:
            if num not in predicted_nums and num != 0:
                pos = result_nums.index(num) + 1
                dark_horses.append(f"{race_name} {num}番{pos}着")

        # 複勝ハズレの収集（AI外れ分析用）
        is_fukusho_miss = not m.get("fukusho_hit", False)
        if is_fukusho_miss and pred and result_nums:
            fukusho_miss_races.append((race_name, pred, result_nums))

        # パターン4: 的中
        hit_parts = []
        if m.get("fukusho_hit"):
            hit_parts.append("複勝")
        if m.get("umaren_hit"):
            hit_parts.append("馬連")
        if m.get("sanrenpuku_hit"):
            hit_parts.append("3連複")
        if hit_parts:
            wins.append(f"{race_name}{'・'.join(hit_parts)}")

    # レポート組み立て
    lines = ["📊 **【KEIBA EDGE 週次分析】**", sep]

    if danger_hits:
        lines.append(f"⚠️ 危険馬が3着以内: {len(danger_hits)}件")
        for h in danger_hits:
            lines.append(f"　{h}")
    else:
        lines.append("⚠️ 危険馬が3着以内: 0件")

    if honmei_losses:
        lines.append(f"💔 本命大敗: {len(honmei_losses)}件")
        for h in honmei_losses:
            lines.append(f"　{h}")
    else:
        lines.append("💔 本命大敗: 0件")

    if dark_horses:
        lines.append(f"🎯 穴馬見逃し: {len(dark_horses)}件")
        for h in dark_horses[:5]:  # 最大5件
            lines.append(f"　{h}")
    else:
        lines.append("🎯 穴馬見逃し: 0件")

    if wins:
        lines.append(f"✅ 的中: {', '.join(wins)}")
    else:
        lines.append("✅ 的中: なし")

    lines.append(sep)

    # 示唆
    if danger_hits:
        lines.append("💡 来週への示唆: 危険馬判定の閾値見直しを検討")
    elif honmei_losses:
        lines.append("💡 来週への示唆: 本命選定の精度向上を検討")
    elif dark_horses:
        lines.append("💡 来週への示唆: 穴馬検出の感度向上を検討")
    else:
        lines.append("💡 来週への示唆: 好調維持。現在のモデル設定を継続")

    # AI外れ分析コメント（複勝ハズレのレースのみ）
    if fukusho_miss_races and os.environ.get("ANTHROPIC_API_KEY"):
        lines += ["", sep, "💔 **本命大敗レース AI分析**", sep]
        for rname, rpred, rresult in fukusho_miss_races[:3]:  # 最大3レース
            hon = rpred.get("honmei", {})
            hon_name = hon.get("horse_name", "?")
            hon_num = hon.get("horse_number", "?")
            # 本命の着順
            hon_pos = "着外"
            if hon_num in rresult:
                hon_pos = f"{rresult.index(hon_num) + 1}着"
            lines.append(f"{rname}: ◎{hon_name}{hon_pos}")
            comment = generate_loss_comment(rname, rpred, rresult)
            if comment:
                lines.append(f"🤖 AI分析: 「{comment}」")
            lines.append("")

    # 騎手別・コース別成績サマリー
    try:
        from keiba_predictor.history import load_history, jockey_stats, course_stats
        hist_df = load_history()
        if not hist_df.empty:
            lines += ["", sep, "📊 **成績サマリー**"]

            j_stats = jockey_stats(hist_df)
            if j_stats:
                lines.append("🏅 本命馬別（3戦以上）:")
                for j in j_stats[:5]:
                    lines.append(f"　{j['name']} 複勝{j['fukusho_rate']*100:.0f}% ({j['n']}戦)")

            c_stats = course_stats(hist_df)
            if c_stats:
                lines.append("📍 コース別:")
                for c in c_stats[:5]:
                    lines.append(f"　{c['course']} 複勝{c['fukusho_rate']*100:.0f}% ({c['n']}戦)")
    except Exception as e:
        logger.debug(f"成績サマリー生成失敗: {e}")

    return "\n".join(lines)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    report = analyze_week()
    if not report:
        print("分析対象のデータがありません")
        return

    print(report)

    # Discord送信
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if webhook_url:
        import requests
        try:
            resp = requests.post(webhook_url, json={"content": report}, timeout=15)
            ok = resp.status_code in (200, 204)
            print(f"Discord送信: {'成功' if ok else f'失敗({resp.status_code})'}")
        except Exception as e:
            print(f"Discord送信失敗: {e}")
    else:
        print("DISCORD_WEBHOOK_URL 未設定 → Discord送信スキップ")


if __name__ == "__main__":
    main()
