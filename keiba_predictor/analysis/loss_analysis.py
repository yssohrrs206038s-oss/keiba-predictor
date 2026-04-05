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
    """今週のレース結果を集計し、的中率・回収率のみ返す。"""
    from datetime import date, timedelta

    if not HISTORY_PATH.exists():
        return ""

    try:
        hist = pd.read_csv(HISTORY_PATH, encoding="utf-8-sig", dtype=str)
    except Exception:
        return ""

    if hist.empty:
        return ""

    # 今週（月曜〜日曜）でフィルタ
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)
    ws_str = week_start.isoformat()
    we_str = week_end.isoformat()

    mask = hist["date"].apply(lambda d: ws_str <= str(d)[:10] <= we_str if pd.notna(d) else False)
    week = hist[mask]
    if week.empty:
        return ""

    n = len(week)
    f_hits = sum(1 for _, r in week.iterrows() if r.get("fukusho_hit") == "True")
    u_hits = sum(1 for _, r in week.iterrows()
                 if r.get("umaren_hit") == "True" or r.get("wide_hit") == "True")
    s_hits = sum(1 for _, r in week.iterrows() if r.get("sanrenpuku_hit") == "True")

    total_bet = sum(int(r.get("bet_total") or 0) for _, r in week.iterrows())
    total_ret = sum(int(r.get("return_total") or 0) for _, r in week.iterrows())
    roi = (total_ret / total_bet * 100) if total_bet > 0 else 0

    sep = "━" * 16
    lines = [
        "📊 **【KEIBA EDGE】今週の成績**",
        sep,
        f"複勝的中率: {f_hits/n*100:.0f}%（{f_hits}/{n}）",
        f"馬連的中率: {u_hits/n*100:.0f}%（{u_hits}/{n}）",
        f"3連複的中率: {s_hits/n*100:.0f}%（{s_hits}/{n}）",
        f"回収率: {roi:.0f}%",
        sep,
    ]

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
