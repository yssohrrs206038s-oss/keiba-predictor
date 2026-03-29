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
