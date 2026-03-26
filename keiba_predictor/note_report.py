"""
KEIBA EDGE 週次予想レポート生成（note 記事用 Markdown）

【使用方法】
    python -m keiba_predictor.note_report
    python -m keiba_predictor.note_report --output /path/to/output.md

【出力先】
    keiba_predictor/data/note_report_YYYYMMDD.md

【データソース】
    keiba_predictor/data/predictions_cache.json  ← 金曜予想時に生成
    keiba_predictor/data/results_history.csv     ← 累計実績
"""

import json
import logging
import os
import re
import traceback
import urllib.request
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from typing import Optional

from keiba_predictor.history import (
    cumulative_summary,
    hit_streak,
    load_history,
)

logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent / "data"
CACHE_PATH = DATA_DIR / "predictions_cache.json"

JRA_VENUES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

_GRADE_RE = {
    "GI":   re.compile(r"[（(]G\s*[1Ⅰ][）)]|[（(]GI[）)]",   re.I),
    "GII":  re.compile(r"[（(]G\s*[2Ⅱ][）)]|[（(]GII[）)]",  re.I),
    "GIII": re.compile(r"[（(]G\s*[3Ⅲ][）)]|[（(]GIII[）)]", re.I),
}


# ── ヘルパー ─────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"予想キャッシュが見つかりません: {CACHE_PATH}")
    with open(CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _grade_from_name(race_name: str) -> str:
    for grade, pat in _GRADE_RE.items():
        if pat.search(race_name):
            return grade
    return ""


def _venue_from_race_id(race_id: str) -> str:
    return JRA_VENUES.get(race_id[4:6], "")


def _weekend_label(race_dates: list[str]) -> str:
    """レース日付リストから '2026/03/28-29' 形式を返す。"""
    parsed = []
    for d in race_dates:
        try:
            from datetime import datetime
            parsed.append(datetime.strptime(d, "%Y-%m-%d").date())
        except Exception:
            pass
    if not parsed:
        return ""
    parsed.sort()
    first, last = parsed[0], parsed[-1]
    # 土曜起点の週末ラベル
    wd = first.weekday()
    sat = first - timedelta(days=(wd - 5) % 7) if wd != 5 else first
    sun = sat + timedelta(days=1)
    return f"{sat.year}/{sat.month:02d}/{sat.day:02d}-{sun.day:02d}"


# ── レポート生成 ──────────────────────────────────────────────────────

def generate_note_report(output_path: Optional[Path] = None) -> str:
    """
    predictions_cache.json と results_history.csv から note 記事用 Markdown を生成する。

    Returns:
        生成した Markdown 文字列
    """
    cache = _load_cache()
    if not cache:
        raise ValueError("予想キャッシュが空です。先に notify --mode predict を実行してください。")

    # 週末ラベル
    race_dates = [r.get("race_date", "") for r in cache.values()]
    weekend_label = _weekend_label(race_dates)

    # 累計実績
    hist_df = load_history()
    c_stats = cumulative_summary(hist_df)
    streak  = hit_streak(hist_df)

    lines: list[str] = []

    # ── タイトル ──────────────────────────────────────────────────
    lines += [
        f"# 【KEIBA EDGE】今週の重賞AI予想 {weekend_label}",
        "",
        "## AIが選ぶ今週の注目レース",
        "",
    ]

    # ── レース別予想 ──────────────────────────────────────────────
    for race_id, r in cache.items():
        race_name   = r.get("race_name", race_id)
        course_info = r.get("course_info", "")
        venue       = r.get("venue", _venue_from_race_id(race_id))

        venue_str  = f"  {venue}"   if venue       else ""
        course_str = f"  {course_info}" if course_info else ""
        lines += [f"### 🏇 {race_name}{venue_str}{course_str}", ""]

        ai_comments: dict = r.get("ai_comments", {})

        # ev マップ（馬番 → ev_score）
        ev_map: dict[int, float] = {
            int(e["horse_number"]): e["ev_score"]
            for e in r.get("ev_top3", [])
            if e.get("horse_number") is not None
        }

        # 予想印（◎○☆）
        for role, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "☆")]:
            p = r.get(role, {})
            if not p or not p.get("horse_name"):
                continue
            num  = p.get("horse_number")
            name = p.get("horse_name", "")
            prob = p.get("prob", 0) * 100
            ev   = ev_map.get(int(num), 0) if num is not None else 0
            ev_str = f"  EV{ev:.2f}" if ev else ""
            lines.append(f"{mark} {num}番 {name}  {prob:.1f}%{ev_str}")

            comment = ai_comments.get(str(num), "")
            if comment:
                lines.append(f"📝 {comment}")
            lines.append("")

        # 危険馬
        dangerous = r.get("dangerous_horses", [])
        for d in dangerous:
            num  = d.get("horse_number", "?")
            name = d.get("horse_name", "")
            pop  = d.get("popularity", "?")
            lines.append(f"⚠️ 危険馬：{num}番 {name}（{pop}番人気）")
            for rsn in d.get("reasons", []):
                lines.append(f"  {rsn}")
        if dangerous:
            lines.append("")

        # 穴馬（ev_top3 の中で predicted_top3_nums 外かつ EV ≥ 1.0 の最上位1頭）
        pred_nums = set(r.get("predicted_top3_nums", []))
        for e in r.get("ev_top3", []):
            enum = e.get("horse_number")
            if enum is None:
                continue
            if int(enum) not in pred_nums and e.get("ev_score", 0) >= 1.0:
                ename = e.get("horse_name", "")
                odds  = e.get("odds", 0)
                lines.append(f"★ 穴馬注目：{enum}番 {ename}  EV{e['ev_score']:.2f}（{odds:.0f}倍）")
                lines.append("")
                break

        # 推奨買い目
        pnums = [n for n in r.get("predicted_top3_nums", []) if n is not None]
        if len(pnums) >= 2:
            # 点数: 複勝1 + 馬連(C(3,2)=3) + 3連複1 = 5
            bet_count = 1 + len(list(combinations(pnums[:3], 2))) + (1 if len(pnums) >= 3 else 0)
            lines += [f"### 💰 推奨買い目（{bet_count}点）", ""]
            lines.append(f"複勝：{pnums[0]}番")
            umaren_combos = " / ".join(
                f"{a}-{b}" for a, b in combinations(pnums[:3], 2)
            )
            lines.append(f"馬連：{umaren_combos}")
            if len(pnums) >= 3:
                others = "/".join(str(n) for n in pnums[1:])
                lines.append(f"3連複：軸{pnums[0]}番 × {others}")
            lines += ["", "---", ""]

    # ── 実績セクション ───────────────────────────────────────────
    lines += ["## 📊 KEIBA EDGE 直近実績", ""]
    if streak >= 1:
        lines.append(f"✅ 重賞{streak}連続複勝的中")
    if c_stats["n_races"] > 0:
        lines.append(f"📈 複勝的中率：{c_stats['fukusho_rate'] * 100:.0f}%")
        lines.append(f"💰 回収率：{c_stats['roi'] * 100:.0f}%")
    else:
        lines.append("（実績データ蓄積中）")
    lines += ["", "---", ""]

    # ── フッター ─────────────────────────────────────────────────
    lines += [
        "※本予想はAIによる分析です。",
        "馬券購入は自己責任でお願いします。",
    ]

    report = "\n".join(lines)

    # 保存
    if output_path is None:
        today_str = date.today().strftime("%Y%m%d")
        output_path = DATA_DIR / f"note_report_{today_str}.md"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"note レポート保存: {output_path}")

    # Discord 送信
    discord_msg = _build_discord_message(cache, weekend_label, c_stats, streak)
    send_note_report_to_discord(discord_msg)

    return report


def _build_discord_message(cache: dict, weekend_label: str, c_stats: dict, streak: int) -> str:
    """Discord 送信用のサマリメッセージを組み立てる。"""
    SEP = "━" * 20
    lines = [
        f"📝 今週のKEIBA EDGE週次レポート",
        SEP,
    ]

    for race_id, r in cache.items():
        race_name = r.get("race_name", race_id)
        honmei = r.get("honmei", {})
        taikou = r.get("taikou", {})
        line = f"🏇 {race_name}"
        if honmei and honmei.get("horse_name"):
            line += f"  ◎{honmei['horse_number']}番{honmei['horse_name']}"
        if taikou and taikou.get("horse_name"):
            line += f"  ○{taikou['horse_number']}番{taikou['horse_name']}"
        lines.append(line)

    lines.append(SEP)

    if streak >= 1:
        lines.append(f"✅ 重賞{streak}連続複勝的中")
    if c_stats.get("n_races", 0) > 0:
        lines.append(
            f"📈 複勝的中率：{c_stats['fukusho_rate'] * 100:.0f}%"
            f"  💰 回収率：{c_stats['roi'] * 100:.0f}%"
        )

    lines += [SEP, "📊 詳細はnoteで公開予定"]
    return "\n".join(lines)


def send_note_report_to_discord(message: str) -> None:
    """DISCORD_REPORT_WEBHOOK_URL にメッセージを送信する。2000字超は自動分割。"""
    url = os.environ.get("DISCORD_REPORT_WEBHOOK_URL")
    if url is None:
        print("[note_report] DISCORD_REPORT_WEBHOOK_URL = None（未設定）→ Discord送信スキップ", flush=True)
        return
    if url == "":
        print("[note_report] DISCORD_REPORT_WEBHOOK_URL = ''（空文字）→ Discord送信スキップ", flush=True)
        return

    print(f"[note_report] Sending to direct URL: {url[:10]}...", flush=True)

    chunks = [message[i: i + 1900] for i in range(0, len(message), 1900)]
    for idx, chunk in enumerate(chunks):
        payload = json.dumps({"content": chunk}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                status = resp.status
            ok = status in (200, 204)
            print(
                f"[note_report] Discord送信 chunk {idx+1}/{len(chunks)}: "
                f"status={status} {'✅ 成功' if ok else '⚠️ 予期しないステータス'}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[note_report] Discord送信 chunk {idx+1}/{len(chunks)}: "
                f"❌ 失敗 {type(exc).__name__}: {exc}",
                flush=True,
            )
            print(traceback.format_exc(), flush=True)


# ── エントリポイント ──────────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(description="note 記事用週次予想レポート生成")
    p.add_argument("--output", type=Path, default=None,
                   help="出力先ファイルパス（省略時: data/note_report_YYYYMMDD.md）")
    args = p.parse_args()
    report = generate_note_report(output_path=args.output)
    print(report)


if __name__ == "__main__":
    main()
