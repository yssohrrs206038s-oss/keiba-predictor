"""
KEIBA EDGE 週次予想レポート生成（note 記事用 Markdown）
Gemini API で展開予測・馬別解説・期待値分析を生成する。

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
import time
import traceback
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

MODEL_ID = "gemini-2.0-flash"

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
    first = parsed[0]
    wd = first.weekday()
    sat = first - timedelta(days=(wd - 5) % 7) if wd != 5 else first
    sun = sat + timedelta(days=1)
    return f"{sat.year}/{sat.month:02d}/{sat.day:02d}-{sun.day:02d}"


def _extract_json_object(text: str) -> str:
    """レスポンステキストから JSON オブジェクト部分を抽出する。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return text[start:]


# ── Gemini API ───────────────────────────────────────────────────────

def _gemini_call(prompt: str, api_key: str) -> str:
    """
    Gemini API を呼び出してテキストを返す。失敗時は "" を返す。
    429対策: 初回2秒待機、リトライ10秒/20秒。
    """
    try:
        from google import genai
    except ImportError:
        print("[note_report] google-genai 未インストール → Gemini分析スキップ", flush=True)
        return ""

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1"})
    for attempt in range(3):
        wait = 2 if attempt == 0 else 10 * attempt
        time.sleep(wait)
        try:
            print(
                f"[note_report] Gemini API call attempt={attempt+1}/3"
                f" model={MODEL_ID!r} prompt_len={len(prompt)}",
                flush=True,
            )
            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            print(f"[note_report] Gemini response {len(response.text)} chars", flush=True)
            return response.text.strip()
        except Exception as e:
            err_str = str(e).lower()
            print(f"[note_report] Gemini error attempt={attempt+1}: {type(e).__name__}: {e}", flush=True)
            if "404" in err_str or "not found" in err_str:
                return ""
            if attempt == 2:
                print(traceback.format_exc(), flush=True)
    return ""


def _generate_race_analysis(race_data: dict, race_name: str, course_info: str, api_key: str) -> dict:
    """
    1レース分の Gemini 分析を生成する。

    Returns:
        {
          "tenkai": "展開予測テキスト（300文字以内）",
          "horses": {"馬番str": "個別解説（200文字以内）", ...},
          "ev_analysis": "期待値分析（200文字以内）"
        }
        失敗時は {"tenkai": "", "horses": {}, "ev_analysis": ""}
    """
    empty: dict = {"tenkai": "", "horses": {}, "ev_analysis": ""}
    if not api_key:
        return empty

    ev_top3 = race_data.get("ev_top3", [])

    # 上位馬情報を組み立て
    horses_info = []
    for role, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "▲")]:
        p = race_data.get(role, {})
        if not p or not p.get("horse_name"):
            continue
        ev_val = next(
            (e.get("ev_score", 0) for e in ev_top3
             if str(e.get("horse_number")) == str(p.get("horse_number"))),
            0,
        )
        horses_info.append({
            "印": mark,
            "馬番": p.get("horse_number"),
            "馬名": p.get("horse_name"),
            "AI3着以内確率": f"{p.get('prob', 0) * 100:.1f}%",
            "EVスコア": f"{ev_val:.2f}",
        })

    # 穴馬追加
    pred_set = set(race_data.get("predicted_top3_nums", []))
    for e in ev_top3:
        enum = e.get("horse_number")
        if enum and int(enum) not in pred_set and e.get("ev_score", 0) >= 1.0:
            horses_info.append({
                "印": "穴🚀",
                "馬番": enum,
                "馬名": e.get("horse_name", ""),
                "AI3着以内確率": f"{e.get('prob', 0) * 100:.1f}%",
                "EVスコア": f"{e.get('ev_score', 0):.2f}",
                "オッズ": f"{e.get('odds', 0):.0f}倍",
            })
            break

    if not horses_info:
        return empty

    race_label = f"{race_name}（{course_info}）" if course_info else race_name

    prompt = (
        f"あなたは競馬予測AIの解説ライターです。\n"
        f"以下は「{race_label}」の予測データです。\n\n"
        f"対象馬データ（JSON）:\n"
        f"{json.dumps(horses_info, ensure_ascii=False, indent=2)}\n\n"
        f"以下の3項目を生成してください。\n\n"
        f"1. **tenkai**（展開予測、300文字以内）\n"
        f"   各馬の逃げ・先行・差し・追い込みの位置取りを予測し、"
        f"ペース想定と有利な脚質を馬名を挙げて解説。\n\n"
        f"2. **horses**（馬番をキーとした個別解説、各200文字以内）\n"
        f"   血統・騎手・前走・展開面を踏まえて解説。"
        f"◎は推奨理由を力強く、穴馬は激走ポイントを強調。"
        f"危険と判断した馬には忖度なしの毒舌も可。\n\n"
        f"3. **ev_analysis**（期待値分析の総括、200文字以内）\n"
        f"   なぜこのレースでこの馬を買うべきか、期待値と回収率の観点でまとめる。\n\n"
        f"必ず以下のJSONのみを返してください（コードブロック・前後の説明文不要）:\n"
        f'{{"tenkai": "...", "horses": {{"1": "...", "3": "..."}}, "ev_analysis": "..."}}\n'
    )

    raw = _gemini_call(prompt, api_key)
    if not raw:
        return empty

    try:
        json_str = _extract_json_object(raw)
        data = json.loads(json_str)
        return {
            "tenkai":      str(data.get("tenkai", ""))[:300],
            "horses":      {str(k): str(v)[:200] for k, v in data.get("horses", {}).items()},
            "ev_analysis": str(data.get("ev_analysis", ""))[:200],
        }
    except Exception as e:
        print(f"[note_report] Gemini JSON解析失敗: {e}  raw={raw[:300]!r}", flush=True)
        return empty


# ── レポート生成 ──────────────────────────────────────────────────────

def generate_note_report(output_path: Optional[Path] = None) -> str:
    """
    predictions_cache.json と results_history.csv から note 記事用 Markdown を生成する。
    GEMINI_API_KEY が設定されていれば Gemini で展開予測・馬別解説・期待値分析も生成する。

    Returns:
        生成した Markdown 文字列
    """
    cache = _load_cache()
    if not cache:
        raise ValueError("予想キャッシュが空です。先に notify --mode predict を実行してください。")

    api_key   = os.environ.get("GEMINI_API_KEY", "")
    today_str = date.today().strftime("%Y/%m/%d")

    race_dates    = [r.get("race_date", "") for r in cache.values()]
    weekend_label = _weekend_label(race_dates)

    hist_df = load_history()
    c_stats = cumulative_summary(hist_df)
    streak  = hit_streak(hist_df)

    lines: list[str] = []

    # ── タイトル ──────────────────────────────────────────────────
    lines += [
        f"# 【KEIBA EDGE】今週の重賞AI予想レポート {today_str}",
        "",
    ]

    # ── 今週のレース概要 ──────────────────────────────────────────
    lines += ["## 今週のレース概要", ""]
    for race_id, r in cache.items():
        race_name  = r.get("race_name", race_id)
        honmei     = r.get("honmei", {})
        grade      = _grade_from_name(race_name)
        grade_str  = f"（{grade}）" if grade else ""
        honmei_str = ""
        if honmei and honmei.get("horse_name"):
            honmei_str = f"　◎{honmei['horse_number']}番 {honmei['horse_name']}"
        lines.append(f"- 🏇 {race_name}{grade_str}{honmei_str}")
    lines += ["", "---", ""]

    # ── レース別詳細分析 ──────────────────────────────────────────
    for race_id, r in cache.items():
        race_name   = r.get("race_name", race_id)
        course_info = r.get("course_info", "")
        venue       = r.get("venue", _venue_from_race_id(race_id))
        grade       = _grade_from_name(race_name)

        grade_str  = f"（{grade}）" if grade else ""
        venue_str  = f"  {venue}"       if venue       else ""
        course_str = f"  {course_info}" if course_info else ""

        lines += [f"### 🏇 {race_name}{grade_str}{venue_str}{course_str}", ""]

        # Gemini 分析生成
        print(f"[note_report] {race_name} の Gemini分析を生成中...", flush=True)
        analysis = _generate_race_analysis(r, race_name, course_info, api_key)

        ev_map: dict[int, float] = {
            int(e["horse_number"]): e["ev_score"]
            for e in r.get("ev_top3", [])
            if e.get("horse_number") is not None
        }
        pred_set = set(r.get("predicted_top3_nums", []))

        # ── AIスコア上位馬 ──────────────────────────────────────
        lines += ["#### AIスコア上位馬", ""]
        for role, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "▲")]:
            p = r.get(role, {})
            if not p or not p.get("horse_name"):
                continue
            num  = p.get("horse_number")
            name = p.get("horse_name", "")
            prob = p.get("prob", 0) * 100
            ev   = ev_map.get(int(num), 0) if num is not None else 0
            ev_str = f"  EV{ev:.2f}" if ev else ""
            lines.append(f"{mark} {num}番 **{name}**　AI確率 {prob:.1f}%{ev_str}")

        for e in r.get("ev_top3", []):
            enum = e.get("horse_number")
            if enum and int(enum) not in pred_set and e.get("ev_score", 0) >= 1.0:
                lines.append(
                    f"穴🚀 {enum}番 **{e.get('horse_name', '')}**"
                    f"　EV{e['ev_score']:.2f}（{e.get('odds', 0):.0f}倍）"
                )
                break
        lines.append("")

        # ── 展開予測 ────────────────────────────────────────────
        lines += ["#### 展開予測", ""]
        lines.append(analysis["tenkai"] if analysis["tenkai"] else "（展開予測データなし）")
        lines.append("")

        # ── 注目馬 Gemini解説 ─────────────────────────────────
        lines += ["#### 注目馬 Gemini解説", ""]
        for role, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "▲")]:
            p = r.get(role, {})
            if not p or not p.get("horse_name"):
                continue
            num     = p.get("horse_number")
            name    = p.get("horse_name", "")
            comment = (
                analysis["horses"].get(str(num))
                or r.get("ai_comments", {}).get(str(num), "")
            )
            lines += [f"**{mark} {num}番 {name}**", ""]
            if comment:
                lines.append(comment)
            lines.append("")

        # 穴馬解説
        for e in r.get("ev_top3", []):
            enum = e.get("horse_number")
            if enum and int(enum) not in pred_set and e.get("ev_score", 0) >= 1.0:
                ename   = e.get("horse_name", "")
                comment = (
                    analysis["horses"].get(str(enum))
                    or r.get("ai_comments", {}).get(str(enum), "")
                )
                lines += [f"**穴🚀 {enum}番 {ename}**", ""]
                if comment:
                    lines.append(comment)
                lines.append("")
                break

        # 危険馬
        dangerous = r.get("dangerous_horses", [])
        if dangerous:
            lines += ["#### ⚠️ 危険馬", ""]
            for d in dangerous:
                dnum  = d.get("horse_number", "?")
                dname = d.get("horse_name", "")
                dpop  = d.get("popularity", "?")
                lines.append(f"**{dnum}番 {dname}**（{dpop}番人気）")
                for rsn in d.get("reasons", []):
                    lines.append(f"- {rsn}")
            lines.append("")

        # ── 期待値分析 ────────────────────────────────────────
        lines += ["#### 期待値分析", ""]
        if analysis["ev_analysis"]:
            lines.append(analysis["ev_analysis"])
        else:
            for e in r.get("ev_top3", []):
                ev   = e.get("ev_score", 0)
                prob = e.get("prob", 0) * 100
                odds = e.get("odds", 0)
                mark = "★ EV+" if ev >= 1.0 else "　"
                lines.append(
                    f"{mark} {e.get('horse_number')}番 {e.get('horse_name', '')}"
                    f"　EV{ev:.2f}（確率{prob:.1f}% / {odds:.1f}倍）"
                )
        lines.append("")

        # ── 買い目 ────────────────────────────────────────────
        pnums = [n for n in r.get("predicted_top3_nums", []) if n is not None]
        if len(pnums) >= 2:
            bet_count = 1 + len(list(combinations(pnums[:3], 2))) + (1 if len(pnums) >= 3 else 0)
            lines += [f"#### 💰 買い目（{bet_count}点）", ""]
            lines.append(f"**複勝**：{pnums[0]}番")
            umaren_combos = " / ".join(f"{a}-{b}" for a, b in combinations(pnums[:3], 2))
            lines.append(f"**馬連**：{umaren_combos}")
            if len(pnums) >= 3:
                others = "/".join(str(n) for n in pnums[1:])
                lines.append(f"**3連複**：軸{pnums[0]}番 × {others}")
            lines.append("")

        lines += ["---", ""]

    # ── 的中実績・回収率サマリー ──────────────────────────────────
    lines += ["## 📊 的中実績・回収率サマリー", ""]
    if streak >= 1:
        lines.append(f"✅ 重賞{streak}連続複勝的中")
    if c_stats["n_races"] > 0:
        lines += [
            f"📈 複勝的中率：{c_stats['fukusho_rate'] * 100:.0f}%",
            f"💰 累計回収率：{c_stats['roi'] * 100:.0f}%",
            f"📋 通算 {c_stats['n_races']} レース分析済み",
        ]
    else:
        lines.append("（実績データ蓄積中）")
    lines += ["", "---", ""]

    # ── KEIBA EDGEについて ────────────────────────────────────────
    lines += [
        "## KEIBA EDGEについて",
        "",
        "KEIBA EDGEは機械学習（XGBoost）と Gemini AIを組み合わせた重賞予想システムです。",
        "- 過去の出走データ・騎手成績・コース適性を学習した独自モデルで3着以内確率を算出",
        "- 期待値（EV）スコアにより「オッズと確率のギャップ」がある穴馬を発見",
        "- Gemini AIが展開予測・血統分析・個別解説を生成",
        "",
        "※本予想はAIによる分析です。馬券購入は自己責任でお願いします。",
    ]

    report = "\n".join(lines)

    # 保存
    if output_path is None:
        today_file = date.today().strftime("%Y%m%d")
        output_path = DATA_DIR / f"note_report_{today_file}.md"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"note レポート保存: {output_path}")

    # Discord 送信
    discord_msg = _build_discord_message(cache, weekend_label, c_stats, streak)
    send_note_report_to_discord(discord_msg)

    return report


# ── Discord 送信 ──────────────────────────────────────────────────────

def _build_discord_message(cache: dict, weekend_label: str, c_stats: dict, streak: int) -> str:
    """Discord 送信用のサマリメッセージを組み立てる。"""
    SEP = "━" * 20
    lines = ["📝 今週のKEIBA EDGE週次レポート", SEP]

    for race_id, r in cache.items():
        race_name = r.get("race_name", race_id)
        honmei    = r.get("honmei", {})
        taikou    = r.get("taikou", {})
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
    """DISCORD_REPORT_WEBHOOK_URL にメッセージを送信する。discord_notify.send_discord() を使用。"""
    url = os.environ.get("DISCORD_REPORT_WEBHOOK_URL")
    if url is None:
        print("[note_report] DISCORD_REPORT_WEBHOOK_URL = None（未設定）→ Discord送信スキップ", flush=True)
        return
    if url == "":
        print("[note_report] DISCORD_REPORT_WEBHOOK_URL = ''（空文字）→ Discord送信スキップ", flush=True)
        return

    print(f"[note_report] Sending to direct URL: {url[:10]}...", flush=True)
    from keiba_predictor.discord_notify import send_discord
    ok = send_discord(url, message)
    print(f"[note_report] Discord送信{'✅ 成功' if ok else '❌ 失敗'}", flush=True)


# ── エントリポイント ──────────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(description="note 記事用週次予想レポート生成（Gemini AI 解説付き）")
    p.add_argument("--output", type=Path, default=None,
                   help="出力先ファイルパス（省略時: data/note_report_YYYYMMDD.md）")
    args = p.parse_args()
    report = generate_note_report(output_path=args.output)
    print(report)


if __name__ == "__main__":
    main()
