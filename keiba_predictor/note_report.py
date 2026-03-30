"""
KEIBA EDGE 週次予想レポート生成（note 記事用 Markdown）
Claude API で展開予測・馬別解説・期待値分析を生成する。

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

MODEL_ID = "claude-haiku-4-5-20251001"

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


# ── Claude API ───────────────────────────────────────────────────────

def _claude_call(prompt: str, api_key: str) -> str:
    """
    Claude API を呼び出してテキストを返す。失敗時は "" を返す。
    429対策: 初回2秒待機、リトライ10秒/20秒。
    """
    try:
        import anthropic
    except ImportError:
        print("[note_report] anthropic 未インストール → Claude分析スキップ", flush=True)
        return ""

    client = anthropic.Anthropic(api_key=api_key)
    for attempt in range(3):
        wait = 2 if attempt == 0 else 10 * attempt
        time.sleep(wait)
        try:
            print(
                f"[note_report] Claude API call attempt={attempt+1}/3"
                f" model={MODEL_ID!r} prompt_len={len(prompt)}",
                flush=True,
            )
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text
            print(f"[note_report] Claude response {len(response_text)} chars", flush=True)
            if attempt > 0:
                print("[Claude] リトライ成功", flush=True)
            return response_text.strip()
        except Exception as e:
            err_str = str(e).lower()
            print(f"[note_report] Claude error attempt={attempt+1}: {type(e).__name__}: {e}", flush=True)
            if "404" in err_str or "not found" in err_str:
                return ""
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                print(f"[Claude] 429エラー: 60秒待機してリトライ ({attempt+1}/3)", flush=True)
                time.sleep(60)
                if attempt < 2:
                    continue
                print("[Claude] 3回失敗 スキップします", flush=True)
                return ""
            if attempt == 2:
                print("[Claude] 3回失敗 スキップします", flush=True)
                print(traceback.format_exc(), flush=True)
    return ""


def _generate_race_analysis(race_data: dict, race_name: str, course_info: str, api_key: str) -> dict:
    """
    1レース分の Claude 分析を生成する。

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

    raw = _claude_call(prompt, api_key)
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
        print(f"[note_report] Claude JSON解析失敗: {e}  raw={raw[:300]!r}", flush=True)
        return empty


# ── レポート生成 ──────────────────────────────────────────────────────

def _build_note_race_markdown(race_id: str, r: dict, analysis: dict) -> str:
    """1レース分のnote/BOOKERS用プレミアムレポートMarkdownを生成する。"""
    race_name   = r.get("race_name", race_id)
    race_date   = r.get("race_date", "")
    course_info = r.get("course_info", "")
    venue       = r.get("venue", _venue_from_race_id(race_id))
    grade       = _grade_from_name(race_name)
    grade_str   = f" {grade}" if grade else ""

    ev_map: dict[int, float] = {
        int(e["horse_number"]): e["ev_score"]
        for e in r.get("ev_top3", [])
        if e.get("horse_number") is not None
    }
    sim = r.get("simulation", {})
    ai_comments = r.get("ai_comments", {})

    # モデル精度情報を取得
    auc_str = "N/A"
    fukusho_str = "N/A"
    try:
        import pickle
        model_path = Path(__file__).parent / "model" / "xgb_model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            auc_str = f"{bundle.get('cv_auc_mean', 0):.4f}"
            fukusho_str = f"{bundle.get('cv_fukusho_mean', 0) * 100:.1f}"
    except Exception:
        pass

    # 出走頭数
    top5_data = r.get("predicted_top5", [])
    n_horses = len(top5_data) if top5_data else "?"

    lines: list[str] = []

    # ── タイトル ─────────────────────────────────────────────
    lines += [
        f"# 【{race_name}{grade_str}】KEIBA EDGE プレミアム予想 {race_date}",
        "",
        "> ⚠️ 本予想はAIによる分析です。馬券購入は自己責任でお願いします。",
        "",
    ]

    # ── レース概要 ───────────────────────────────────────────
    lines += [
        "## 📊 レース概要",
        f"- 開催：{race_date} {venue} {course_info}",
        f"- 出走頭数：{n_horses}頭",
        f"- AIモデル精度：AUC {auc_str} / 複勝的中率 {fukusho_str}%",
        "",
    ]

    # ── AI予想印テーブル ─────────────────────────────────────
    lines += [
        "## 🏆 AI予想印",
        "",
        "| 印 | 馬番 | 馬名 | AI確率 | EVスコア | 判定 |",
        "|---|---|---|---|---|---|",
    ]

    roles_data: list[dict] = []
    for role, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "▲")]:
        p = r.get(role, {})
        if not p or not p.get("horse_name"):
            continue
        num  = p.get("horse_number", "?")
        name = p.get("horse_name", "")
        prob = p.get("prob", 0) * 100
        ev   = ev_map.get(int(num), 0) if num is not None else 0
        ev_str = f"{ev:.2f}" if ev else "—"

        # 安定軸 / 展開依存 判定
        mc = sim.get(str(num), {})
        if mc and mc.get("is_stable") is not None:
            tag = "🔒安定軸" if mc["is_stable"] else "⚡展開依存"
        elif prob >= 50:
            tag = "🔒安定軸"
        else:
            tag = "⚡展開依存"

        lines.append(f"| {mark} | {num}番 | {name} | {prob:.1f}% | {ev_str} | {tag} |")
        roles_data.append({
            "role": role, "mark": mark, "num": num, "name": name,
            "prob": prob, "ev": ev, "tag": tag, "mc": mc,
        })

    lines.append("")

    # ── 注目馬詳細分析 ───────────────────────────────────────
    lines += ["## 🔍 注目馬詳細分析", ""]

    for rd in roles_data:
        num, name, mark = rd["num"], rd["name"], rd["mark"]
        prob, ev, tag = rd["prob"], rd["ev"], rd["tag"]
        mc = rd["mc"]

        lines.append(f"### {mark} {num}番 {name}")
        lines.append(f"**AI確率{prob:.1f}% / EVスコア{ev:.2f} / {tag}**")
        lines.append("")

        # 【強み】: SHAP プラス要因
        shap_top = None
        for role_key in ["honmei", "taikou", "ana"]:
            p = r.get(role_key, {})
            if p.get("horse_number") == num and p.get("shap_top"):
                shap_top = p["shap_top"]
                break

        strengths = []
        concerns = []
        if shap_top:
            strengths = [s["label"] for s in shap_top if s.get("value", 0) > 0]
            concerns = [s["label"] for s in shap_top if s.get("value", 0) < 0]

        lines.append("**【強み】**")
        if strengths:
            for s in strengths:
                lines.append(f"- {s}")
        else:
            lines.append(f"- AI確率{prob:.1f}%で上位評価")
        lines.append("")

        # 【モンテカルロ分析】
        if mc:
            hi = mc.get("scenario", {}).get("high_pace", 0) * 100
            sl = mc.get("scenario", {}).get("slow_pace", 0) * 100
            lines += [
                "**【モンテカルロ分析（1万回シミュレーション）】**",
                f"- ハイペース時：3着以内{hi:.0f}%",
                f"- スローペース時：3着以内{sl:.0f}%",
                "",
            ]

        # 【AI解説】
        comment = (
            analysis.get("horses", {}).get(str(num))
            or ai_comments.get(str(num), "")
        )
        if comment:
            lines += ["**【AI解説】**", comment, ""]

        # 【SHAP要因分析】
        if shap_top:
            lines.append("**【SHAP要因分析】**")
            if strengths:
                lines.append(f"✅ プラス要因：{'、'.join(strengths)}")
            if concerns:
                lines.append(f"⚠️ マイナス要因：{'、'.join(concerns)}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # ── 危険馬分析 ───────────────────────────────────────────
    dangerous = r.get("dangerous_horses", [])
    if dangerous:
        lines += ["## ⚠️ 危険馬分析", ""]
        for d in dangerous:
            dnum    = d.get("horse_number", "?")
            dname   = d.get("horse_name", "")
            dpop    = d.get("popularity", "?")
            dprob   = d.get("prob", 0) * 100
            reasons = d.get("reasons", [])
            lines.append(f"### {dnum}番 {dname}（{dpop}番人気）")
            for rsn in reasons:
                lines.append(f"- 理由：{rsn}")
            lines.append(f"- AI確率：{dprob:.1f}%（人気に対して低い）")
            lines.append("")

    # ── 穴馬候補 ─────────────────────────────────────────────
    ana_info = r.get("ana_horse_info", {})
    ana_num = r.get("ana_horse_num")
    if ana_num and ana_info.get("horse_name"):
        a_name = ana_info["horse_name"]
        a_prob = ana_info.get("prob", 0) * 100
        a_pop  = ana_info.get("popularity", "?")
        a_ev   = ev_map.get(ana_num, 0)
        lines += [
            "## 🎯 穴馬候補",
            "",
            f"### ★ {ana_num}番 {a_name}（{a_pop}番人気）",
            f"- AI確率：{a_prob:.1f}%（{a_pop}番人気なのに高評価）",
            f"- EVスコア：{a_ev:.2f}" if a_ev else "",
        ]
        # モンテカルロの激走条件
        a_mc = sim.get(str(ana_num), {})
        if a_mc:
            hi = a_mc.get("scenario", {}).get("high_pace", 0) * 100
            sl = a_mc.get("scenario", {}).get("slow_pace", 0) * 100
            better = "ハイペース" if hi > sl else "スローペース"
            better_pct = max(hi, sl)
            lines.append(f"- 激走条件：{better}で確率{better_pct:.0f}%まで上昇")
        lines.append("")

    # ── AI自動決定買い目 ─────────────────────────────────────
    bs = r.get("bet_strategy", {})
    if bs and bs.get("total_points", 0) > 0:
        total = bs["total_points"]
        lines += [
            f"## 💰 AI自動決定買い目",
            "",
            "| 券種 | 買い目 | 点数 |",
            "|---|---|---|",
        ]
        if bs.get("fukusho"):
            f = bs["fukusho"][0]
            lines.append(f"| 複勝 | {f['num']}番 {f.get('name', '')} | 1点 |")

        if bs.get("use_wide") and bs.get("wide"):
            wide_str = " / ".join(f"{w['nums'][0]}-{w['nums'][1]}" for w in bs["wide"])
            lines.append(f"| ワイド | {wide_str} | {len(bs['wide'])}点 |")
        if bs.get("umaren"):
            umaren_str = " / ".join(f"{u['nums'][0]}-{u['nums'][1]}" for u in bs["umaren"])
            lines.append(f"| 馬連 | {umaren_str} | {len(bs['umaren'])}点 |")

        sr = bs.get("sanrenpuku", {})
        if sr and sr.get("jiku") and sr.get("aite"):
            jiku = sr["jiku"]
            aite = sr["aite"]
            if len(jiku) == 1:
                sr_pt = len(list(combinations(aite, 2)))
                lines.append(f"| 3連複 | 軸{jiku[0]} × {'/'.join(str(n) for n in aite)} | {sr_pt}点 |")
            elif len(jiku) == 2:
                sr_pt = len(aite)
                lines.append(f"| 3連複 | 軸{jiku[0]}-{jiku[1]} × {'/'.join(str(n) for n in aite)} | {sr_pt}点 |")

        lines += [
            "",
            f"合計{total}点 / 想定投資額{total * 100}円",
            "",
        ]
        if bs.get("strategy_note"):
            lines.append(f"💡 {bs['strategy_note']}")
            lines.append("")
    else:
        # フォールバック: 従来の固定買い目
        pnums = [n for n in r.get("predicted_top5_nums", r.get("predicted_top3_nums", [])) if n is not None]
        if len(pnums) >= 2:
            hon = pnums[0]
            umaren_pairs = list(combinations(pnums[:3], 2))
            umaren_str = " / ".join(f"{a}-{b}" for a, b in umaren_pairs)
            partners = pnums[1:5]
            sanren_pt = len(list(combinations(partners, 2)))
            total = 1 + len(umaren_pairs) + sanren_pt
            lines += [
                f"## 💰 推奨買い目（合計{total}点）",
                "",
                "| 券種 | 買い目 | 点数 |",
                "|---|---|---|",
                f"| 複勝 | {hon}番 | 1点 |",
                f"| 馬連 | {umaren_str} | {len(umaren_pairs)}点 |",
                f"| 3連複 | 軸{hon} × {'/'.join(str(n) for n in partners)} | {sanren_pt}点 |",
                "",
            ]

    # ── 期待値分析 ───────────────────────────────────────────
    lines += ["## 📈 期待値分析", ""]
    if analysis.get("ev_analysis"):
        lines.append(analysis["ev_analysis"])
    else:
        for e in r.get("ev_top3", []):
            ev   = e.get("ev_score", 0)
            prob = e.get("prob", 0) * 100
            name = e.get("horse_name", "")
            num  = e.get("horse_number", "?")
            tag  = "★妙味あり" if ev >= 3.0 else ""
            lines.append(f"- {num}番 {name}：EV{ev:.2f}（AI確率{prob:.1f}%）{tag}")
    lines.append("")

    # ── フッター ─────────────────────────────────────────────
    lines += [
        "---",
        f"*KEIBA EDGE — XGBoost({auc_str}) × Claude AIによる競馬予想システム*",
        "*ダッシュボード：https://yssohrrs206038s-oss.github.io/keiba-predictor/*",
    ]

    return "\n".join(lines)


def generate_note_report(output_path: Optional[Path] = None) -> str:
    """
    predictions_cache.json と results_history.csv から note 記事用 Markdown を生成する。
    ANTHROPIC_API_KEY が設定されていれば Claude で展開予測・馬別解説・期待値分析も生成する。

    Returns:
        生成した Markdown 文字列
    """
    cache = _load_cache()
    if not cache:
        raise ValueError("予想キャッシュが空です。先に notify --mode predict を実行してください。")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    all_reports: list[str] = []

    for race_id, r in cache.items():
        race_name   = r.get("race_name", race_id)
        course_info = r.get("course_info", "")

        # キャッシュの ai_comments から analysis を構築（APIコール不要）
        ai_comments = r.get("ai_comments", {})
        if ai_comments:
            logger.info(f"[note_report] {race_name}: キャッシュのAI解説を使用（{len(ai_comments)}頭分）")
            analysis = {"tenkai": "", "horses": ai_comments, "ev_analysis": ""}
        else:
            # ai_comments が空の場合のみ API コールでフォールバック
            logger.info(f"[note_report] {race_name}: AI解説なし → API呼び出しでフォールバック")
            analysis = _generate_race_analysis(r, race_name, course_info, api_key)

        report = _build_note_race_markdown(race_id, r, analysis)
        all_reports.append(report)

    full_report = "\n\n".join(all_reports)

    # 保存
    if output_path is None:
        today_file = date.today().strftime("%Y%m%d")
        output_path = DATA_DIR / f"note_report_{today_file}.md"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_report, encoding="utf-8")
    logger.info(f"note レポート保存: {output_path}")

    # Discord 送信（レースごとに個別メッセージ）
    send_discord_per_race(cache)

    return full_report


# ── Discord 送信 ──────────────────────────────────────────────────────

def _build_race_discord_message(race_id: str, r: dict) -> str:
    """レース1件分の Discord 送信メッセージをnote風Markdownで組み立てる。"""
    from itertools import combinations as _comb

    race_name   = r.get("race_name", race_id)
    race_date   = r.get("race_date", "")
    course_info = r.get("course_info", "")
    grade       = _grade_from_name(race_name)
    grade_str   = f" {grade}" if grade else ""

    ev_map: dict[int, float] = {
        int(e["horse_number"]): e["ev_score"]
        for e in r.get("ev_top3", [])
        if e.get("horse_number") is not None
    }

    lines = [
        f"# 【{race_name}{grade_str}】KEIBA EDGE AI予想 {race_date}",
        "",
        "> ⚠️ 本予想はAIによる分析です。馬券購入は自己責任でお願いします。",
        "",
        "## 📊 AIスコア上位馬",
        "",
        "| 印 | 馬番 | 馬名 | AI確率 | EVスコア |",
        "|---|---|---|---|---|",
    ]

    for role, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "▲")]:
        p = r.get(role, {})
        if not p or not p.get("horse_name"):
            continue
        num  = p.get("horse_number", "?")
        name = p.get("horse_name", "")
        prob = p.get("prob", 0) * 100
        ev   = ev_map.get(int(num), 0) if num is not None else 0
        ev_str = f"{ev:.2f}" if ev else "—"
        lines.append(f"| {mark} | {num}番 | {name} | {prob:.1f}% | {ev_str} |")

    lines.append("")

    # 注目馬AI解説
    ai_comments = r.get("ai_comments", {})
    if ai_comments:
        lines += ["## 🔍 注目馬AI解説", ""]
        for role, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "▲")]:
            p = r.get(role, {})
            if not p or not p.get("horse_name"):
                continue
            num     = p.get("horse_number")
            name    = p.get("horse_name", "")
            comment = ai_comments.get(str(num), "")
            lines.append(f"### {mark} {num}番 {name}")
            if comment:
                lines.append(comment)
            lines.append("")

    # 危険馬
    dangerous = r.get("dangerous_horses", [])
    if dangerous:
        lines += ["## ⚠️ 危険馬", ""]
        for d in dangerous:
            dnum    = d.get("horse_number", "?")
            dname   = d.get("horse_name", "")
            dpop    = d.get("popularity", "?")
            reasons = d.get("reasons", [])
            reason  = reasons[0] if reasons else "要注意"
            lines.append(f"- **{dnum}番 {dname}**（{dpop}番人気）— {reason}")
        lines.append("")

    # 買い目
    pnums = [n for n in (r.get("predicted_top5_nums") or r.get("predicted_top3_nums", [])) if n is not None]
    if len(pnums) >= 2:
        hon = pnums[0]
        umaren_pairs = list(_comb(pnums[:3], 2))
        umaren_str = " / ".join(f"{a}-{b}" for a, b in umaren_pairs)

        partners = pnums[1:5]
        cached_ana_num = r.get("ana_horse_num")
        if cached_ana_num and cached_ana_num not in partners:
            partners = partners + [cached_ana_num]
        sanren_pt = len(list(_comb(partners, 2)))
        partners_str = "/".join(str(n) for n in partners)
        total = 1 + len(umaren_pairs) + sanren_pt

        lines += [
            f"## 💰 推奨買い目（合計{total}点）",
            "",
            "| 券種 | 買い目 | 点数 |",
            "|---|---|---|",
            f"| 複勝 | {hon}番 | 1点 |",
            f"| 馬連 | {umaren_str} | {len(umaren_pairs)}点 |",
            f"| 3連複 | 軸{hon} × {partners_str} | {sanren_pt}点 |",
            "",
        ]

    # 期待値分析
    best_ev = r.get("ev_top3", [{}])[0] if r.get("ev_top3") else {}
    ev_val  = best_ev.get("ev_score", 0)
    prob    = best_ev.get("prob", 0) * 100
    if ev_val:
        lines += [
            "## 📈 期待値分析",
            f"EVスコア{ev_val:.2f}は期待値投資として成立。"
            f"AI確率{prob:.1f}%の信頼度から、複勝・馬連での安定回収が見込める。",
            "",
        ]

    lines += [
        "---",
        "*KEIBA EDGE — XGBoost × Claude AIによる競馬予想システム*",
        "*詳細はダッシュボードへ👉 https://yssohrrs206038s-oss.github.io/keiba-predictor/*",
    ]

    return "\n".join(lines)


def send_discord_per_race(cache: dict) -> None:
    """レースごとに個別メッセージを DISCORD_REPORT_WEBHOOK_URL に送信する。"""
    url = os.environ.get("DISCORD_REPORT_WEBHOOK_URL")
    if url is None:
        print("[note_report] DISCORD_REPORT_WEBHOOK_URL = None（未設定）→ Discord送信スキップ", flush=True)
        return
    if url == "":
        print("[note_report] DISCORD_REPORT_WEBHOOK_URL = ''（空文字）→ Discord送信スキップ", flush=True)
        return

    print(f"[note_report] Sending to direct URL: {url[:10]}...", flush=True)
    from keiba_predictor.discord_notify import send_discord

    for race_id, r in cache.items():
        race_name = r.get("race_name", race_id)
        msg = _build_race_discord_message(race_id, r)
        ok  = send_discord(url, msg)
        print(f"[note_report] {race_name} Discord送信{'✅ 成功' if ok else '❌ 失敗'}", flush=True)


# ── エントリポイント ──────────────────────────────────────────────────

def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(description="note 記事用週次予想レポート生成（Claude AI 解説付き）")
    p.add_argument("--output", type=Path, default=None,
                   help="出力先ファイルパス（省略時: data/note_report_YYYYMMDD.md）")
    args = p.parse_args()
    report = generate_note_report(output_path=args.output)
    print(report)


if __name__ == "__main__":
    main()
