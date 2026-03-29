"""
的中実績の記録・集計・レポート生成モジュール

results_history.csv スキーマ:
  date, race_id, race_name, race_grade,
  pred1_name, pred1_num, pred1_prob,
  pred2_name, pred2_num, pred2_prob,
  pred3_name, pred3_num, pred3_prob,
  actual1_name, actual1_num,
  actual2_name, actual2_num,
  actual3_name, actual3_num,
  fukusho_hit,
  umaren_hit, umaren_payout,
  wide_hit, wide_payout,
  sanrenpuku_hit, sanrenpuku_payout,
  bet_total, return_total
"""

import logging
import re
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR      = Path(__file__).parent / "data"
HISTORY_PATH  = DATA_DIR / "results_history.csv"
REPORTS_DIR   = DATA_DIR / "reports"

# 1 口 100 円・1 レースあたりの買い目: 複勝1+馬連3+3連複10 = 14 口
UNIT_BET  = 100
BETS_PER_RACE = 14

HISTORY_COLS = [
    "date", "race_id", "race_name", "race_grade",
    "pred1_name", "pred1_num", "pred1_prob",
    "pred2_name", "pred2_num", "pred2_prob",
    "pred3_name", "pred3_num", "pred3_prob",
    "actual1_name", "actual1_num",
    "actual2_name", "actual2_num",
    "actual3_name", "actual3_num",
    "fukusho_hit",
    "umaren_hit",     "umaren_payout",
    "wide_hit",       "wide_payout",
    "sanrenpuku_hit", "sanrenpuku_payout",
    "bet_total",      "return_total",
]

# レース格パターン（feature_engineering.py と同じロジック）
_GRADE_PATTERNS = [
    (re.compile(r"[（(]G\s*[1Ⅰ][）)]|[（(]GI[）)]",  re.I), "G1"),
    (re.compile(r"[（(]G\s*[2Ⅱ][）)]|[（(]GII[）)]",  re.I), "G2"),
    (re.compile(r"[（(]G\s*[3Ⅲ][）)]|[（(]GIII[）)]", re.I), "G3"),
    (re.compile(r"[（(]L[）)]|オープン|（OP）|\(OP\)",  re.I), "OP"),
    (re.compile(r"3勝クラス|1600万"),                         "3勝"),
    (re.compile(r"2勝クラス|1000万|900万"),                   "2勝"),
]


def _grade_label(race_name: str) -> str:
    if not isinstance(race_name, str):
        return "-"
    for pat, label in _GRADE_PATTERNS:
        if pat.search(race_name):
            return label
    return "1勝以下"


def _payout_str_to_int(s: str) -> int:
    """'¥1,450' / '1450' → 1450, 空文字 → 0"""
    if not s:
        return 0
    try:
        return int(re.sub(r"[¥,\s円]", "", s))
    except ValueError:
        return 0


def _top3_actual(actual_df: pd.DataFrame) -> list[dict]:
    """actual_df から 1〜3 着馬の [{'name': ..., 'num': ...}, ...] を返す。"""
    df = actual_df.copy()
    df["_fp"] = pd.to_numeric(df["finish_position"], errors="coerce")
    top3 = (
        df[df["_fp"].isin([1, 2, 3])]
        .sort_values("_fp")
        .head(3)
    )
    result = []
    for _, row in top3.iterrows():
        result.append({
            "name": str(row.get("horse_name", "")),
            "num":  int(row["horse_number"]) if pd.notna(row.get("horse_number")) else 0,
        })
    return result


def _pred_row(pred: dict, role: str) -> dict:
    """キャッシュ dict から pred1/2/3 用の {name, num, prob} を返す。"""
    p = pred.get(role, {})
    return {
        "name": p.get("horse_name", ""),
        "num":  p.get("horse_number") or 0,
        "prob": round(float(p.get("prob", 0.0)) * 100, 1),
    }


# ══════════════════════════════════════════════════════════════
# 公開 API
# ══════════════════════════════════════════════════════════════

def load_history() -> pd.DataFrame:
    """results_history.csv を読み込む。ファイルがなければ空の DataFrame を返す。"""
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLS)
    df = pd.read_csv(HISTORY_PATH, encoding="utf-8-sig", dtype=str)

    # ── 旧カラム名 → 新カラム名へのリネーム ──────────────────
    _rename = {
        "grade":      "race_grade",
        "rentan_hit": "umaren_hit",
        "payout":     "return_total",
        "investment": "bet_total",
    }
    df.rename(columns={k: v for k, v in _rename.items() if k in df.columns}, inplace=True)

    # ── 不足カラムをデフォルト値で補完 ───────────────────────
    _defaults: dict = {
        "race_grade": "",
        "pred1_name": "", "pred1_num": "0", "pred1_prob": "0",
        "pred2_name": "", "pred2_num": "0", "pred2_prob": "0",
        "pred3_name": "", "pred3_num": "0", "pred3_prob": "0",
        "actual1_name": "", "actual1_num": "0",
        "actual2_name": "", "actual2_num": "0",
        "actual3_name": "", "actual3_num": "0",
        "fukusho_hit": "False",
        "umaren_hit": "False",   "umaren_payout": "0",
        "wide_hit": "False",     "wide_payout": "0",
        "sanrenpuku_hit": "False", "sanrenpuku_payout": "0",
        "bet_total": "0",        "return_total": "0",
    }
    for col, default in _defaults.items():
        if col not in df.columns:
            df[col] = default

    # ── 型変換 ───────────────────────────────────────────────
    for col in ("pred1_prob", "pred2_prob", "pred3_prob"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("fukusho_hit", "umaren_hit", "wide_hit", "sanrenpuku_hit"):
        df[col] = df[col].map({"True": True, "False": False, True: True, False: False})
    for col in ("umaren_payout", "wide_payout", "sanrenpuku_payout",
                "bet_total", "return_total"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def record_result(
    race_id:   str,
    race_name: str,
    race_date: str,
    pred:      dict,
    actual_df: pd.DataFrame,
    payouts:   dict,
) -> dict:
    """
    1 レース分の結果を results_history.csv に追記する。

    Args:
        pred:      _load_cache()[race_id] の形式
        actual_df: scrape_race_result の戻り値
        payouts:   scrape_payouts の戻り値

    Returns:
        記録した 1 行分の dict
    """
    from keiba_predictor.discord_notify import (
        _check_umaren_raw, _check_wide_pairs_raw, _check_sanrenpuku_raw,
    )

    grade = _grade_label(race_name)

    # 予想馬
    p1 = _pred_row(pred, "honmei")
    p2 = _pred_row(pred, "taikou")
    p3 = _pred_row(pred, "ana")
    predicted_nums = pred.get("predicted_top3_nums", [])

    # 実際の着順
    actuals = _top3_actual(actual_df)
    actual_nums = [a["num"] for a in actuals]

    def _a(i: int) -> dict:
        return actuals[i] if i < len(actuals) else {"name": "", "num": 0}

    # 複勝的中: ◎（honmei = predicted_nums[0]）が 3 着以内に入ったか
    fukusho_hit = bool(predicted_nums) and (predicted_nums[0] in actual_nums)

    # 馬連・ワイド・3連複
    umaren_hit, umaren_pay_str  = _check_umaren_raw(predicted_nums, actual_nums, payouts)
    wide_pairs                  = _check_wide_pairs_raw(predicted_nums, actual_nums, payouts)
    ana_horse_num = pred.get("ana_horse_num")
    sanren_hit, sanren_pay_str  = _check_sanrenpuku_raw(predicted_nums, actual_nums, payouts, ana_horse_num)

    wide_hit    = any(h for _, h, _ in wide_pairs)
    wide_payout = sum(_payout_str_to_int(p) for _, h, p in wide_pairs if h)

    umaren_payout   = _payout_str_to_int(umaren_pay_str)
    sanren_payout   = _payout_str_to_int(sanren_pay_str)

    bet_total    = UNIT_BET * BETS_PER_RACE
    return_total = umaren_payout + wide_payout + sanren_payout

    row = {
        "date":       race_date,
        "race_id":    race_id,
        "race_name":  race_name,
        "race_grade": grade,
        "pred1_name": p1["name"], "pred1_num": p1["num"], "pred1_prob": p1["prob"],
        "pred2_name": p2["name"], "pred2_num": p2["num"], "pred2_prob": p2["prob"],
        "pred3_name": p3["name"], "pred3_num": p3["num"], "pred3_prob": p3["prob"],
        "actual1_name": _a(0)["name"], "actual1_num": _a(0)["num"],
        "actual2_name": _a(1)["name"], "actual2_num": _a(1)["num"],
        "actual3_name": _a(2)["name"], "actual3_num": _a(2)["num"],
        "fukusho_hit":     fukusho_hit,
        "umaren_hit":      umaren_hit,     "umaren_payout":   umaren_payout,
        "wide_hit":        wide_hit,       "wide_payout":     wide_payout,
        "sanrenpuku_hit":  sanren_hit,     "sanrenpuku_payout": sanren_payout,
        "bet_total":       bet_total,
        "return_total":    return_total,
    }

    # CSV 追記
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new_row_df = pd.DataFrame([row])
    if HISTORY_PATH.exists():
        new_row_df.to_csv(HISTORY_PATH, mode="a", header=False,
                          index=False, encoding="utf-8-sig")
    else:
        new_row_df.to_csv(HISTORY_PATH, mode="w", header=True,
                          index=False, encoding="utf-8-sig")

    logger.info(
        f"  [history] 記録: {race_name} "
        f"fukusho={fukusho_hit} umaren={umaren_hit} "
        f"wide={wide_hit} sanren={sanren_hit} "
        f"return=¥{return_total:,}"
    )
    return row


def weekly_summary(df: pd.DataFrame, week_end: date) -> dict:
    """
    week_end を含む週（月〜日）の集計 dict を返す。

    Returns:
        {
          "n_races": int,
          "fukusho_rate": float,
          "umaren_rate": float,
          "wide_rate": float,
          "sanrenpuku_rate": float,
          "bet_total": int,
          "return_total": int,
          "roi": float,          # 回収率 (return / bet)
        }
    """
    week_start = week_end - timedelta(days=week_end.weekday())  # 月曜
    ws = pd.Timestamp(week_start)
    we = pd.Timestamp(week_end)
    mask = (df["date"] >= ws) & (df["date"] <= we + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    wdf = df[mask]

    if wdf.empty:
        return {"n_races": 0, "fukusho_rate": 0.0, "umaren_rate": 0.0,
                "wide_rate": 0.0, "sanrenpuku_rate": 0.0,
                "bet_total": 0, "return_total": 0, "roi": 0.0}

    n = len(wdf)
    bet   = int(wdf["bet_total"].sum())
    ret   = int(wdf["return_total"].sum())
    return {
        "n_races":         n,
        "fukusho_rate":    float(wdf["fukusho_hit"].sum() / n),
        "umaren_rate":     float(wdf["umaren_hit"].sum() / n),
        "wide_rate":       float(wdf["wide_hit"].sum() / n),
        "sanrenpuku_rate": float(wdf["sanrenpuku_hit"].sum() / n),
        "bet_total":       bet,
        "return_total":    ret,
        "roi":             ret / bet if bet else 0.0,
    }


def cumulative_summary(df: pd.DataFrame) -> dict:
    """全期間の累計集計 dict を返す。"""
    if df.empty:
        return {"n_races": 0, "fukusho_rate": 0.0, "umaren_rate": 0.0,
                "wide_rate": 0.0, "sanrenpuku_rate": 0.0,
                "bet_total": 0, "return_total": 0, "roi": 0.0}
    n   = len(df)
    bet = int(df["bet_total"].sum())
    ret = int(df["return_total"].sum())
    return {
        "n_races":         n,
        "fukusho_rate":    float(df["fukusho_hit"].sum() / n),
        "umaren_rate":     float(df["umaren_hit"].sum() / n),
        "wide_rate":       float(df["wide_hit"].sum() / n),
        "sanrenpuku_rate": float(df["sanrenpuku_hit"].sum() / n),
        "bet_total":       bet,
        "return_total":    ret,
        "roi":             ret / bet if bet else 0.0,
    }


def hit_streak(df: pd.DataFrame) -> int:
    """複勝的中した連続週数（最新から遡る）を返す。"""
    if df.empty:
        return 0
    # 週ごとに複勝的中があったかをチェック
    df = df.copy()
    df["week"] = df["date"].dt.to_period("W")
    weekly = df.groupby("week")["fukusho_hit"].any().sort_index(ascending=False)
    streak = 0
    for hit in weekly:
        if hit:
            streak += 1
        else:
            break
    return streak


def format_summary_message(
    week_stats: dict,
    cum_stats: dict,
    streak: int,
) -> str:
    """Discord 用サマリーメッセージを生成する。"""
    RULE = "━" * 24
    w = week_stats
    c = cum_stats

    # 今週の複勝的中数
    week_wins = int(round(w["fukusho_rate"] * w["n_races"]))

    lines = [
        RULE,
        f"📊 今週成績  {w['n_races']}戦{week_wins}勝",
        f"📈 累計複勝的中率  {c['fukusho_rate'] * 100:.0f}%",
        f"💰 累計回収率  {c['roi'] * 100:.0f}%",
    ]
    if streak >= 2:
        lines.append(f"🔥 {streak}週連続複勝的中中！")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# note 用週次レポート生成
# ══════════════════════════════════════════════════════════════

def _load_pred_cache() -> dict:
    """predictions_cache.json を読む（なければ空 dict）。"""
    import json
    cache_path = DATA_DIR / "predictions_cache.json"
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_weekly_report(week_date: str, output_path: Optional[Path] = None) -> str:
    """
    指定週のレポートを Markdown 文字列で返し、ファイルにも保存する。

    Args:
        week_date: 'YYYY-MM-DD' 形式（その週の日曜日 or 任意の日付）
    """
    target_date = pd.to_datetime(week_date).date()
    week_end    = target_date + timedelta(days=(6 - target_date.weekday()))  # 次の日曜
    week_start  = week_end - timedelta(days=6)

    df      = load_history()
    cache   = _load_pred_cache()
    w_stats = weekly_summary(df, week_end)
    c_stats = cumulative_summary(df)
    streak  = hit_streak(df)

    # 当週のレース
    _ws = pd.Timestamp(week_start)
    _we = pd.Timestamp(week_end)
    mask    = (df["date"] >= _ws) & (df["date"] <= _we + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    week_df = df[mask].copy()

    lines: list[str] = []

    # ── タイトル ────────────────────────────────────────────
    lines += [
        "# KEIBA EDGE 週次レポート",
        f"**{week_start.strftime('%Y年%m月%d日')}〜{week_end.strftime('%m月%d日')}**",
        "",
        "> *KEIBA EDGE 独自の期待値分析・危険馬判定搭載*",
        "",
        "---",
        "",
    ]

    # ── レース別結果 ─────────────────────────────────────────
    if week_df.empty:
        lines.append("> この週のレース記録はありません。")
    else:
        lines += ["## 各重賞の予想と結果", ""]
        for _, row in week_df.iterrows():
            race_id  = str(row["race_id"])
            hit_icon = "✅" if row["fukusho_hit"] else "❌"
            lines += [
                f"### {hit_icon} {row['race_name']}  `{row['race_grade']}`  "
                f"({pd.Timestamp(row['date']).strftime('%m/%d')})",
                "",
                "**AI予想（3着以内確率順）**",
                "",
                f"| 印 | 馬名 | 馬番 | 確率 |",
                f"|:--:|:-----|:----:|-----:|",
                f"| ◎ | {row['pred1_name']} | {row['pred1_num']}番 | {row['pred1_prob']:.1f}% |",
                f"| ○ | {row['pred2_name']} | {row['pred2_num']}番 | {row['pred2_prob']:.1f}% |",
                f"| △ | {row['pred3_name']} | {row['pred3_num']}番 | {row['pred3_prob']:.1f}% |",
                "",
                "**確定着順**",
                "",
                f"| 着順 | 馬名 | 馬番 |",
                f"|:----:|:-----|:----:|",
                f"| 1着 | {row['actual1_name']} | {row['actual1_num']}番 |",
                f"| 2着 | {row['actual2_name']} | {row['actual2_num']}番 |",
                f"| 3着 | {row['actual3_name']} | {row['actual3_num']}番 |",
                "",
            ]

            # KEIBA EDGE 独自分析: EV・危険馬（キャッシュから取得）
            race_cache = cache.get(race_id, {})

            ev_top3 = race_cache.get("ev_top3", [])
            if ev_top3:
                lines += [
                    "**📊 KEIBA EDGE 独自分析 ─ 期待値（EV）分析**",
                    "",
                    "| 馬番 | 馬名 | 確率 | オッズ | 期待値 | 評価 |",
                    "|:----:|:-----|-----:|------:|------:|:----:|",
                ]
                for e in ev_top3:
                    ev    = e.get("ev_score", 0)
                    prob  = e.get("prob", 0) * 100
                    odds  = e.get("odds", 0)
                    mark  = "★ **EV+**" if ev >= 1.0 else "−"
                    lines.append(
                        f"| {e.get('horse_number','-')}番 "
                        f"| {e.get('horse_name','')} "
                        f"| {prob:.1f}% "
                        f"| {odds:.1f}倍 "
                        f"| **{ev:.2f}** "
                        f"| {mark} |"
                    )
                lines.append("")

            dangerous = race_cache.get("dangerous_horses", [])
            if dangerous:
                lines += [
                    "**⚠️ KEIBA EDGE 独自分析 ─ 危険な人気馬**",
                    "",
                ]
                for d in dangerous:
                    num  = d.get("horse_number", "-")
                    name = d.get("horse_name", "")
                    pop  = d.get("popularity", "?")
                    lines.append(f"- ⚠️ **{num}番 {name}**（{pop}番人気）")
                    for rsn in d.get("reasons", []):
                        lines.append(f"  - {rsn}")
                lines.append("")

            # 推奨買い目（2パターン）
            prob_nums: list[int] = [
                n for n in race_cache.get("predicted_top3_nums", []) if n is not None
            ]
            ev_nums: list[int] = [
                e["horse_number"]
                for e in race_cache.get("ev_top3", [])
                if e.get("horse_number") is not None
            ]
            if not ev_nums:
                ev_nums = prob_nums
            if prob_nums:
                lines.append("**推奨買い目（予想時点）**")
                lines.append("")

                def _md_combo(nums: list[int]) -> list[str]:
                    out: list[str] = []
                    if len(nums) >= 2:
                        out.append("馬連 / ワイド:")
                        for a, b in combinations(nums, 2):
                            out.append(f"- {a}-{b}")
                    if len(nums) >= 3:
                        out.append(f"三連複: {nums[0]}-{nums[1]}-{nums[2]}")
                    return out

                if set(prob_nums) == set(ev_nums):
                    lines.append("【安定重視】確率TOP3で堅く")
                    lines += _md_combo(prob_nums)
                else:
                    lines.append("【安定重視】確率TOP3で堅く")
                    lines += _md_combo(prob_nums)
                    lines.append("")
                    lines.append("【期待値重視】EV上位3頭で配当狙い")
                    lines += _md_combo(ev_nums)
                lines.append("")

            # 買い目結果
            bet_lines: list[str] = []
            umaren_icon = "✅" if row["umaren_hit"] else "❌"
            bet_lines.append(
                f"- 馬連: {umaren_icon}"
                + (f" → **¥{int(row['umaren_payout']):,}**" if row["umaren_hit"] else "")
            )
            wide_icon = "✅" if row["wide_hit"] else "❌"
            bet_lines.append(
                f"- ワイド: {wide_icon}"
                + (f" → **¥{int(row['wide_payout']):,}**" if row["wide_hit"] else "")
            )
            sanren_icon = "✅" if row["sanrenpuku_hit"] else "❌"
            bet_lines.append(
                f"- 3連複: {sanren_icon}"
                + (f" → **¥{int(row['sanrenpuku_payout']):,}**" if row["sanrenpuku_hit"] else "")
            )

            bet_spent  = int(row["bet_total"])
            bet_return = int(row["return_total"])
            profit     = bet_return - bet_spent
            profit_str = f"+¥{profit:,}" if profit >= 0 else f"¥{profit:,}"
            bet_lines.append(
                f"- 収支: ¥{bet_spent:,} 投資 → ¥{bet_return:,} 回収 （{profit_str}）"
            )

            lines += ["**推奨買い目 結果**", ""] + bet_lines + ["", "---", ""]

    # ── 今週の総収支 ─────────────────────────────────────────
    lines += [
        "## 今週の総収支",
        "",
        f"| 項目 | 値 |",
        f"|:-----|:--|",
        f"| レース数 | {w_stats['n_races']}レース |",
        f"| 複勝的中率 | {w_stats['fukusho_rate']*100:.1f}% |",
        f"| 馬連的中率 | {w_stats['umaren_rate']*100:.1f}% |",
        f"| ワイド的中率 | {w_stats['wide_rate']*100:.1f}% |",
        f"| 3連複的中率 | {w_stats['sanrenpuku_rate']*100:.1f}% |",
        f"| 投資合計 | ¥{w_stats['bet_total']:,} |",
        f"| 回収合計 | ¥{w_stats['return_total']:,} |",
        f"| 回収率 | {w_stats['roi']*100:.1f}% |",
        "",
        "---",
        "",
        "## 累計実績",
        "",
        f"| 項目 | 値 |",
        f"|:-----|:--|",
        f"| 通算レース数 | {c_stats['n_races']}レース |",
        f"| 複勝的中率 | {c_stats['fukusho_rate']*100:.1f}% |",
        f"| 馬連的中率 | {c_stats['umaren_rate']*100:.1f}% |",
        f"| ワイド的中率 | {c_stats['wide_rate']*100:.1f}% |",
        f"| 3連複的中率 | {c_stats['sanrenpuku_rate']*100:.1f}% |",
        f"| 累計投資 | ¥{c_stats['bet_total']:,} |",
        f"| 累計回収 | ¥{c_stats['return_total']:,} |",
        f"| 累計回収率 | {c_stats['roi']*100:.1f}% |",
        f"| 連続的中 | {streak}週連続 |",
        "",
        "---",
        "",
        "*このレポートは KEIBA EDGE AI 予測システムにより自動生成されました。*",
    ]

    report_text = "\n".join(lines)

    # 保存
    if output_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_DIR / f"report_{week_end.strftime('%Y%m%d')}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    logger.info(f"レポート保存: {output_path}")

    return report_text
