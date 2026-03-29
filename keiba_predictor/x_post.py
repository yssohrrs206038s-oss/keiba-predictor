"""
X（Twitter）自動投稿モジュール

【環境変数】
    TWITTER_API_KEY              : Consumer Key（API Key）
    TWITTER_API_SECRET           : Consumer Secret（API Secret）
    TWITTER_ACCESS_TOKEN         : Access Token
    TWITTER_ACCESS_TOKEN_SECRET  : Access Token Secret

環境変数が未設定の場合は投稿をスキップし、エラーにはなりません。
"""

import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)

# 文字数の安全上限
_CHAR_LIMIT = 140

# グレード判定パターン
_GRADE_PATS = [
    ("GI",   re.compile(r"[（(]G\s*[1Ⅰ][）)]|[（(]GI[）)]",   re.I)),
    ("GII",  re.compile(r"[（(]G\s*[2Ⅱ][）)]|[（(]GII[）)]",  re.I)),
    ("GIII", re.compile(r"[（(]G\s*[3Ⅲ][）)]|[（(]GIII[）)]", re.I)),
]


# ── 内部ユーティリティ ────────────────────────────────────────────────

def _build_client():
    """tweepy.Client を環境変数から構築する。未設定なら None を返す。"""
    try:
        import tweepy
    except ImportError:
        logger.warning("[X] tweepy がインストールされていません: pip install tweepy")
        return None

    keys = {
        "consumer_key":        os.environ.get("TWITTER_API_KEY", ""),
        "consumer_secret":     os.environ.get("TWITTER_API_SECRET", ""),
        "access_token":        os.environ.get("TWITTER_ACCESS_TOKEN", ""),
        "access_token_secret": os.environ.get("TWITTER_ACCESS_TOKEN_SECRET", ""),
    }
    if not all(keys.values()):
        missing = [k for k, v in keys.items() if not v]
        logger.info(f"[X] 資格情報未設定のためスキップ ({missing})")
        return None
    return tweepy.Client(**keys)


def _grade_label(race_name: str) -> str:
    for label, pat in _GRADE_PATS:
        if pat.search(race_name):
            return label
    return ""


def _short_name(race_name: str) -> str:
    """括弧内グレード表記を除いた短縮レース名。ハッシュタグ用。"""
    return re.sub(r"[（(]G[^）)]*[）)]", "", race_name).strip()


def _ev_stars(ev: float) -> str:
    if ev >= 15:
        return "★★★"
    elif ev >= 12:
        return "★★"
    elif ev >= 9:
        return "★"
    return ""


def _safe_post(client, text: str) -> bool:
    """ツイートを投稿し、成否を返す。上限超は末尾を切り詰める。"""
    if len(text) > _CHAR_LIMIT:
        text = text[: _CHAR_LIMIT - 1] + "…"
    try:
        resp = client.create_tweet(text=text)
        tweet_id = resp.data.get("id", "?")
        logger.info(f"[X] 投稿完了 id={tweet_id}")
        return True
    except Exception as e:
        logger.warning(f"[X] 投稿失敗: {e}")
        return False


# ── 予想ツイート ──────────────────────────────────────────────────────

def build_predict_tweet(race_name: str, cache_entry: dict) -> str:
    grade = _grade_label(race_name)
    short = _short_name(race_name)

    ev_map = {e["horse_number"]: e["ev_score"]
              for e in cache_entry.get("ev_top3", [])}

    lines = [f"🏇【{short}{' ' + grade if grade else ''}】KEIBA EDGE予想"]

    for role, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "▲")]:
        p = cache_entry.get(role, {})
        if not p or not p.get("horse_name"):
            continue
        num   = p.get("horse_number", "?")
        name  = p.get("horse_name", "")
        ev    = ev_map.get(num, 0)
        stars = _ev_stars(ev)
        lines.append(f"{mark}{num}番{name}{stars}")

    # 穴馬（ana_horse_num）
    ana_num = cache_entry.get("ana_horse_num")
    ana_info = cache_entry.get("ana_horse_info", {})
    if ana_num and ana_info.get("horse_name"):
        prob = ana_info.get("prob", 0) * 100
        pop = ana_info.get("popularity", "?")
        lines.append(f"★穴{ana_num}番{ana_info['horse_name']}（AI{prob:.0f}% {pop}人気）")

    # 危険馬（1頭のみ）
    for d in cache_entry.get("dangerous_horses", [])[:1]:
        lines.append(f"⚠️{d['horse_number']}番{d['horse_name']}({d['popularity']}人気)")

    lines.append(f"#競馬予想 #{short} #KEIBA_EDGE")

    return "\n".join(lines)


def post_predict_tweet(race_name: str, cache_entry: dict) -> bool:
    """予想ツイートを X に投稿する。資格情報未設定時はスキップ（エラーなし）。"""
    client = _build_client()
    if client is None:
        return False
    text = build_predict_tweet(race_name, cache_entry)
    print(f"[X予想ツイート]\n{text}", flush=True)
    return _safe_post(client, text)


# ── 結果ツイート ──────────────────────────────────────────────────────

def build_result_tweet(
    race_name: str,
    actual_df: pd.DataFrame,
    pred: dict,
    payouts: dict,
    roi_pct: float,
) -> str:
    from keiba_predictor.discord_notify import (
        _check_sanrenpuku_raw, _check_umaren_raw,
    )

    grade = _grade_label(race_name)
    short = _short_name(race_name)

    predicted_nums = pred.get("predicted_top3_nums", [])
    ana_horse_num = pred.get("ana_horse_num")

    # 本命情報
    honmei = pred.get("honmei", {})
    honmei_num = honmei.get("horse_number")
    honmei_name = honmei.get("horse_name", "")

    # 実際の3着以内
    df = actual_df.copy()
    df["_fp"] = pd.to_numeric(df["finish_position"], errors="coerce")
    top3 = df[df["_fp"].isin([1, 2, 3])].sort_values("_fp").head(3)
    actual_nums: list[int] = []
    for _, r in top3.iterrows():
        num = int(r["horse_number"]) if pd.notna(r.get("horse_number")) else 0
        actual_nums.append(num)

    # 的中判定
    fukusho_hit = honmei_num is not None and int(honmei_num) in actual_nums
    umaren_hit, umaren_pay = _check_umaren_raw(predicted_nums, actual_nums, payouts)
    sanren_hit, sanren_pay = _check_sanrenpuku_raw(
        predicted_nums, actual_nums, payouts, ana_horse_num)

    f_icon = "✅" if fukusho_hit else "❌"
    u_icon = "✅" if umaren_hit else "❌"
    s_icon = "✅" if sanren_hit else "❌"

    SEP = "━━━━━━━━━━━━━━━━"

    any_hit = fukusho_hit or umaren_hit or sanren_hit

    if sanren_hit:
        # 3連複的中: 特別フォーマット
        pay_str = re.sub(r"[¥,]", "", str(sanren_pay)) if sanren_pay else ""
        lines = [
            "🎯💥 KEIBA EDGE 的中！",
            SEP,
            f"【{short}{' ' + grade if grade else ''}】",
            f"複勝{f_icon} 馬連{u_icon} 3連複{s_icon}",
            "",
            f"3連複 {pay_str}円的中！" if pay_str else "3連複 的中！",
            f"回収率 {roi_pct:.0f}%" if roi_pct > 0 else "",
            "",
            f"◎{honmei_num}番{honmei_name}" if honmei_num else "",
            f"#競馬的中 #{short} #KEIBA_EDGE",
        ]
    elif any_hit:
        # 複勝または馬連のみ的中
        lines = [
            "🎯 KEIBA EDGE 的中！",
            f"【{short}{' ' + grade if grade else ''}】",
            f"複勝{f_icon} 馬連{u_icon} 3連複{s_icon}",
            f"回収率{roi_pct:.0f}%" if roi_pct > 0 else "",
            f"#競馬的中 #{short} #KEIBA_EDGE",
        ]
    else:
        # 全外れ: ハッシュタグなし
        lines = [
            f"🏆【{short}{' ' + grade if grade else ''}】KEIBA EDGE結果",
            f"複勝{f_icon} 馬連{u_icon} 3連複{s_icon}",
            f"累計回収率{roi_pct:.0f}%" if roi_pct > 0 else "",
        ]

    # 空行を除去
    return "\n".join(line for line in lines if line)


def post_result_tweet(
    race_name: str,
    actual_df: pd.DataFrame,
    pred: dict,
    payouts: dict,
) -> bool:
    """結果ツイートを X に投稿する。累計回収率は results_history.csv から自動取得。"""
    client = _build_client()
    if client is None:
        return False

    roi_pct = 0.0
    try:
        from keiba_predictor.history import cumulative_summary, load_history
        roi_pct = cumulative_summary(load_history())["roi"] * 100
    except Exception as e:
        logger.debug(f"[X] 累計回収率取得失敗: {e}")

    text = build_result_tweet(race_name, actual_df, pred, payouts, roi_pct)
    print(f"[X結果ツイート]\n{text}", flush=True)
    return _safe_post(client, text)


# ── 週次サマリーツイート ─────────────────────────────────────────────────

def build_weekly_summary_tweet(results: list[dict]) -> str:
    """
    週次サマリーツイートを構築する。

    Args:
        results: [{"race_name": "日経賞", "fukusho": True, "umaren": True,
                    "sanren": False, "bet": 1400, "return_total": 1200}, ...]
    """
    SEP = "━━━━━━━━━━━━━━━━"

    lines = [
        "📊 今週のKEIBA EDGE成績",
        SEP,
    ]

    hit_count = 0
    for r in results:
        name = _short_name(r.get("race_name", ""))
        f_icon = "✅" if r.get("fukusho") else "❌"
        u_icon = "✅" if r.get("umaren") else "❌"
        s_icon = "✅" if r.get("sanren") else "❌"
        if r.get("fukusho") or r.get("umaren") or r.get("sanren"):
            hit_count += 1
        lines.append(f"{name} {f_icon}{u_icon}{s_icon}")

    total = len(results)
    fukusho_hits = sum(1 for r in results if r.get("fukusho"))
    fukusho_rate = (fukusho_hits / total * 100) if total else 0

    total_bet = sum(r.get("bet", 0) for r in results)
    total_ret = sum(r.get("return_total", 0) for r in results)
    roi = (total_ret / total_bet * 100) if total_bet > 0 else 0

    lines += [
        "",
        f"{hit_count}/{total}レース的中",
        f"複勝的中率: {fukusho_rate:.0f}%",
        f"回収率: {roi:.0f}%",
        SEP,
        "来週も予想します👇",
        "#KEIBA_EDGE #AI競馬予想",
    ]

    return "\n".join(lines)


def post_weekly_summary_tweet(results: list[dict]) -> bool:
    """週次サマリーをXに投稿する。"""
    client = _build_client()
    if client is None:
        return False

    text = build_weekly_summary_tweet(results)
    print(f"[X週次サマリーツイート]\n{text}", flush=True)
    return _safe_post(client, text)
