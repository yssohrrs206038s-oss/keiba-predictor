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
    tag = f"#{short.replace(' ', '')} #KEIBA_EDGE"

    lines = [f"🏇{short}{' ' + grade if grade else ''} AI予想"]

    for role, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "▲")]:
        p = cache_entry.get(role, {})
        if not p or not p.get("horse_name"):
            continue
        lines.append(f"{mark}{p['horse_number']}番{p['horse_name']}")

    # 穴馬
    ana_num = cache_entry.get("ana_horse_num")
    ana_info = cache_entry.get("ana_horse_info", {})
    if ana_num and ana_info.get("horse_name"):
        lines.append(f"★穴{ana_num}番{ana_info['horse_name']}")

    # 危険馬（余裕があれば）
    text = "\n".join(lines) + "\n" + tag
    if len(text) <= 120:
        for d in cache_entry.get("dangerous_horses", [])[:1]:
            lines.append(f"⚠{d['horse_number']}番{d['horse_name']}")

    text = "\n".join(lines) + "\n" + tag
    if len(text) > _CHAR_LIMIT:
        text = "\n".join(lines[:-1]) + "\n" + tag
    return text


def post_predict_tweet(race_name: str, cache_entry: dict) -> bool:
    """予想ツイートを X に投稿する。資格情報未設定時はスキップ（エラーなし）。"""
    client = _build_client()
    if client is None:
        return False
    text = build_predict_tweet(race_name, cache_entry)
    print(f"[X予想ツイート]\n{text}", flush=True)
    return _safe_post(client, text)


# ── 結果ツイート ──────────────────────────────────────────────────────

def _build_result_tweets(
    race_name: str,
    actual_df: pd.DataFrame,
    pred: dict,
    payouts: dict,
    roi_pct: float,
) -> tuple[str, str]:
    """
    結果ツイートを構築する。(1投稿目, 2投稿目) を返す。
    2投稿目は的中時のみ（外れ時は空文字列）。
    """
    from keiba_predictor.discord_notify import (
        _check_sanrenpuku_raw, _check_umaren_raw,
    )

    grade = _grade_label(race_name)
    short = _short_name(race_name)

    predicted_nums = pred.get("predicted_top3_nums", [])
    ana_horse_num = pred.get("ana_horse_num")

    honmei = pred.get("honmei", {})
    honmei_num = honmei.get("horse_number")
    honmei_name = honmei.get("horse_name", "")

    # 実際の3着以内
    df = actual_df.copy()
    df["_fp"] = pd.to_numeric(df["finish_position"], errors="coerce")
    top3 = df[df["_fp"].isin([1, 2, 3])].sort_values("_fp").head(3)
    actual_nums: list[int] = []
    actual_entries: list[tuple[int, int, str]] = []  # (fp, num, name)
    for _, r in top3.iterrows():
        fp  = int(r["_fp"])
        num = int(r["horse_number"]) if pd.notna(r.get("horse_number")) else 0
        name = str(r.get("horse_name", ""))
        actual_nums.append(num)
        actual_entries.append((fp, num, name))

    # 的中判定
    fukusho_hit = honmei_num is not None and int(honmei_num) in actual_nums
    umaren_hit, umaren_pay = _check_umaren_raw(predicted_nums, actual_nums, payouts)
    sanren_hit, sanren_pay = _check_sanrenpuku_raw(
        predicted_nums, actual_nums, payouts, ana_horse_num)

    f_icon = "✅" if fukusho_hit else "❌"
    u_icon = "✅" if umaren_hit else "❌"
    s_icon = "✅" if sanren_hit else "❌"

    any_hit = fukusho_hit or umaren_hit or sanren_hit

    # ── 1投稿目 ──────────────────────────────────────────────
    tag = f"#{short.replace(' ', '')} #KEIBA_EDGE"
    if sanren_hit:
        pay_str = re.sub(r"[¥,]", "", str(sanren_pay)) if sanren_pay else ""
        lines1 = [
            f"🎯3連複的中！{short}",
            f"{pay_str}円 回収率{roi_pct:.0f}%" if pay_str and roi_pct > 0
                else (f"{pay_str}円的中！" if pay_str else ""),
            tag,
        ]
    elif any_hit:
        lines1 = [
            f"🎯的中！{short}",
            f"複勝{f_icon} 馬連{u_icon}",
            tag,
        ]
    else:
        result_line = " ".join(f"{fp}着{num}番" for fp, num, _ in actual_entries[:3])
        lines1 = [
            f"{short} 結果",
            result_line,
            f"複勝❌馬連❌3連複❌",
        ]

    tweet1 = "\n".join(line for line in lines1 if line)

    # ── 2投稿目は廃止（140字制限のため1投稿に集約） ──
    tweet2 = ""

    return tweet1, tweet2


# 後方互換: 旧関数名
def build_result_tweet(
    race_name: str,
    actual_df: pd.DataFrame,
    pred: dict,
    payouts: dict,
    roi_pct: float,
) -> str:
    tweet1, _ = _build_result_tweets(race_name, actual_df, pred, payouts, roi_pct)
    return tweet1


def post_result_tweet(
    race_name: str,
    actual_df: pd.DataFrame,
    pred: dict,
    payouts: dict,
) -> bool:
    """結果ツイートを X に投稿。的中時は2投稿目をリプライでスレッド投稿。"""
    client = _build_client()
    if client is None:
        return False

    roi_pct = 0.0
    try:
        from keiba_predictor.history import cumulative_summary, load_history
        roi_pct = cumulative_summary(load_history())["roi"] * 100
    except Exception as e:
        logger.debug(f"[X] 累計回収率取得失敗: {e}")

    tweet1, tweet2 = _build_result_tweets(race_name, actual_df, pred, payouts, roi_pct)
    print(f"[X結果ツイート1]\n{tweet1}", flush=True)

    # 1投稿目
    if len(tweet1) > _CHAR_LIMIT:
        tweet1 = tweet1[: _CHAR_LIMIT - 1] + "…"
    try:
        resp1 = client.create_tweet(text=tweet1)
        tweet_id = resp1.data.get("id", "")
        logger.info(f"[X] 1投稿目完了 id={tweet_id}")
    except Exception as e:
        logger.warning(f"[X] 1投稿目失敗: {e}")
        return False

    return True


# ── 週末重賞予告ツイート ──────────────────────────────────────────────────

def build_preview_tweet(races: list[dict]) -> str:
    """
    今週末の重賞予告ツイートを生成（140字以内）。

    Args:
        races: [{"race_name", "race_date", "venue", "course_info"}, ...]
    """
    if not races:
        return ""

    # 土曜/日曜に分類
    sat = [r for r in races if r.get("race_date", "").endswith(("土", "")) and "土" not in r.get("race_date", "")]
    sun = []
    # race_date が YYYY-MM-DD 形式なら曜日で分ける
    from datetime import datetime
    sat_races, sun_races = [], []
    for r in races:
        try:
            d = datetime.strptime(r["race_date"], "%Y-%m-%d")
            if d.weekday() == 5:
                sat_races.append(r)
            else:
                sun_races.append(r)
        except Exception:
            sat_races.append(r)

    tag = "#KEIBA_EDGE"
    lines = ["🏇今週末の重賞"]

    for label, rs in [("土", sat_races), ("日", sun_races)]:
        if not rs:
            continue
        for r in rs[:2]:
            name = _short_name(r.get("race_name", ""))
            grade = _grade_label(r.get("race_name", ""))
            venue = r.get("venue", "")
            g = f"({grade})" if grade else ""
            lines.append(f"{label} {venue}{name}{g}")

    lines += ["当日14時にAI予想公開", tag]

    text = "\n".join(lines)
    if len(text) > _CHAR_LIMIT:
        # レース行を減らす
        lines = [lines[0]] + lines[1:3] + lines[-2:]
        text = "\n".join(lines)
    return text


def post_preview_tweet(races: list[dict]) -> bool:
    """週末重賞予告をXに投稿する。"""
    client = _build_client()
    if client is None:
        return False
    text = build_preview_tweet(races)
    if not text:
        return False
    print(f"[X予告ツイート]\n{text}", flush=True)
    return _safe_post(client, text)


# ── 週次サマリーツイート ─────────────────────────────────────────────────

def build_weekly_summary_tweet(results: list[dict]) -> str:
    """週次サマリーツイートを構築する（140字以内）。"""
    total = len(results)
    if not total:
        return ""

    hit_count = sum(1 for r in results if r.get("fukusho") or r.get("umaren") or r.get("sanren"))
    fukusho_hits = sum(1 for r in results if r.get("fukusho"))
    fukusho_rate = (fukusho_hits / total * 100) if total else 0
    total_bet = sum(r.get("bet", 0) for r in results)
    total_ret = sum(r.get("return_total", 0) for r in results)
    roi = (total_ret / total_bet * 100) if total_bet > 0 else 0

    lines = [
        f"📊今週のAI成績 {hit_count}/{total}的中",
        f"複勝{fukusho_rate:.0f}% 回収率{roi:.0f}%",
    ]

    # レース別結果（余裕がある分だけ）
    race_lines = []
    for r in results:
        name = _short_name(r.get("race_name", ""))[:6]
        f = "○" if r.get("fukusho") else "×"
        u = "○" if r.get("umaren") else "×"
        s = "○" if r.get("sanren") else "×"
        race_lines.append(f"{name}{f}{u}{s}")

    tag = "#KEIBA_EDGE"
    base = "\n".join(lines)
    for rl in race_lines:
        test = base + "\n" + rl + "\n" + tag
        if len(test) <= _CHAR_LIMIT:
            base = base + "\n" + rl
        else:
            break

    return base + "\n" + tag


def post_weekly_summary_tweet(results: list[dict]) -> bool:
    """週次サマリーをXに投稿する。"""
    client = _build_client()
    if client is None:
        return False

    text = build_weekly_summary_tweet(results)
    print(f"[X週次サマリーツイート]\n{text}", flush=True)
    return _safe_post(client, text)
