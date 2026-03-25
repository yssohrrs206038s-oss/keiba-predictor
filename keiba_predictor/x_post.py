"""
X（Twitter）自動投稿モジュール

【環境変数】
    X_API_KEY         : Consumer Key（API Key）
    X_API_SECRET      : Consumer Secret（API Secret）
    X_ACCESS_TOKEN    : Access Token
    X_ACCESS_SECRET   : Access Token Secret

環境変数が未設定の場合は投稿をスキップし、エラーにはなりません。
"""

import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)

# 文字数の安全上限（280 - バッファ10）
_CHAR_LIMIT = 270

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
        "consumer_key":        os.environ.get("X_API_KEY", ""),
        "consumer_secret":     os.environ.get("X_API_SECRET", ""),
        "access_token":        os.environ.get("X_ACCESS_TOKEN", ""),
        "access_token_secret": os.environ.get("X_ACCESS_SECRET", ""),
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
    """
    予想ツイート文字列を生成する。

    Args:
        race_name:   レース名（グレード表記込み）
        cache_entry: predictions_cache.json の 1 レース分エントリ
    """
    grade = _grade_label(race_name)
    short = _short_name(race_name)
    lines = [f"🏇 KEIBA EDGE AI予想", f"【{short} {grade}】"]

    # 印（◎○☆）
    for role, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "☆")]:
        p = cache_entry.get(role, {})
        if not p or not p.get("horse_name"):
            continue
        num  = p.get("horse_number", "?")
        name = p.get("horse_name", "")
        prob = p.get("prob", 0) * 100
        lines.append(f"{mark} {num}番 {name} {prob:.1f}%")

    # 危険馬（1頭のみ）
    for d in cache_entry.get("dangerous_horses", [])[:1]:
        num  = d.get("horse_number", "?")
        name = d.get("horse_name", "")
        pop  = d.get("popularity", "?")
        lines.append(f"⚠️危険：{num}番{name}（{pop}番人気）")

    # 穴馬（predicted_top3_nums 外・EV ≥ 1.0 の最上位1頭）
    pred_nums = set(cache_entry.get("predicted_top3_nums", []))
    for e in cache_entry.get("ev_top3", []):
        enum = e.get("horse_number")
        if enum is not None and int(enum) not in pred_nums and e.get("ev_score", 0) >= 1.0:
            lines.append(f"★穴馬：{enum}番{e.get('horse_name','')} EV{e['ev_score']:.2f}")
            break

    lines += ["詳細はnoteで👇", "note.com/keiba_edge",
              f"#競馬予想 #{short} #KEIBAREDGE #AI競馬"]
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
    """
    結果ツイート文字列を生成する。

    Args:
        roi_pct: 累計回収率（%）
    """
    from keiba_predictor.discord_notify import _check_sanrenpuku_raw

    grade = _grade_label(race_name)
    short = _short_name(race_name)
    lines = [f"🏆 KEIBA EDGE 結果", f"【{short} {grade}】"]

    # 予想馬番 → 印マッピング
    pred_num_to_mark: dict[int, str] = {}
    for role, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "☆")]:
        p = pred.get(role, {})
        num = p.get("horse_number")
        if num is not None:
            pred_num_to_mark[int(num)] = mark

    predicted_nums = pred.get("predicted_top3_nums", [])

    # 実際の 1〜3 着
    df = actual_df.copy()
    df["_fp"] = pd.to_numeric(df["finish_position"], errors="coerce")
    top3 = df[df["_fp"].isin([1, 2, 3])].sort_values("_fp").head(3)
    actual_nums: list[int] = []
    for _, r in top3.iterrows():
        fp   = int(r["_fp"])
        num  = int(r["horse_number"]) if pd.notna(r.get("horse_number")) else 0
        name = str(r.get("horse_name", ""))
        actual_nums.append(num)
        mark = pred_num_to_mark.get(num, "　")
        icon = " ✅" if num in predicted_nums else ""
        lines.append(f"{fp}着 {mark} {num}番 {name}{icon}")

    # 3連複
    sanren_hit, _ = _check_sanrenpuku_raw(predicted_nums, actual_nums, payouts)
    lines.append(f"3連複 {'✅ 的中！' if sanren_hit else '❌ ハズレ'}")

    if roi_pct > 0:
        lines.append(f"累計回収率：{roi_pct:.0f}%")

    lines.append(f"#競馬 #{'的中' if sanren_hit else 'AI予想'} #{short} #KEIBAREDGE")
    return "\n".join(lines)


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
