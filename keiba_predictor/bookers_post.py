"""
BOOKERS（bookers.tech）への自動記事投稿モジュール。

【前提】
    - BOOKERS_API_KEY 環境変数にAPIキーを設定
    - 未設定時はスキップ（グレースフルデグラデーション）

【API 仕様（bookers.tech/api/docs）】
    POST https://bookers.tech/api/postcreate/
    Authorization: Token {BOOKERS_API_KEY}
    Body (JSON):
        title       : 記事タイトル
        text        : 本文（プレーンテキスト）
        description : 記事説明（100文字程度）
        price       : 0=無料, 正整数=有料（円）
        category    : カテゴリID（整数）

NOTE: 実際のカテゴリIDは BOOKERS 管理画面で確認し
      BOOKERS_CATEGORY 環境変数で上書き可能。
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent / "data"
CACHE_PATH = DATA_DIR / "predictions_cache.json"

BOOKERS_API_BASE = "https://bookers.tech/api"
DEFAULT_CATEGORY: Optional[int] = None  # BOOKERS管理画面でIDを確認してBOOKERS_CATEGORY環境変数に整数で設定


# ══════════════════════════════════════════════════════════════
# ヘルパー
# ══════════════════════════════════════════════════════════════

def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    with open(CACHE_PATH, encoding="utf-8") as f:
        return json.load(f)


def _resolve_api_key(api_key: Optional[str] = None) -> str:
    return api_key or os.environ.get("BOOKERS_API_KEY", "")


# ══════════════════════════════════════════════════════════════
# 記事本文の組み立て
# ══════════════════════════════════════════════════════════════

MARKS = {"honmei": "◎", "taikou": "○", "ana": "☆"}
# △は predicted_top3_nums[3] があれば使用
ALL_MARKS = ["◎", "○", "☆", "△", "　"]


def _build_ev_map(entry: dict) -> dict[int, float]:
    return {
        int(e["horse_number"]): e["ev_score"]
        for e in entry.get("ev_top3", [])
        if e.get("horse_number") is not None
    }


def _build_article_body(race_id: str, entry: dict) -> str:
    """1レース分の本文を生成する（無料範囲＋有料範囲を1つのtextに）。"""
    race_name   = entry.get("race_name", race_id)
    race_date   = entry.get("race_date", "")
    course_info = entry.get("course_info", "")
    ai_comments: dict = entry.get("ai_comments", {})
    ev_map      = _build_ev_map(entry)
    SEP         = "━" * 20
    lines: list[str] = []

    # ── ヘッダー ─────────────────────────────────────────────
    lines += [SEP, f"🏇 {race_name}　AI予想"]
    if course_info:
        lines.append(f"📍 {course_info}　{race_date}")
    elif race_date:
        lines.append(f"📍 {race_date}")
    lines += [SEP, ""]

    # ── 本命馬（無料プレビュー相当） ─────────────────────────
    honmei = entry.get("honmei", {})
    if honmei and honmei.get("horse_name"):
        h_num    = honmei.get("horse_number")
        h_name   = honmei.get("horse_name", "")
        h_prob   = honmei.get("prob", 0) * 100
        h_ev     = ev_map.get(int(h_num), 0) if h_num is not None else 0
        h_ev_str = f"・期待値{h_ev:.2f}" if h_ev else ""
        lines += [
            f"◎ 本命： {h_num}番 {h_name}",
            f"AI確率{h_prob:.1f}%{h_ev_str}",
        ]
    lines += [SEP, ""]

    # ── 全予想印（◎○☆）────────────────────────────────────
    lines.append("【予想印】")
    for role, mark in MARKS.items():
        p = entry.get(role, {})
        if not p or not p.get("horse_name"):
            continue
        num  = p.get("horse_number")
        name = p.get("horse_name", "")
        prob = p.get("prob", 0) * 100
        ev   = ev_map.get(int(num), 0) if num is not None else 0
        ev_str = f"　EV{ev:.2f}" if ev else ""
        lines.append(f"{mark} {num}番 {name}　AI確率{prob:.1f}%{ev_str}")
        comment = ai_comments.get(str(num), "")
        if comment:
            lines.append(f"  📝 {comment}")
    lines.append("")

    # ── 穴馬 ─────────────────────────────────────────────────
    pred_nums = entry.get("predicted_top3_nums", [])
    pred_set  = set(pred_nums)
    for e in entry.get("ev_top3", []):
        enum = e.get("horse_number")
        if enum is None or int(enum) in pred_set:
            continue
        if e.get("ev_score", 0) >= 1.5:
            ename   = e.get("horse_name", "")
            odds    = e.get("odds", 0)
            pop     = e.get("popularity", "")
            pop_str = f"{pop}人気 " if pop else ""
            lines += [
                "【穴馬注目】",
                f"★ {enum}番 {ename}　EV{e['ev_score']:.2f}（{pop_str}{odds:.0f}倍）",
                "",
            ]
            break

    # ── 危険馬 ────────────────────────────────────────────────
    dangerous = entry.get("dangerous_horses", [])
    if dangerous:
        lines.append("【危険な人気馬】")
        for d in dangerous:
            dnum  = d.get("horse_number", "?")
            dname = d.get("horse_name", "")
            dpop  = d.get("popularity", "?")
            lines.append(f"⚠ {dnum}番 {dname}（{dpop}番人気）")
            for rsn in d.get("reasons", []):
                lines.append(f"  ・{rsn}")
        lines.append("")

    # ── 買い目 ────────────────────────────────────────────────
    if len(pred_nums) >= 2:
        axis   = pred_nums[0]
        umaren = pred_nums[1:4]
        sanren = pred_nums[1:6]
        lines += [
            "【買い目】",
            f"■ 複勝：{axis}番",
            f"■ 馬連：{' / '.join(f'{axis}-{n}' for n in umaren)}",
            f"■ 3連複：軸{axis}番 × {'/'.join(str(n) for n in sanren)}",
            "",
        ]

    lines += [
        SEP,
        "本予想はXGBoostモデルによるAI分析です。投資は自己責任でお願いします。",
        "",
        "#競馬 #AI予想 #KEIBAREDGE",
    ]
    return "\n".join(lines)


def _build_description(entry: dict) -> str:
    """記事説明文（100文字程度）を生成する。"""
    race_name   = entry.get("race_name", "")
    course_info = entry.get("course_info", "")
    honmei = entry.get("honmei", {})
    taikou = entry.get("taikou", {})
    h_name = honmei.get("horse_name", "")
    t_name = taikou.get("horse_name", "")
    parts = [f"【KEIBA EDGE AI予想】{race_name}"]
    if course_info:
        parts.append(f"（{course_info}）")
    if h_name:
        parts.append(f"◎{h_name}")
    if t_name:
        parts.append(f"○{t_name}")
    return "　".join(parts)[:200]


# ══════════════════════════════════════════════════════════════
# 投稿
# ══════════════════════════════════════════════════════════════

def post_article(
    title: str,
    body: str,
    description: str = "",
    price: int = 0,
    api_key: Optional[str] = None,
    category: Optional[int] = None,
    dry_run: bool = False,
) -> Optional[str]:
    """
    BOOKERS に記事を投稿する。

    Returns:
        成功時: 記事UUID（str）
        失敗時: None
    """
    key = _resolve_api_key(api_key)
    if not key:
        logger.warning("[BOOKERS] BOOKERS_API_KEY が未設定のためスキップ")
        return None

    # category は整数 PK が必要。BOOKERS_CATEGORY 環境変数に整数を設定すれば使用。
    # 未設定時は payload から除外（文字列スラッグを送ると 400 エラーになるため）。
    cat_env = category if isinstance(category, int) else None
    if cat_env is None:
        env_val = os.environ.get("BOOKERS_CATEGORY", "")
        if env_val.strip().lstrip("-").isdigit():
            cat_env = int(env_val)

    # 無料公開フィールド名を特定するため、候補を全て同時送信する。
    # レスポンスのJSONで null 以外の値が入っているキーが正解。
    FREE_BODY_PROBE = f"[無料公開テスト] {body[:100]}"
    payload: dict = {
        "title":        title,
        "text":         body,           # 現在の本命
        "body":         FREE_BODY_PROBE,
        "content":      FREE_BODY_PROBE,
        "free_content": FREE_BODY_PROBE,
        "open_body":    FREE_BODY_PROBE,
        "description":  description,
        "price":        price,
        # 下書き制御フィールド候補（どれが有効か確認）
        "is_draft":     True,
        "status":       "draft",
        "published":    False,
    }
    if cat_env is not None:
        payload["category"] = cat_env

    # デバッグ: 送信するpayloadのキー一覧を出力
    print(f"[BOOKERS] 送信payload keys: {list(payload.keys())}", flush=True)

    if dry_run:
        print(f"[BOOKERS dry-run] POST /api/postcreate/")
        print(f"  title={title!r}  price={price}  category={cat_env!r}")
        print(f"  text({len(body)}文字):\n{body[:200]}...")
        return "dry-run-uuid"

    url = f"{BOOKERS_API_BASE}/postcreate/"
    headers = {
        "Authorization": f"Token {key}",
        "Content-Type":  "application/json; charset=utf-8",
    }
    try:
        resp = requests.post(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            timeout=30,
        )
        # デバッグ: ステータスコードとレスポンスJSON全体を出力
        print(f"[BOOKERS] response status={resp.status_code}", flush=True)
        try:
            resp_json = resp.json()
            print(f"[BOOKERS] response JSON (全フィールド):", flush=True)
            for k, v in resp_json.items():
                v_preview = str(v)[:120] if v is not None else "null"
                print(f"  {k}: {v_preview}", flush=True)
        except Exception:
            print(f"[BOOKERS] response text={resp.text[:800]}", flush=True)

        if resp.status_code in (200, 201):
            data = resp.json()
            uuid = data.get("uuid") or data.get("id") or data.get("post_id", "")
            logger.info(f"[BOOKERS] 投稿完了: {title!r}  uuid={uuid}")
            print(f"[BOOKERS] ✅ 投稿完了: {title!r}  uuid={uuid}", flush=True)
            return str(uuid)
        else:
            logger.error(f"[BOOKERS] 投稿失敗: {resp.status_code} {resp.text[:300]}")
            print(f"[BOOKERS] ❌ 投稿失敗: {resp.status_code} {resp.text[:300]}", flush=True)
            return None
    except requests.RequestException as e:
        logger.error(f"[BOOKERS] 通信エラー: {e}")
        print(f"[BOOKERS] ❌ 通信エラー: {e}", flush=True)
        return None


def post_predictions(
    api_key: Optional[str] = None,
    price: int = 0,
    dry_run: bool = False,
) -> int:
    """
    predictions_cache.json の全レースを BOOKERS に投稿する。

    Returns:
        投稿成功件数
    """
    key = _resolve_api_key(api_key)
    if not key:
        print("[BOOKERS] BOOKERS_API_KEY が未設定のためスキップします", flush=True)
        return 0

    cache = _load_cache()
    if not cache:
        print("[BOOKERS] 予想キャッシュが空のためスキップします", flush=True)
        return 0

    print(f"[BOOKERS] {len(cache)} レース分を投稿開始 (price={price})", flush=True)
    success = 0
    for race_id, entry in cache.items():
        race_name = entry.get("race_name", race_id)
        title = f"【KEIBA EDGE】{race_name} AI予想"
        body  = _build_article_body(race_id, entry)
        desc  = _build_description(entry)

        uuid = post_article(
            title=title,
            body=body,
            description=desc,
            price=price,
            api_key=key,
            dry_run=dry_run,
        )
        if uuid:
            success += 1

    print(f"[BOOKERS] 投稿完了: {success}/{len(cache)} 件", flush=True)
    return success


# ══════════════════════════════════════════════════════════════
# エントリポイント
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dry = "--dry-run" in sys.argv
    if dry:
        print("[BOOKERS] ドライランモード（実際には投稿しません）")

    posted = post_predictions(dry_run=dry)
    sys.exit(0 if posted >= 0 else 1)
