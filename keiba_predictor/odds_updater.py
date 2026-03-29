"""
当日開催レースのオッズをリアルタイム取得して EV スコアを再計算する。

使い方:
    python -m keiba_predictor.odds_updater
"""

import json
import logging
from datetime import date
from keiba_predictor.model.predict import _ensure_utf8_stdout
_ensure_utf8_stdout()
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR  = Path(__file__).parent / "data"
PRED_CACHE = DATA_DIR / "predictions_cache.json"


# ══════════════════════════════════════════════════════════════
# キャッシュ I/O
# ══════════════════════════════════════════════════════════════

def _load_cache() -> dict:
    if PRED_CACHE.exists():
        try:
            return json.loads(PRED_CACHE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"キャッシュ読み込み失敗: {e}")
    return {}


def _save_cache(cache: dict) -> None:
    PRED_CACHE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"predictions_cache.json 保存完了 ({len(cache)} レース)")


# ══════════════════════════════════════════════════════════════
# オッズ更新（1レース）
# ══════════════════════════════════════════════════════════════

def update_odds_for_race(race_id: str, entry: dict) -> dict:
    """出馬表をスクレイピングして EV を再計算したエントリを返す。
    取得失敗時は元のエントリをそのまま返す。"""
    from keiba_predictor.scraper.shutuba_scraper import scrape_shutuba

    try:
        shutuba = scrape_shutuba(race_id)
    except Exception as e:
        logger.warning(f"[{race_id}] 出馬表取得失敗: {e}")
        return entry

    if shutuba is None:
        logger.warning(f"[{race_id}] 出馬表が取得できませんでした")
        return entry

    horses_df = shutuba["horses"]


    # horse_number → odds / popularity のマップを構築
    odds_map: dict[int, float] = {}
    pop_map:  dict[int, int]   = {}
    for _, row in horses_df.iterrows():
        num = row["horse_number"]
        if not pd.notna(num):
            continue
        num = int(num)
        try:
            o = pd.to_numeric(row["odds"], errors="coerce")
            if pd.notna(o):
                odds_map[num] = float(o)
        except (KeyError, TypeError):
            pass
        try:
            p = pd.to_numeric(row["popularity"], errors="coerce")
            if pd.notna(p):
                pop_map[num] = int(p)
        except (KeyError, TypeError):
            pass

    logger.info(f"[{race_id}] odds_map: {len(odds_map)}頭")

    if not odds_map:
        logger.warning(f"[{race_id}] オッズデータが0件 → スキップ")
        return entry

    logger.info(f"[{race_id}] オッズ取得: {len(odds_map)}頭分")

    # honmei / taikou / ana のオッズも更新
    for role in ("honmei", "taikou", "ana"):
        h = entry.get(role, {})
        num = h.get("horse_number")
        if num and num in odds_map:
            h["odds"] = round(odds_map[num], 1)

    # ev_top3 の EV を再計算
    ev_top3 = entry.get("ev_top3", [])
    for h in ev_top3:
        num = h.get("horse_number")
        if num and num in odds_map:
            new_odds = odds_map[num]
            h["odds"]     = round(new_odds, 1)
            h["ev_score"] = round(h["prob"] * new_odds, 3)
        if num and num in pop_map:
            h["popularity"] = pop_map[num]

    entry["ev_top3"] = sorted(ev_top3, key=lambda x: x.get("ev_score", 0), reverse=True)

    # ana_horse_num の更新: predicted_top5_nums 外で最高 EV の馬
    top5_set = set(entry.get("predicted_top5_nums", []))
    best_num, best_ev = None, 0.0
    for h in entry["ev_top3"]:
        num = h.get("horse_number")
        ev  = h.get("ev_score", 0.0)
        if num and num not in top5_set and ev > best_ev:
            best_ev, best_num = ev, num
    if best_num:
        entry["ana_horse_num"] = best_num

    return entry


# ══════════════════════════════════════════════════════════════
# メイン処理
# ══════════════════════════════════════════════════════════════

def run_odds_update() -> int:
    """当日開催レースのオッズを更新して保存する。更新件数を返す。"""
    cache = _load_cache()
    if not cache:
        logger.info("predictions_cache.json が空です")
        return 0

    today = date.today().isoformat()  # "YYYY-MM-DD"
    logger.info(f"本日: {today}  キャッシュ内レース数: {len(cache)}")

    updated = 0
    for race_id, entry in cache.items():
        race_date = entry.get("race_date", "")
        race_name = entry.get("race_name", race_id)

        if race_date != today:
            logger.info(f"  スキップ ({race_date} ≠ {today}): {race_name}")
            continue

        logger.info(f"  オッズ更新中: {race_name} ({race_id})")
        cache[race_id] = update_odds_for_race(race_id, entry)
        updated += 1

    if updated == 0:
        logger.info(f"本日 ({today}) 開催のレースが見つかりませんでした")
        return 0

    _save_cache(cache)
    logger.info(f"更新完了: {updated} レース")
    return updated


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_odds_update()
