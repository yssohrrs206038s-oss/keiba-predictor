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

    # 大口投票検知
    try:
        from keiba_predictor.odds_monitor import detect_large_bets
        horses_for_detect = []
        for _, row in horses_df.iterrows():
            num = row.get("horse_number")
            if not pd.notna(num):
                continue
            o = pd.to_numeric(row.get("odds"), errors="coerce")
            fo = pd.to_numeric(row.get("fukusho_odds"), errors="coerce") if "fukusho_odds" in row.index else None
            p = pd.to_numeric(row.get("popularity"), errors="coerce")
            horses_for_detect.append({
                "horse_number": int(num),
                "horse_name": str(row.get("horse_name", "")),
                "odds": float(o) if pd.notna(o) else 0,
                "popularity": int(p) if pd.notna(p) else 0,
                "fukusho_odds": float(fo) if pd.notna(fo) else None,
            })
        flags = detect_large_bets(race_id, horses_for_detect)
        if flags:
            entry["large_bet_flags"] = [
                {"horse_number": f["horse_number"], "horse_name": f["horse_name"],
                 "rank": f["rank"], "score": f["score"],
                 "gap_score": f["gap_score"], "divergence_score": f["divergence_score"]}
                for f in flags
            ]
            logger.info(f"[{race_id}] 大口検知: {len(flags)}頭")
        else:
            entry.pop("large_bet_flags", None)
    except Exception as e:
        logger.debug(f"[{race_id}] 大口検知失敗: {e}")

    return entry


# ══════════════════════════════════════════════════════════════
# メイン処理
# ══════════════════════════════════════════════════════════════

def _build_ev_rank_message(cache: dict) -> str:
    """発走2時間以内のレースで EV≥1.2 の馬を抽出して直前期待値ランクメッセージを生成する。"""
    from datetime import datetime

    now = datetime.now()
    now_str = now.strftime("%H:%M")
    SEP = "━" * 20

    race_blocks: list[str] = []

    for race_id, entry in cache.items():
        race_date = entry.get("race_date", "")
        if race_date != date.today().isoformat():
            continue

        # 発走2時間以内か判定
        start_time = entry.get("start_time", "")
        if start_time:
            try:
                st = datetime.strptime(f"{race_date} {start_time}", "%Y-%m-%d %H:%M")
                diff_min = (st - now).total_seconds() / 60
                if diff_min < 0 or diff_min > 120:
                    continue
            except Exception:
                continue
        else:
            continue

        race_name = entry.get("race_name", race_id)

        # EV≥1.2 の馬を収集
        ev_horses: list[dict] = []
        for h in entry.get("ev_top3", []):
            ev = h.get("ev_score", 0)
            if ev >= 1.2:
                ev_horses.append(h)

        # predicted_top5 からも補完
        top5 = entry.get("predicted_top5", [])
        seen = {h.get("horse_number") for h in ev_horses}
        for h in top5:
            num = h.get("horse_number")
            odds = h.get("odds")
            prob = h.get("prob", 0)
            if num not in seen and odds and prob:
                ev = prob * float(odds)
                if ev >= 1.2:
                    ev_horses.append({
                        "horse_number": num,
                        "horse_name": h.get("horse_name", ""),
                        "ev_score": round(ev, 2),
                        "prob": prob,
                        "odds": odds,
                    })

        if not ev_horses:
            continue

        ev_horses.sort(key=lambda x: -x.get("ev_score", 0))
        lines = [f"🏇 {race_name}"]
        for h in ev_horses[:5]:
            num  = h.get("horse_number", "?")
            name = h.get("horse_name", "")
            ev   = h.get("ev_score", 0)
            prob = h.get("prob", 0) * 100
            odds = h.get("odds", 0)

            if ev >= 5.0:
                icon = "🔥"
            elif ev >= 2.0:
                icon = "✅"
            else:
                icon = "⚠️"

            note = ""
            pop = h.get("popularity")
            if pop and int(pop) <= 2 and ev < 2.0:
                note = "・過剰人気"

            lines.append(
                f"{icon} {num}番{name} EV{ev:.2f}"
                f"（確率{prob:.0f}% × {float(odds):.1f}倍{note}）"
            )

        race_blocks.append("\n".join(lines))

    if not race_blocks:
        return ""

    header = [
        SEP,
        f"📊 直前期待値ランク更新（{now_str}時点）",
        SEP,
    ]
    footer = [
        SEP,
        "EV1.2以上 = 買い得、1.0未満 = 割高",
    ]

    return "\n".join(header + ["\n".join(race_blocks)] + footer)


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

    # 直前期待値ランクを Discord に送信
    import os
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if webhook_url:
        msg = _build_ev_rank_message(cache)
        if msg:
            try:
                import requests
                resp = requests.post(webhook_url, json={"content": msg}, timeout=15)
                ok = resp.status_code in (200, 204)
                logger.info(f"直前期待値ランク Discord送信{'完了' if ok else '失敗'}")
            except Exception as e:
                logger.warning(f"直前期待値ランク Discord送信失敗: {e}")

    # 大口投票検知アラートを Discord に送信
    if webhook_url:
        try:
            from keiba_predictor.odds_monitor import format_large_bet_alert
            for race_id, entry in cache.items():
                flags = entry.get("large_bet_flags", [])
                if flags:
                    race_name = entry.get("race_name", race_id)
                    alert = format_large_bet_alert(race_name, flags)
                    if alert:
                        import requests as _req
                        _req.post(webhook_url, json={"content": alert}, timeout=15)
                        logger.info(f"大口検知アラート送信: {race_name}")
        except Exception as e:
            logger.warning(f"大口検知アラート送信失敗: {e}")

    return updated


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_odds_update()
