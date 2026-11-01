"""
cleaned_races.csv.gz の JRA 行の race_date / venue を正しい値に修正する。

背景:
  旧データクリーナーは JRA race_id の先頭8桁（YYYY+競馬場+開催回）を YYYYMMDD とみなして
  race_date を作っていたため、2021〜2025年の JRA 全レースの日付が誤っていた。
  例: 202405021211（日本ダービー）→ 2024-05-02（正しくは 2024-05-26）
  日付順が崩れると、前走・直近成績・Elo などの「過去」特徴量に未来のレース結果が混入する。

手順:
  1. --build-map で netkeiba の開催カレンダーから対応表 jra_kaisai_dates.csv を作る（再開可能）
  2. 対応表で JRA 行の race_date を全件上書きし、venue を race_id[4:6] から設定する
  3. 対応表に無い JRA レースが閾値を超えたら書き込まずに異常終了する

使い方:
  python scripts/repair_jra_dates.py --build-map --start-year 2021 --dry-run
  python scripts/repair_jra_dates.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from keiba_predictor.scraper.jra_dates import (  # noqa: E402
    KAISAI_MAP_PATH,
    build_kaisai_map,
    jra_venue_from_race_id,
    load_kaisai_map,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("repair_jra_dates")

CLEANED_GZ = ROOT / "keiba_predictor" / "data" / "cleaned_races.csv.gz"

# 修正後に必ず一致すべき既知のレース（race_id: 正しい開催日）
KNOWN = {
    "202405021211": "2024-05-26",  # 第91回 東京優駿
    "202406050811": "2024-12-22",  # 第69回 有馬記念
    "202305021211": "2023-05-28",  # 第90回 東京優駿
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-map", action="store_true", help="開催カレンダーから対応表を作成・追記する")
    ap.add_argument("--start-year", type=int, default=2021)
    ap.add_argument("--end-year", type=int, default=date.today().year)
    ap.add_argument("--dry-run", action="store_true", help="集計のみで cleaned_races.csv.gz を書き換えない")
    ap.add_argument("--max-unmatched-ratio", type=float, default=0.002,
                    help="対応表に無い JRA レースの許容割合（超えたら異常終了）")
    args = ap.parse_args()

    if args.build_map:
        m = build_kaisai_map(args.start_year, args.end_year, end_month=12)
        logger.info(f"対応表: {len(m)} 開催キー → {KAISAI_MAP_PATH}")

    kaisai_map = load_kaisai_map()
    if not kaisai_map:
        logger.error("対応表が空です。--build-map を付けて実行してください。")
        return 1

    df = pd.read_csv(CLEANED_GZ, encoding="utf-8-sig", dtype=str)
    logger.info(f"読み込み: {len(df)} 行")

    rid = df["race_id"].astype(str)
    is_jra = df["league"].astype(str).str.upper().eq("JRA")
    jra_races = rid[is_jra].unique()

    mapped = rid[is_jra].str[:10].map(kaisai_map)
    unmatched = sorted(set(rid[is_jra][mapped.isna()]))
    ratio = len(unmatched) / max(len(jra_races), 1)
    logger.info(f"JRA: {len(jra_races)} レース / 対応表に無い: {len(unmatched)} レース ({ratio:.2%})")
    if unmatched:
        logger.info(f"  例: {unmatched[:20]}")

    old_date = df.loc[is_jra, "race_date"].astype(str).str[:10]
    changed = (mapped.notna() & (old_date != mapped)).sum()
    logger.info(f"日付が変わる JRA 行: {changed} / {int(is_jra.sum())}")

    new_venue = rid[is_jra].map(jra_venue_from_race_id)
    venue_changed = (df.loc[is_jra, "venue"].astype(str) != new_venue.astype(str)).sum()
    logger.info(f"venue が変わる JRA 行: {venue_changed} / {int(is_jra.sum())}")

    # 既知レースでの検算
    ok = True
    for r, d in KNOWN.items():
        got = kaisai_map.get(r[:10])
        if r in set(jra_races) or got is not None:
            status = "OK" if got == d else "NG"
            ok &= got == d
            logger.info(f"  検算 {r}: 対応表={got} 正解={d} → {status}")
    if not ok:
        logger.error("既知レースの検算に失敗しました。書き込みを中止します。")
        return 1

    if ratio > args.max_unmatched_ratio:
        logger.error(f"対応表に無いレースが多すぎます（{ratio:.2%} > {args.max_unmatched_ratio:.2%}）。書き込みを中止します。")
        return 1

    if args.dry_run:
        logger.info("dry-run のため書き込みません。")
        return 0

    df.loc[is_jra & mapped.reindex(df.index).notna(), "race_date"] = mapped.dropna()
    df.loc[is_jra, "venue"] = new_venue
    # 対応表に無い JRA 行は日付不明として空にする（誤った日付を残さない）
    df.loc[is_jra & mapped.reindex(df.index).isna(), "race_date"] = pd.NA

    df.to_csv(CLEANED_GZ, index=False, encoding="utf-8-sig", compression="gzip")
    logger.info(f"保存: {CLEANED_GZ} ({len(df)} 行)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
