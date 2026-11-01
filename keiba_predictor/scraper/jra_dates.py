"""
race_id から正しい開催日・競馬場を引くためのユーティリティ

JRA の race_id は YYYY + 競馬場(2) + 開催回(2) + 開催日目(2) + レース番号(2) で、
日付を含まない。例: 202405021211 = 2024年 東京(05) 2回 12日目 11R（日本ダービー, 2024-05-26）。
先頭8桁を YYYYMMDD とみなすと 2024-05-02 という誤った日付になる。

そのため JRA は netkeiba の開催カレンダーから作った対応表
（race_id 先頭10桁 → 開催日）で日付を引く。

NAR の race_id は YYYY + 競馬場(2) + MM + DD + レース番号(2) なので、
[6:10] から月日を直接取り出せる。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
KAISAI_MAP_PATH = DATA_DIR / "jra_kaisai_dates.csv"

JRA_VENUE_CODE_MAP: dict[str, str] = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


def load_kaisai_map(path: Path | None = None) -> dict[str, str]:
    """対応表 CSV（kaisai_key, race_date）を {先頭10桁: 'YYYY-MM-DD'} で返す。"""
    path = path or KAISAI_MAP_PATH
    if not path.exists():
        logger.warning(f"JRA開催日対応表がありません: {path}")
        return {}
    df = pd.read_csv(path, dtype=str)
    return dict(zip(df["kaisai_key"], df["race_date"]))


def jra_date_from_race_id(race_id: str, kaisai_map: dict[str, str]) -> str | None:
    rid = str(race_id)
    return kaisai_map.get(rid[:10]) if len(rid) >= 10 else None


def nar_date_from_race_id(race_id: str) -> str | None:
    rid = str(race_id)
    if len(rid) < 10 or not rid[:10].isdigit():
        return None
    mm, dd = int(rid[6:8]), int(rid[8:10])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None
    return f"{rid[:4]}-{rid[6:8]}-{rid[8:10]}"


def jra_venue_from_race_id(race_id: str) -> str | None:
    rid = str(race_id)
    return JRA_VENUE_CODE_MAP.get(rid[4:6]) if len(rid) >= 6 else None


def build_kaisai_map(
    start_year: int,
    end_year: int,
    path: Path | None = None,
    end_month: int = 12,
) -> pd.DataFrame:
    """
    netkeiba の開催カレンダー → 各開催日のレース一覧から、
    race_id 先頭10桁と開催日の対応表を作って CSV に保存する。

    既存 CSV にある開催日は再取得しない（途中で止まっても再実行で続きから取れる）。
    """
    import requests
    from keiba_predictor.scraper.netkeiba_scraper import (
        scrape_kaisai_dates,
        scrape_race_ids_for_date,
    )

    path = path or KAISAI_MAP_PATH
    rows: list[dict] = []
    done_dates: set[str] = set()
    if path.exists():
        old = pd.read_csv(path, dtype=str)
        rows = old.to_dict("records")
        done_dates = set(old["race_date"].str.replace("-", "", regex=False))

    session = requests.Session()
    for year in range(start_year, end_year + 1):
        last_month = end_month if year == end_year else 12
        for month in range(1, last_month + 1):
            for kd in scrape_kaisai_dates(year, month, session):
                if kd in done_dates:
                    continue
                race_ids = scrape_race_ids_for_date(kd, session)
                date_str = f"{kd[:4]}-{kd[4:6]}-{kd[6:8]}"
                for key in sorted({rid[:10] for rid in race_ids if len(rid) == 12 and rid.isdigit()}):
                    rows.append({"kaisai_key": key, "race_date": date_str})
                done_dates.add(kd)
            # 月ごとに保存（途中停止に備える）
            _save(rows, path)

    return _save(rows, path)


def _save(rows: list[dict], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["kaisai_key", "race_date"])
    if df.empty:
        return df
    df = df.drop_duplicates().sort_values(["race_date", "kaisai_key"]).reset_index(drop=True)
    dup = df[df.duplicated("kaisai_key", keep=False)]
    if not dup.empty:
        logger.warning(f"同じ開催キーに複数の日付があります（要確認）:\n{dup.head(20)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return df
