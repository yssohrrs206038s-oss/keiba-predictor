"""
JRA race_id から開催日・競馬場を正しく引けることのテスト。

JRA race_id は YYYY + 競馬場(2) + 開催回(2) + 開催日目(2) + R(2) で日付を含まない。
先頭8桁を日付とみなす旧処理（202405021211 → 2024-05-02）に戻らないことを確認する。
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from keiba_predictor.scraper import jra_dates
from keiba_predictor.scraper.data_cleaner import clean_raw_data

KMAP = {"2024050212": "2024-05-26", "2024060508": "2024-12-22"}


def test_jra_date_uses_kaisai_map_not_race_id_prefix():
    assert jra_dates.jra_date_from_race_id("202405021211", KMAP) == "2024-05-26"
    assert jra_dates.jra_date_from_race_id("202406050811", KMAP) == "2024-12-22"
    assert jra_dates.jra_date_from_race_id("202499999901", KMAP) is None


def test_nar_date_from_race_id():
    assert jra_dates.nar_date_from_race_id("202644101805") == "2026-10-18"
    assert jra_dates.nar_date_from_race_id("202644139905") is None


def test_jra_venue_from_race_id():
    assert jra_dates.jra_venue_from_race_id("202405021211") == "東京"
    assert jra_dates.jra_venue_from_race_id("202406050811") == "中山"


def _raw_row(race_id, race_date, league, venue):
    return {
        "race_id": race_id, "race_name": "x", "race_date": race_date, "league": league,
        "venue": venue, "finish_position": "1", "frame_number": "1", "horse_number": "1",
        "horse_name": "テスト", "sex_age": "牡3", "weight_carried": "57", "odds": "2.0",
        "popularity": "1", "time": "2:24.0", "horse_weight": "480(+2)",
        "course_type": "芝", "distance": "2400", "weather": "晴", "track_condition": "良",
    }


def test_cleaner_repairs_fake_and_missing_jra_dates(tmp_path, monkeypatch):
    mp = tmp_path / "jra_kaisai_dates.csv"
    pd.DataFrame({"kaisai_key": list(KMAP), "race_date": list(KMAP.values())}).to_csv(mp, index=False)
    monkeypatch.setattr(jra_dates, "KAISAI_MAP_PATH", mp)

    raw = pd.DataFrame([
        _raw_row("202405021211", "2024-05-02", "JRA", "12"),   # 旧処理で作られた誤日付
        _raw_row("202406050811", None, "JRA", "京都"),           # 日付欠損
        _raw_row("202644101805", None, "NAR", "大井"),           # NAR 欠損
    ])
    out = clean_raw_data(raw).set_index("race_id")

    assert str(out.loc["202405021211", "race_date"].date()) == "2024-05-26"
    assert str(out.loc["202406050811", "race_date"].date()) == "2024-12-22"
    assert str(out.loc["202644101805", "race_date"].date()) == "2026-10-18"
    assert out.loc["202405021211", "venue"] == "東京"
    assert out.loc["202406050811", "venue"] == "中山"
