"""
netkeiba.com の過去レース結果スクレイパー

対象: レース結果ページ（https://db.netkeiba.com/race/）
注意: 過度なアクセスを避けるため1〜2秒のsleepを設けています
"""

import time
import random
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://db.netkeiba.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
DATA_DIR = Path(__file__).parent.parent / "data"


def _sleep():
    """1〜2秒のランダムスリープ（サーバー負荷軽減）"""
    time.sleep(random.uniform(1.0, 2.0))


def _get(url: str, session: requests.Session) -> Optional[BeautifulSoup]:
    """GETリクエストを送り BeautifulSoup を返す。失敗時はNone。"""
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        resp.encoding = "EUC-JP"
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        logger.warning(f"Request failed: {url} -> {e}")
        return None


def build_race_id(year: int, place_code: str, kai: int, day: int, race_num: int) -> str:
    """
    netkeibaのレースIDを生成する。
    例: 2023年 東京(05) 1回 1日目 1R -> "202305010101"
    """
    return f"{year}{place_code}{kai:02d}{day:02d}{race_num:02d}"


# 競馬場コード表
PLACE_CODES = {
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04",
    "東京": "05", "中山": "06", "中京": "07", "京都": "08",
    "阪神": "09", "小倉": "10",
}


def scrape_race_list(year: int, month: int, session: requests.Session) -> list[str]:
    """
    指定年月の開催レースID一覧を取得する。
    netkeibaのカレンダーページから開催日・場・レース番号を収集する。
    """
    race_ids = []
    url = f"{BASE_URL}/?pid=race_list&word=&start_year={year}&start_mon={month}&end_year={year}&end_mon={month}&jyo%5B%5D=&sort=date&list=100"
    soup = _get(url, session)
    if soup is None:
        return race_ids

    # レース結果リンクから race_id を抽出
    for a in soup.select("a[href*='/race/']"):
        href = a["href"]
        m = re.search(r"/race/(\d{12})", href)
        if m:
            race_id = m.group(1)
            if race_id not in race_ids:
                race_ids.append(race_id)

    _sleep()
    return race_ids


def scrape_race_result(race_id: str, session: requests.Session) -> Optional[pd.DataFrame]:
    """
    1レースの結果を取得してDataFrameで返す。

    Returns:
        DataFrame with columns:
            race_id, horse_name, finish_position, time, jockey,
            trainer, odds, popularity, track_condition, weather,
            course_type, distance, weight_carried, horse_weight,
            horse_weight_diff, race_date
    """
    url = f"{BASE_URL}/race/{race_id}/"
    soup = _get(url, session)
    if soup is None:
        return None

    # ── レース基本情報 ──────────────────────────────────────────
    race_info: dict = {"race_id": race_id}

    # レース名・日付
    title_el = soup.select_one("div.race_head_inner h1")
    race_info["race_name"] = title_el.get_text(strip=True) if title_el else ""

    # 日付
    date_el = soup.select_one("div.race_head_inner p.smalltxt")
    if date_el:
        date_text = date_el.get_text()
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", date_text)
        if m:
            race_info["race_date"] = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # コース情報（芝/ダート、距離）
    race_data_el = soup.select_one("div.race_head_inner p.smalltxt")
    if race_data_el:
        text = race_data_el.get_text()
        m = re.search(r"(芝|ダ|障)(\d{3,4})m", text)
        if m:
            race_info["course_type"] = "芝" if m.group(1) == "芝" else ("障害" if m.group(1) == "障" else "ダート")
            race_info["distance"] = int(m.group(2))

    # 天気・馬場状態
    for span in soup.select("div.race_head_inner span"):
        t = span.get_text(strip=True)
        if t.startswith("天候:"):
            race_info["weather"] = t.replace("天候:", "").strip()
        elif t.startswith("馬場:"):
            race_info["track_condition"] = t.replace("馬場:", "").strip()

    # ── 着順テーブル ──────────────────────────────────────────
    result_table = soup.select_one("table.race_table_01")
    if result_table is None:
        logger.warning(f"Result table not found: {race_id}")
        return None

    rows = []
    for tr in result_table.select("tr")[1:]:  # ヘッダ除く
        tds = tr.select("td")
        if len(tds) < 10:
            continue

        row = dict(race_info)

        # 着順
        row["finish_position"] = tds[0].get_text(strip=True)

        # 枠番・馬番
        row["frame_number"] = tds[1].get_text(strip=True)
        row["horse_number"] = tds[2].get_text(strip=True)

        # 馬名
        horse_el = tds[3].select_one("a")
        row["horse_name"] = horse_el.get_text(strip=True) if horse_el else tds[3].get_text(strip=True)
        horse_href = horse_el["href"] if horse_el and horse_el.get("href") else ""
        m = re.search(r"/horse/(\w+)/", horse_href)
        row["horse_id"] = m.group(1) if m else ""

        # 性齢
        row["sex_age"] = tds[4].get_text(strip=True)

        # 斤量
        row["weight_carried"] = tds[5].get_text(strip=True)

        # 騎手
        jockey_el = tds[6].select_one("a")
        row["jockey"] = jockey_el.get_text(strip=True) if jockey_el else tds[6].get_text(strip=True)
        jockey_href = jockey_el["href"] if jockey_el and jockey_el.get("href") else ""
        m = re.search(r"/jockey/(\w+)/", jockey_href)
        row["jockey_id"] = m.group(1) if m else ""

        # タイム
        row["time"] = tds[7].get_text(strip=True)

        # 着差
        row["margin"] = tds[8].get_text(strip=True)

        # 通過順
        row["passing_order"] = tds[10].get_text(strip=True) if len(tds) > 10 else ""

        # 上がり
        row["last_3f"] = tds[11].get_text(strip=True) if len(tds) > 11 else ""

        # オッズ
        row["odds"] = tds[12].get_text(strip=True) if len(tds) > 12 else ""

        # 人気
        row["popularity"] = tds[13].get_text(strip=True) if len(tds) > 13 else ""

        # 馬体重
        weight_text = tds[14].get_text(strip=True) if len(tds) > 14 else ""
        m = re.match(r"(\d+)\(([+-]?\d+)\)", weight_text)
        if m:
            row["horse_weight"] = int(m.group(1))
            row["horse_weight_diff"] = int(m.group(2))
        else:
            row["horse_weight"] = None
            row["horse_weight_diff"] = None

        # 調教師
        trainer_el = tds[18].select_one("a") if len(tds) > 18 else None
        row["trainer"] = trainer_el.get_text(strip=True) if trainer_el else (tds[18].get_text(strip=True) if len(tds) > 18 else "")
        trainer_href = trainer_el["href"] if trainer_el and trainer_el.get("href") else ""
        m = re.search(r"/trainer/(\w+)/", trainer_href)
        row["trainer_id"] = m.group(1) if m else ""

        # 3着以内フラグ（目的変数）
        try:
            pos = int(row["finish_position"])
            row["top3"] = 1 if pos <= 3 else 0
        except ValueError:
            row["top3"] = None  # 除外・中止など

        rows.append(row)

    _sleep()
    return pd.DataFrame(rows) if rows else None


def scrape_races(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    指定期間のレース結果をすべて取得してCSVに保存する。

    Args:
        start_year, start_month: 取得開始年月
        end_year,   end_month:   取得終了年月
        output_path: CSVの保存先（Noneの場合はdata/raw_races.csv）

    Returns:
        結合したDataFrame
    """
    if output_path is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DATA_DIR / "raw_races.csv"

    session = requests.Session()
    all_dfs = []

    # 既存データのrace_id一覧を読み込んでスキップ
    existing_ids: set = set()
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path, usecols=["race_id"])
            existing_ids = set(existing_df["race_id"].astype(str))
            logger.info(f"既存データ: {len(existing_ids)} races")
        except Exception:
            pass

    # 対象年月を列挙
    cur = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    months = []
    while cur <= end:
        months.append((cur.year, cur.month))
        # 翌月へ
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    for year, month in months:
        logger.info(f"月別レースID取得: {year}年{month}月")
        race_ids = scrape_race_list(year, month, session)
        logger.info(f"  -> {len(race_ids)} races found")

        for race_id in race_ids:
            if race_id in existing_ids:
                logger.debug(f"  SKIP (already exists): {race_id}")
                continue

            logger.info(f"  Scraping race: {race_id}")
            df = scrape_race_result(race_id, session)
            if df is not None and not df.empty:
                all_dfs.append(df)
                existing_ids.add(race_id)

    if not all_dfs:
        logger.warning("新規取得データがありませんでした")
        if output_path.exists():
            return pd.read_csv(output_path)
        return pd.DataFrame()

    new_df = pd.concat(all_dfs, ignore_index=True)

    # 既存ファイルとマージして保存
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["race_id", "horse_name"], inplace=True)
    else:
        combined = new_df

    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"保存完了: {output_path} ({len(combined)} rows)")
    return combined


if __name__ == "__main__":
    # 直近1ヶ月分を試験取得
    today = datetime.today()
    df = scrape_races(
        start_year=today.year,
        start_month=today.month,
        end_year=today.year,
        end_month=today.month,
    )
    print(df.head())
