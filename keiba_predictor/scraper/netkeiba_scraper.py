"""
netkeiba.com の過去レース結果スクレイパー

【取得フロー】
  1. カレンダーページで開催日(kaisai_date)を収集
     https://race.netkeiba.com/top/calendar.html?year=YYYY&month=MM
  2. 各開催日の静的HTMLフラグメントからrace_idを収集
     https://race.netkeiba.com/top/race_list_sub.html?kaisai_date=YYYYMMDD
  3. レース結果ページから着順・タイム等を取得
     https://db.netkeiba.com/race/{race_id}/

注意: 過度なアクセスを避けるため1〜2秒のsleepを設けています
"""

import time
import random
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── URL定数 ────────────────────────────────────────────────────
RACE_TOP_URL    = "https://race.netkeiba.com"
DB_URL          = "https://db.netkeiba.com"
CALENDAR_URL    = RACE_TOP_URL + "/top/calendar.html"
RACE_LIST_URL   = RACE_TOP_URL + "/top/race_list_sub.html"  # 静的HTMLフラグメント
RACE_RESULT_URL = DB_URL + "/race/{race_id}/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Referer": "https://race.netkeiba.com/",
}

DATA_DIR = Path(__file__).parent.parent / "data"


def _sleep():
    """1〜2秒のランダムスリープ（サーバー負荷軽減）"""
    time.sleep(random.uniform(1.0, 2.0))


def _get(url: str, session: requests.Session, encoding: str = "EUC-JP") -> Optional[BeautifulSoup]:
    """
    GETリクエストを送り BeautifulSoup を返す。失敗時はNone。
    netkeibaはEUC-JPエンコーディングのため明示指定する。
    """
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        resp.encoding = encoding
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        logger.warning(f"Request failed [{resp.status_code if 'resp' in dir() else 'N/A'}]: {url} -> {e}")
        return None


# ── Step 1: カレンダーページから開催日を取得 ────────────────────────
def scrape_kaisai_dates(year: int, month: int, session: requests.Session) -> list[str]:
    """
    カレンダーページから指定年月の開催日(kaisai_date)リストを返す。

    Returns:
        ["20240106", "20240107", ...] 形式の日付文字列リスト
    """
    url = f"{CALENDAR_URL}?year={year}&month={month}"
    soup = _get(url, session)
    if soup is None:
        return []

    dates: list[str] = []

    # カレンダーの各セルに kaisai_date=YYYYMMDD 形式のリンクが含まれる
    # <td class="RaceCellBox"><a href="...kaisai_date=20240106">6</a></td>
    for a in soup.select("a[href*='kaisai_date=']"):
        href = a.get("href", "")
        m = re.search(r"kaisai_date=(\d{8})", href)
        if m:
            d = m.group(1)
            if d not in dates:
                dates.append(d)

    logger.info(f"  カレンダー取得: {year}年{month}月 -> {len(dates)}開催日")
    _sleep()
    return dates


# ── Step 2: race_list_sub.html からrace_idを取得 ─────────────────
def scrape_race_ids_for_date(kaisai_date: str, session: requests.Session) -> list[str]:
    """
    1開催日のレースID一覧を静的HTMLフラグメントから取得する。

    race_list_sub.html はJavaScriptなしで読める静的なHTMLを返す。
    HTMLの構造（確認済み）:
        <div id="RaceTopRace">
          <div class="RaceList_Box">
            <dl class="RaceList_DataList">
              <dd>
                <ul>
                  <li class="RaceList_DataItem">
                    <a class="RaceList_btn02" href="/race/result.html?race_id=202301050801&...">
    """
    url = f"{RACE_LIST_URL}?kaisai_date={kaisai_date}"
    soup = _get(url, session)
    if soup is None:
        return []

    race_ids: list[str] = []

    # パターン1: RaceList_DataItem 内の a タグ（主要パターン）
    for a in soup.select("li.RaceList_DataItem a"):
        href = a.get("href", "")
        m = re.search(r"race_id=(\d{12})", href)
        if m:
            rid = m.group(1)
            if rid not in race_ids:
                race_ids.append(rid)

    # パターン2: href に race_id= が含まれる全 a タグ（フォールバック）
    if not race_ids:
        for a in soup.select("a[href*='race_id=']"):
            href = a.get("href", "")
            m = re.search(r"race_id=(\d{12})", href)
            if m:
                rid = m.group(1)
                if rid not in race_ids:
                    race_ids.append(rid)

    # パターン3: /race/XXXXXXXXXXXX/ 形式のURL（db.netkeiba.com形式）
    if not race_ids:
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            m = re.search(r"/race/(\d{12})/?", href)
            if m:
                rid = m.group(1)
                if rid not in race_ids:
                    race_ids.append(rid)

    logger.info(f"    {kaisai_date}: {len(race_ids)} races")
    _sleep()
    return race_ids


# ── Step 3: 個別レース結果を取得 ──────────────────────────────────
def scrape_race_result(race_id: str, session: requests.Session) -> Optional[pd.DataFrame]:
    """
    db.netkeiba.com/race/{race_id}/ からレース結果を取得してDataFrameで返す。

    db.netkeiba.com のレース結果ページHTML構造（2024年確認済み）:
        <div class="race_head_inner">
          <h1 class="RaceName">...</h1>
          <div class="RaceData01">芝2000m / 天候: 晴 / 馬場: 良</div>
          <div class="RaceData02"><span>...</span><span>...</span></div>
        </div>
        <table class="race_table_01 nk_tb_common">
          <tr>
            <td>[着順][枠][馬番][馬名][性齢][斤量][騎手][タイム][着差][人気][単勝][体重][調教師]</td>
          </tr>
        </table>
    """
    url = RACE_RESULT_URL.format(race_id=race_id)
    soup = _get(url, session)
    if soup is None:
        return None

    # ── レース基本情報（全フィールドをデフォルト値で初期化） ──────
    # ※ 正規表現がマッチしない場合もカラムが必ず存在するようにする
    race_info: dict = {
        "race_id":         race_id,
        "race_name":       "",
        "race_date":       None,
        "course_type":     None,
        "distance":        None,
        "weather":         None,
        "track_condition": None,
    }

    # レース名
    name_el = soup.select_one("h1.RaceName") or soup.select_one("div.race_head_inner h1")
    race_info["race_name"] = name_el.get_text(strip=True) if name_el else ""

    # RaceData01: コース・距離・馬場・天気情報
    # 実際の書式例:
    #   "芝・右2000m / 天候:晴 / 馬場:良 / 発走:15:25"
    #   "ダート・右1800m / 天候:曇 / 馬場:稍重"
    #   "障害・芝3390m"
    data01 = soup.select_one("div.RaceData01") or soup.select_one("p.smalltxt")
    if data01:
        text = data01.get_text()

        # コース種別・距離
        # "芝" / "ダート"(ダで始まる) / "障"(障害)、続いて任意文字・距離数字m
        m = re.search(r"(芝|ダート?|障)[^\d]*(\d{3,4})m", text)
        if m:
            raw_type = m.group(1)
            if raw_type.startswith("障"):
                race_info["course_type"] = "障害"
            elif raw_type.startswith("ダ"):
                race_info["course_type"] = "ダート"
            else:
                race_info["course_type"] = "芝"
            race_info["distance"] = int(m.group(2))
        else:
            logger.debug(f"  distance not found in RaceData01: {text[:80]!r}")

        # 天候（書式: "天候:晴" or "天候：晴" or "天候 : 晴"）
        m = re.search(r"天候\s*[:/：]\s*(\S+)", text)
        if m:
            race_info["weather"] = m.group(1).rstrip("/").strip()

        # 馬場状態（書式: "馬場:良" etc.）
        m = re.search(r"馬場\s*[:/：]\s*(\S+)", text)
        if m:
            race_info["track_condition"] = m.group(1).rstrip("/").strip()

    # RaceData02: 開催日・開催回・開催場所
    # 例: "2024年1月6日 1回中山1日目"
    data02 = soup.select_one("div.RaceData02")
    if data02:
        text = data02.get_text()
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
        if m:
            race_info["race_date"] = (
                f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            )

    # ── 着順テーブル ──────────────────────────────────────────
    # table.race_table_01 or table.nk_tb_common
    result_table = (
        soup.select_one("table.race_table_01")
        or soup.select_one("table.nk_tb_common")
    )
    if result_table is None:
        logger.warning(f"Result table not found: {race_id}")
        return None

    # ヘッダ行からカラム位置を確認する
    header_row = result_table.select_one("tr")
    col_texts = [th.get_text(strip=True) for th in (header_row.select("th") if header_row else [])]
    logger.debug(f"  Table headers: {col_texts}")

    rows = []
    for tr in result_table.select("tr")[1:]:
        tds = tr.select("td")
        if len(tds) < 10:
            continue

        row = dict(race_info)

        # db.netkeiba.com の race_table_01 カラム順（2024年現在）:
        # 0:着順 1:枠番 2:馬番 3:馬名 4:性齢 5:斤量 6:騎手 7:タイム
        # 8:着差 9:単勝 10:人気 11:馬体重 12:(空) 13:調教師 ...

        row["finish_position"] = tds[0].get_text(strip=True)
        row["frame_number"]    = tds[1].get_text(strip=True)
        row["horse_number"]    = tds[2].get_text(strip=True)

        # 馬名 + horse_id
        horse_el = tds[3].select_one("a")
        row["horse_name"] = (horse_el.get_text(strip=True) if horse_el
                             else tds[3].get_text(strip=True))
        horse_href = horse_el.get("href", "") if horse_el else ""
        m = re.search(r"/horse/(\w+)", horse_href)
        row["horse_id"] = m.group(1) if m else ""

        # 性齢
        row["sex_age"] = tds[4].get_text(strip=True)

        # 斤量
        row["weight_carried"] = tds[5].get_text(strip=True)

        # 騎手 + jockey_id
        jockey_el = tds[6].select_one("a")
        row["jockey"] = (jockey_el.get_text(strip=True) if jockey_el
                         else tds[6].get_text(strip=True))
        jockey_href = jockey_el.get("href", "") if jockey_el else ""
        m = re.search(r"/jockey/result/recent/(\w+)", jockey_href) or re.search(r"/jockey/(\w+)", jockey_href)
        row["jockey_id"] = m.group(1) if m else ""

        # タイム
        row["time"] = tds[7].get_text(strip=True)

        # 着差
        row["margin"] = tds[8].get_text(strip=True)

        # 単勝オッズ・人気（カラム位置が変わることがあるため複数パターン対応）
        if len(tds) >= 12:
            row["odds"]       = tds[9].get_text(strip=True)
            row["popularity"] = tds[10].get_text(strip=True)
        else:
            row["odds"]       = ""
            row["popularity"] = ""

        # 馬体重（例: "480(-4)"）
        weight_col = tds[11] if len(tds) > 11 else None
        weight_text = weight_col.get_text(strip=True) if weight_col else ""
        m = re.match(r"(\d+)\(([+-]?\d+)\)", weight_text)
        if m:
            row["horse_weight"]      = int(m.group(1))
            row["horse_weight_diff"] = int(m.group(2))
        else:
            row["horse_weight"]      = None
            row["horse_weight_diff"] = None

        # 上がり3ハロン（カラム9番目あたり、テーブル構造により異なる）
        # db.netkeiba のレース結果テーブルには上がりタイムが含まれることがある
        row["last_3f"] = ""

        # 調教師 + trainer_id（カラム位置: 通常13番目前後）
        trainer_col = tds[13] if len(tds) > 13 else None
        if trainer_col:
            trainer_el = trainer_col.select_one("a")
            row["trainer"] = (trainer_el.get_text(strip=True) if trainer_el
                              else trainer_col.get_text(strip=True))
            trainer_href = trainer_el.get("href", "") if trainer_el else ""
            m = re.search(r"/trainer/result/recent/(\w+)", trainer_href) or re.search(r"/trainer/(\w+)", trainer_href)
            row["trainer_id"] = m.group(1) if m else ""
        else:
            row["trainer"]    = ""
            row["trainer_id"] = ""

        # 目的変数: 3着以内フラグ
        try:
            pos = int(row["finish_position"])
            row["top3"] = 1 if pos <= 3 else 0
        except (ValueError, TypeError):
            row["top3"] = None  # 除外・中止など

        rows.append(row)

    _sleep()
    return pd.DataFrame(rows) if rows else None


# ── 月単位の一括取得 ──────────────────────────────────────────────
def scrape_race_list(year: int, month: int, session: requests.Session) -> list[str]:
    """
    指定年月の全race_idリストを返す（後方互換のため残す）。

    内部でカレンダー → race_list_sub.html の2段階取得を行う。
    """
    dates = scrape_kaisai_dates(year, month, session)
    race_ids: list[str] = []
    for d in dates:
        race_ids.extend(scrape_race_ids_for_date(d, session))
    return race_ids


# ── メイン取得エントリポイント ────────────────────────────────────
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
    all_dfs: list[pd.DataFrame] = []

    # 既存データのrace_id一覧を読み込んでスキップ（差分取得）
    existing_ids: set = set()
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path, usecols=["race_id"])
            existing_ids = set(existing_df["race_id"].astype(str))
            logger.info(f"既存データ: {len(existing_ids)} レース")
        except Exception:
            pass

    # 対象年月を列挙
    cur = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    months: list[tuple[int, int]] = []
    while cur <= end:
        months.append((cur.year, cur.month))
        cur = cur.replace(month=cur.month + 1) if cur.month < 12 else cur.replace(year=cur.year + 1, month=1)

    for year, month in months:
        logger.info(f"=== {year}年{month}月 取得開始 ===")

        # Step1: 開催日リスト取得
        kaisai_dates = scrape_kaisai_dates(year, month, session)
        if not kaisai_dates:
            logger.warning(f"  開催日なし: {year}年{month}月")
            continue

        for kaisai_date in kaisai_dates:
            # Step2: その日のrace_idリスト取得
            day_race_ids = scrape_race_ids_for_date(kaisai_date, session)

            for race_id in day_race_ids:
                if race_id in existing_ids:
                    logger.debug(f"    SKIP (already exists): {race_id}")
                    continue

                # Step3: レース結果取得
                logger.info(f"    Scraping: {race_id}")
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
    # 動作確認: 直近1ヶ月を試験取得
    today = datetime.today()
    df = scrape_races(
        start_year=today.year,
        start_month=today.month,
        end_year=today.year,
        end_month=today.month,
    )
    print(df.head())
