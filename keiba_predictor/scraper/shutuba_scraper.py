"""
出馬表スクレイパー

https://race.netkeiba.com/race/shutuba.html?race_id={race_id}
から出馬情報・レース基本情報を取得する。
"""

import re
import logging
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd

from keiba_predictor.scraper.netkeiba_scraper import _get, HEADERS, VENUE_CODE_MAP

logger = logging.getLogger(__name__)

SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html"


def _get_html_with_playwright(url: str) -> Optional[str]:
    """
    Playwright (Chromium) で JS レンダリング後の HTML を返す。
    playwright 未インストール時は None を返す。
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        logger.warning("playwright 未インストール → requests にフォールバック")
        return None

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent=HEADERS["User-Agent"],
                locale="ja-JP",
                extra_http_headers={
                    "Accept-Language": HEADERS["Accept-Language"],
                    "Accept": HEADERS["Accept"],
                },
            )
            page = ctx.new_page()
            # トップページで Cookie を取得してから出馬表へ
            page.goto("https://race.netkeiba.com/", wait_until="domcontentloaded", timeout=20000)
            time.sleep(1)
            page.goto(url, wait_until="networkidle", timeout=30000)
            # 出馬表テーブルが描画されるまで最大10秒待機
            try:
                page.wait_for_selector(
                    "table.Shutuba_Table, tr.HorseList, td.Umaban",
                    timeout=10000,
                )
            except PWTimeout:
                logger.warning("Playwright: 出馬表セレクタのタイムアウト（HTML をそのまま使用）")
            html = page.content()
            browser.close()
            logger.info(f"Playwright 取得成功: {len(html)} bytes")
            return html
    except Exception as e:
        logger.warning(f"Playwright 取得失敗: {e}")
        return None

# 性別エンコード（data_cleaner.py と合わせる）
_SEX_ENC = {"牡": 0, "牝": 1, "セ": 2, "騸": 2}


def _parse_horse_weight(s: str) -> tuple[Optional[float], Optional[float]]:
    """
    "486(+2)"  → (486.0,  2.0)
    "486(-4)"  → (486.0, -4.0)
    "486"      → (486.0, None)
    """
    if not isinstance(s, str):
        return None, None
    s = s.strip()
    m = re.match(r"(\d+)\s*\(([+-]?\d+)\)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(s), None
    except ValueError:
        return None, None


def _parse_sex_age(s: str) -> tuple[Optional[str], Optional[int]]:
    """"牡3" → ("牡", 3)"""
    if not isinstance(s, str):
        return None, None
    m = re.match(r"([牡牝セ騸])(\d+)", s.strip())
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def _parse_shutuba_row(tr) -> Optional[dict]:
    """<tr class="HorseList"> 1行から馬情報を抽出する。"""

    def _txt(*sels):
        for sel in sels:
            el = tr.select_one(sel)
            if el:
                return el.get_text(strip=True)
        return ""

    def _link_id(sel, pattern):
        el = tr.select_one(sel)
        if el and el.get("href"):
            m = re.search(pattern, el["href"])
            return m.group(1) if m else ""
        return ""

    # 馬番（必須）
    try:
        horse_number = int(_txt(".Umaban", "td.Umaban"))
    except ValueError:
        return None

    # 枠番
    try:
        frame_number = int(_txt(".Waku", "td.Waku"))
    except ValueError:
        frame_number = (horse_number - 1) // 2 + 1

    # 馬名・馬ID
    horse_link = tr.select_one(".HorseName a") or tr.select_one("td.HorseInfo a")
    horse_name = horse_link.get_text(strip=True) if horse_link else ""
    horse_id = ""
    if horse_link and horse_link.get("href"):
        m = re.search(r"/horse/(\w+)/?", horse_link["href"])
        horse_id = m.group(1) if m else ""

    # 性齢
    sex, age = _parse_sex_age(_txt(".Barei", "td.Barei", "td.sexage"))
    sex_enc = _SEX_ENC.get(sex, 0) if sex else 0

    # 斤量
    try:
        weight_carried = float(_txt(".Futan", "td.Futan", "td.Wt"))
    except ValueError:
        weight_carried = None

    # 馬体重
    horse_weight, horse_weight_diff = _parse_horse_weight(
        _txt(".HorseWeight", "td.HorseWeight", "td.Weight")
    )

    # 騎手・騎手ID
    jockey_link = tr.select_one(".Jockey a") or tr.select_one("td.Jockey a")
    jockey = jockey_link.get_text(strip=True) if jockey_link else ""
    jockey_id = ""
    if jockey_link and jockey_link.get("href"):
        m = re.search(r"/jockey/(?:result/recent/)?(\w+)/?", jockey_link["href"])
        jockey_id = m.group(1) if m else ""

    # 調教師・調教師ID
    trainer_link = tr.select_one(".Trainer a") or tr.select_one("td.Trainer a")
    trainer = trainer_link.get_text(strip=True) if trainer_link else ""
    trainer_id = ""
    if trainer_link and trainer_link.get("href"):
        m = re.search(r"/trainer/(?:result/recent/)?(\w+)/?", trainer_link["href"])
        trainer_id = m.group(1) if m else ""

    # オッズ・人気（発走前は "---" の場合あり）
    try:
        odds = float(_txt(".Odds", "td.Odds").replace(",", ""))
    except ValueError:
        odds = None

    try:
        popularity = int(_txt(".Popular", "td.Popular", "td.popular_rank"))
    except ValueError:
        popularity = None

    return {
        "horse_number":       horse_number,
        "frame_number":       frame_number,
        "horse_name":         horse_name,
        "horse_id":           horse_id,
        "sex":                sex or "",
        "sex_enc":            sex_enc,
        "age":                age,
        "weight_carried":     weight_carried,
        "horse_weight":       horse_weight,
        "horse_weight_diff":  horse_weight_diff,
        "jockey":             jockey,
        "jockey_id":          jockey_id,
        "trainer":            trainer,
        "trainer_id":         trainer_id,
        "odds":               odds,
        "popularity":         popularity,
    }


def scrape_shutuba(race_id: str) -> Optional[dict]:
    """
    出馬表ページから馬情報とレース基本情報を取得する。

    Returns:
        {
            "race_id":         str,
            "race_name":       str,
            "race_date":       str,       # "YYYY-MM-DD"
            "venue":           str,
            "course_info":     str,       # "芝1800m"
            "distance":        int,
            "course_type_enc": int,       # 1=芝 / 0=ダート
            "race_grade_enc":  int,
            "horses":          pd.DataFrame,
        }
        取得失敗時は None。
    """
    url = f"{SHUTUBA_URL}?race_id={race_id}"
    logger.info(f"出馬表を取得: {url}")

    # ── 取得方法1: Playwright (JS レンダリング対応) ────────────
    soup = None
    html = _get_html_with_playwright(url)
    if html:
        soup = BeautifulSoup(html, "html.parser")
        logger.info("Playwright で HTML 取得成功")

    # ── 取得方法2: requests フォールバック ─────────────────────
    if soup is None:
        logger.info("requests にフォールバック")
        session = requests.Session()
        session.headers.update(HEADERS)
        try:
            session.get("https://race.netkeiba.com/", headers=HEADERS, timeout=15)
            time.sleep(1.0)
        except Exception:
            pass
        soup = _get(url, session)

    if soup is None:
        logger.error(f"出馬表の取得に失敗: {race_id}")
        return None

    # ── レース基本情報 ─────────────────────────────────────
    race_name = ""
    for sel in (".RaceName", "h1.RaceName", ".RaceTitle"):
        el = soup.select_one(sel)
        if el:
            race_name = el.get_text(strip=True)
            break

    # 日付: race_id[0:8] = YYYYMMDD（最優先）
    raw_date = str(race_id)[:8]
    if len(raw_date) == 8 and raw_date.isdigit():
        race_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
    else:
        race_date = ""
        for sel in (".RaceData01", ".Race_Date", ".RaceInfo"):
            el = soup.select_one(sel)
            if el:
                m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", el.get_text())
                if m:
                    race_date = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
                    break

    # 会場
    venue = VENUE_CODE_MAP.get(str(race_id)[4:6], "")

    # コース・距離
    distance = 0
    course_type_enc = 1  # デフォルト芝
    course_info = ""
    data01 = soup.select_one(".RaceData01") or soup.select_one(".RaceInfo")
    if data01:
        txt = data01.get_text()
        m = re.search(r"(\d{3,4})m", txt)
        if m:
            distance = int(m.group(1))
        if "ダート" in txt or re.search(r"\bダ\b", txt):
            course_type_enc = 0
            course_info = f"ダート{distance}m"
        else:
            course_type_enc = 1
            course_info = f"芝{distance}m"

    # レース格
    from keiba_predictor.features.feature_engineering import _encode_race_grade
    race_grade_enc = _encode_race_grade(race_name)

    # ── 出馬表テーブル ──────────────────────────────────────
    table = (
        soup.select_one("table.Shutuba_Table")
        or soup.select_one("table#shutuba_table")
        or soup.select_one("table.ShutubaTable")
        or soup.select_one("table[class*='Shutuba']")
    )
    rows = []
    if table:
        trs = table.select("tr.HorseList, tr[class*='HorseList']")
        # フォールバック: クラス名でマッチしない場合は <td class="Umaban"> を持つ行を探す
        if not trs:
            trs = [tr for tr in table.find_all("tr") if tr.select_one("td.Umaban")]
        for tr in trs:
            row = _parse_shutuba_row(tr)
            if row:
                rows.append(row)
    else:
        logger.warning("出馬表テーブルが見つかりませんでした（selector: table.Shutuba_Table）")
        # テーブルが見つからない場合: ページ全体から .HorseList 行を検索
        for tr in soup.select("tr.HorseList, tr[class*='HorseList']"):
            row = _parse_shutuba_row(tr)
            if row:
                rows.append(row)

    if not rows:
        logger.warning(f"出馬表の行データが 0 件です（race_id={race_id}）")

    horses_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    logger.info(
        f"出馬表取得完了: {race_name} {race_date} {course_info} / {len(horses_df)}頭"
    )
    return {
        "race_id":         race_id,
        "race_name":       race_name,
        "race_date":       race_date,
        "venue":           venue,
        "course_info":     course_info,
        "distance":        distance,
        "course_type_enc": course_type_enc,
        "race_grade_enc":  race_grade_enc,
        "horses":          horses_df,
    }
