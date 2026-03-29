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
RACE_RESULT_URL     = DB_URL + "/race/{race_id}/"
RACE_RESULT_SITE_URL = RACE_TOP_URL + "/race/result.html"   # 静的HTML版（距離取得用）

# ── 競馬場コードマッピング ──────────────────────────────────────
# race_id[8:10] = 競馬場コード
VENUE_CODE_MAP: dict[str, str] = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}

# ── NAR（地方競馬）URL定数・競馬場コードマッピング ───────────────
NAR_TOP_URL      = "https://nar.netkeiba.com"
NAR_CALENDAR_URL = NAR_TOP_URL + "/top/calendar.html"
NAR_RACE_LIST_URL = NAR_TOP_URL + "/top/race_list_sub.html"
NAR_RESULT_URL   = NAR_TOP_URL + "/race/result.html"

NAR_VENUE_CODE_MAP: dict[str, str] = {
    # NAR race_id の形式: YYYY + VV(競馬場) + MM(月) + DD(日) + RR(レース番号)
    # 競馬場コードは race_id[4:6] に格納される（JRAとは位置が異なる）
    # 北海道
    "30": "門別",   "31": "帯広",
    # 東北
    "35": "盛岡",   "36": "水沢",
    # 関東
    "42": "浦和",   "43": "船橋",   "44": "大井",   "45": "川崎",
    # 中部・北陸
    "46": "金沢",   "47": "笠松",   "48": "名古屋",
    # 関西
    "50": "園田",   "51": "姫路",
    # 四国・九州
    "54": "高知",   "55": "佐賀",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://race.netkeiba.com/",
}

DATA_DIR = Path(__file__).parent.parent / "data"


def _sleep():
    """1〜2秒のランダムスリープ（サーバー負荷軽減）"""
    time.sleep(random.uniform(1.0, 2.0))


def _get_result_html_with_playwright(url: str) -> Optional[str]:
    """
    Playwright (Chromium) で結果ページの JS レンダリング後 HTML を返す。
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
            page.goto("https://db.netkeiba.com/", wait_until="domcontentloaded", timeout=20000)
            time.sleep(1)
            page.goto(url, wait_until="networkidle", timeout=30000)
            try:
                page.wait_for_selector(
                    "table.race_table_01, div.ResultTableWrap table, table[summary]",
                    timeout=10000,
                )
            except PWTimeout:
                logger.warning("Playwright: 結果テーブルセレクタのタイムアウト（HTMLをそのまま使用）")
            html = page.content()
            browser.close()
            logger.info(f"Playwright 結果取得成功: {len(html)} bytes")
            return html
    except Exception as e:
        logger.warning(f"Playwright 結果取得失敗: {e}")
        return None


def _get(url: str, session: requests.Session, encoding: str = "EUC-JP") -> Optional[BeautifulSoup]:
    """
    GETリクエストを送り BeautifulSoup を返す。失敗時はNone。

    エンコーディング検出順:
      1. Content-Type ヘッダーの charset
      2. 引数 encoding（デフォルト EUC-JP）
      3. HTML内の <meta charset> タグ（BS4が自動検出）
    resp.content (bytes) + from_encoding を使うことで BS4 に正しく検出させる。
    """
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        # Content-Type ヘッダーから charset を取得（あればそちらを優先）
        ct = resp.headers.get("Content-Type", "")
        m = re.search(r"charset=([^\s;,]+)", ct, re.I)
        detected = m.group(1).strip() if m else encoding
        # bytes + from_encoding: BS4 が <meta charset> も考慮して正しく解析する
        return BeautifulSoup(resp.content, "html.parser", from_encoding=detected)
    except requests.RequestException as e:
        logger.warning(f"Request failed: {url} -> {e}")
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
def _parse_course_distance(text: str, race_info: dict) -> None:
    """テキストからコース種別・距離・天候・馬場状態をrace_infoに抽出する。

    netkeibaではコース種別がimg alt属性に、距離がテキストに分かれて
    いることがある。この関数はテキスト部分のみを担当し、img alt は
    _fill_from_img_alts() で補完する。
    """
    # コース種別＋距離が同一テキスト内にある場合
    # 対応フォーマット: "芝2000m" "芝・右2000m" "ダ1400m (左)" "ダート1800m" "障2000m"
    # ※ netkeibaは "ダ" の略記も使うため (芝|ダ|障) で先頭1文字に合わせる
    m = re.search(r"(芝|ダ|障)[^\d]*(\d{3,4})m", text)
    if m:
        raw_type = m.group(1)
        if raw_type.startswith("障"):
            race_info["course_type"] = "障害"
        elif raw_type.startswith("ダ"):
            race_info["course_type"] = "ダート"
        else:
            race_info["course_type"] = "芝"
        race_info["distance"] = int(m.group(2))
    elif race_info["distance"] is None:
        # コース種別がimg alt等に分離されている場合: 距離数字だけ取得する
        # 例: <img alt="芝"> 2000m → get_text()で "2000m" のみ残る
        # \b は日本語文字との境界で機能しないため使わない
        m = re.search(r"(\d{3,4})m", text)
        if m:
            race_info["distance"] = int(m.group(1))

    # 天候（書式: "天候:晴" / "天候：晴" / "天候 : 晴"）
    # 全角コロン・半角コロン・スペース有無すべて対応
    if race_info["weather"] is None:
        m = re.search(r"天候\s*[:/：]\s*([^\s/　\xa0]+)", text)
        if m:
            val = m.group(1).strip()
            if val:  # \xa0 (NBSP) は除外
                race_info["weather"] = val

    # 馬場状態の取得（2パターンに対応）
    # パターン1（旧/共通）: "馬場:良" / "馬場 : 稍重" / "馬場：重"
    if race_info["track_condition"] is None:
        m = re.search(r"馬場\s*[:/：]\s*([^\s/　\xa0]+)", text)
        if m:
            val = m.group(1).strip()
            if val:
                race_info["track_condition"] = val
        if race_info["track_condition"] is None:
            # パターン2（新フォーマット）: "芝:良" / "芝：不良" / "ダ:稍重" など
            m = re.search(r"(?:芝|ダート?|ダ)\s*[:/：]\s*(良|稍重|重|不良)", text)
            if m:
                race_info["track_condition"] = m.group(1).strip()


def _fill_from_img_alts(el, race_info: dict) -> None:
    """要素内の <img alt="..."> からコース種別・天候・馬場状態を補完する。

    netkeibaでは 芝/ダート/天候/馬場 をアイコン画像で表示しており、
    サーバーサイドHTMLでも alt 属性に値が記載されている。
    """
    COURSE_TYPES  = {"芝", "ダート", "障害"}
    WEATHERS      = {"晴", "曇", "小雨", "雨", "雪", "小雪"}
    TRACK_CONDS   = {"良", "稍重", "重", "不良"}

    for img in el.select("img[alt]"):
        alt = img.get("alt", "").strip()
        if not alt:
            continue
        if alt in COURSE_TYPES and race_info["course_type"] is None:
            race_info["course_type"] = alt
        elif alt in WEATHERS and race_info["weather"] is None:
            race_info["weather"] = alt
        elif alt in TRACK_CONDS and race_info["track_condition"] is None:
            race_info["track_condition"] = alt


def _race_id_to_date(race_id: str) -> str:
    """race_id の先頭8桁（YYYYMMDD）から YYYY-MM-DD 形式の日付を返す。

    HTMLに依存せず、race_id が常に正確な日付を持つことを前提とする。
    """
    return f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}"


def _scrape_meta_from_race_site(
    race_id: str, session: requests.Session, race_info: dict
) -> None:
    """race.netkeiba.com/race/result.html から距離・コース・天候・馬場を補完する。

    db.netkeiba.com のレース情報がJavaScript動的描画のため取得できない場合の
    代替手段。race.netkeiba.com は静的HTMLで距離情報を含む可能性がある。

    取得できた項目のみ race_info を更新し、既存の値は上書きしない。
    接続失敗時はスキップ（ログのみ）。
    """
    url = f"{RACE_RESULT_SITE_URL}?race_id={race_id}"
    try:
        resp = session.get(url, headers={**HEADERS, "Referer": RACE_TOP_URL + "/"}, timeout=15)
        resp.raise_for_status()
        resp.encoding = "EUC-JP"
        soup = BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        logger.debug(f"  [race_site] skipped ({e})")
        return

    # race.netkeiba.com の HTML から距離・コース・天候・馬場を取得
    # 既知セレクタを優先順に試みる
    site_candidates = [
        soup.select_one("div.RaceData01"),
        soup.select_one("div.RaceData"),
        soup.select_one("dl.RaceData"),
        soup.select_one("p.smalltxt"),
        soup.select_one("div.data_intro"),
    ]
    for el in site_candidates:
        if el is None:
            continue
        text = el.get_text(" ", strip=True)
        logger.info(f"  [race_site] {el.name}.{' '.join(el.get('class', []))} {text[:120]!r}")
        _parse_course_distance(text, race_info)
        _fill_from_img_alts(el, race_info)
        for child in el.select("span, dd, p"):
            _parse_course_distance(child.get_text(strip=True), race_info)
            _fill_from_img_alts(child, race_info)

    # <title> タグにコース情報が含まれる場合がある
    # 例: "〇〇特別 | 阪神/芝2000m | race.netkeiba.com"
    title_el = soup.select_one("title")
    if title_el and race_info["distance"] is None:
        _parse_course_distance(title_el.get_text(strip=True), race_info)


def scrape_race_result(race_id: str, session: requests.Session) -> Optional[pd.DataFrame]:
    """
    db.netkeiba.com/race/{race_id}/ からレース結果を取得してDataFrameで返す。

    db.netkeiba.com のレース結果ページには2種類のHTML構造が存在する:
    【新構造】
        <div class="RaceData01">芝・右2000m / 天候:晴 / 馬場:良 / ...</div>
        <div class="RaceData02"><span>2024年1月6日</span><span>...</span></div>
    【旧構造】
        <div class="data_intro">
          <p class="smalltxt">2024年01月06日 / 芝・右2000m / 天候 : 晴 / 馬場 : 良</p>
        </div>

    race_id は YYYYMMDD+会場コード+日次+レース番号（12桁）のため、
    先頭8桁から日付を確実に取得できる。
    """
    url = RACE_RESULT_URL.format(race_id=race_id)
    logger.info(f"結果ページ取得: {url}")

    # Playwright 優先 → requests フォールバック
    soup = None
    html = _get_result_html_with_playwright(url)
    if html:
        soup = BeautifulSoup(html, "html.parser")
        logger.info("Playwright で結果HTML取得成功")
    if soup is None:
        logger.info("requests にフォールバック")
        soup = _get(url, session)
    if soup is None:
        logger.warning(f"結果ページ取得失敗: {race_id}")
        return None

    # デバッグ: HTMLをファイルに保存
    try:
        _debug_dir = DATA_DIR / "debug"
        _debug_dir.mkdir(parents=True, exist_ok=True)
        _debug_path = _debug_dir / f"result_{race_id}.html"
        _debug_path.write_text(soup.prettify(), encoding="utf-8")
        logger.info(f"結果HTML保存: {_debug_path.name}")
    except Exception:
        pass

    # ── レース基本情報（全フィールドをデフォルト値で初期化） ──────
    race_info: dict = {
        "race_id":         race_id,
        "race_name":       "",
        "race_date":       None,
        "course_type":     None,
        "distance":        None,
        "weather":         None,
        "track_condition": None,
    }

    # ── race_date: race_idの先頭8桁から確実に取得（YYYYMMDD形式）──
    if len(race_id) >= 8:
        try:
            race_info["race_date"] = (
                f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}"
            )
        except Exception:
            pass

    # ── レース名 ──────────────────────────────────────────────
    name_el = (
        soup.select_one("h1.RaceName")
        or soup.select_one("div.race_head_inner h1")
        or soup.select_one("div.RaceMainColumn h1")
        or soup.select_one("h2.RaceName")
    )
    race_info["race_name"] = name_el.get_text(strip=True) if name_el else ""

    # ── コース・距離・天候・馬場（複数セレクタ＋img alt でフォールバック） ──
    # netkeibaのHTML構造は頻繁に変わるため、複数の方法で取得を試みる:
    #   1. テキスト抽出 (_parse_course_distance)
    #   2. img alt 抽出 (_fill_from_img_alts) ← コース種別/天候/馬場はアイコン画像が多い
    #   3. data_intro 内 p 要素を個別に解析
    #   4. RaceData01 の span 要素を個別に解析
    #   5. 最終手段: ページ全文から距離だけ抽出
    meta_candidates = [
        soup.select_one("div.RaceData01"),
        soup.select_one("div.data_intro p.smalltxt"),  # data_intro内のp優先
        soup.select_one("p.smalltxt"),
        soup.select_one("div.data_intro"),
        soup.select_one("div.race_head_inner"),
        soup.select_one("div.RaceMainColumn"),
    ]
    for el in meta_candidates:
        if el is None:
            continue
        text = el.get_text(" ", strip=True)
        logger.info(f"  [meta selector={el.name}.{' '.join(el.get('class', []))}] {text[:120]!r}")
        _parse_course_distance(text, race_info)
        _fill_from_img_alts(el, race_info)

    # ── RaceData01 の <span> を個別解析（新レイアウト対応） ──────
    race_data01 = soup.select_one("div.RaceData01")
    if race_data01:
        for span in race_data01.select("span"):
            _parse_course_distance(span.get_text(strip=True), race_info)
            _fill_from_img_alts(span, race_info)

    # ── 距離が取れなければページ全文から (\d{3,4})m を探す ────────
    if race_info["distance"] is None:
        all_text = soup.get_text(" ", strip=True)
        _parse_course_distance(all_text, race_info)
        _fill_from_img_alts(soup, race_info)

    # ── それでも取れなければ race.netkeiba.com を試みる ──────────
    # db.netkeiba.com は距離情報がJavaScript描画のため取得できないことがある。
    # race.netkeiba.com/race/result.html は静的HTMLで提供される可能性がある。
    if race_info["distance"] is None:
        _scrape_meta_from_race_site(race_id, session, race_info)

    # ── 最終フォールバック: レース名から距離を推測 ───────────────
    # 例: "〇〇1800m特別" / "芝2000m" 等の表記が名称中にある場合
    if race_info["distance"] is None and race_info["race_name"]:
        m = re.search(r"(\d{3,4})m", race_info["race_name"])
        if m:
            race_info["distance"] = int(m.group(1))
            logger.info(f"  [distance] race_name から取得: {race_info['distance']}m")

    if race_info["distance"] is None:
        logger.warning(f"  [distance] NOT FOUND for {race_id}")

    # ── race_date: 常に race_id の先頭8桁を使用（最終確定） ──────
    # HTMLには "1970年01月01日"（Unix エポックのJSプレースホルダー）が
    # 含まれることがある。ここで再設定することで確実に上書きを防ぐ。
    # race_id = YYYYMMDDVVRRNN（12桁）の先頭8桁が日付。
    race_info["race_date"] = _race_id_to_date(race_id)

    # ── 競馬場名・リーグをrace_idから設定 ───────────────────────
    venue_code = race_id[8:10] if len(race_id) >= 10 else ""
    race_info["venue"]  = VENUE_CODE_MAP.get(venue_code, venue_code)
    race_info["league"] = "JRA"

    logger.info(
        f"  [race_info] {race_id}: date={race_info['race_date']} "
        f"course={race_info['course_type']} dist={race_info['distance']} "
        f"weather={race_info['weather']} cond={race_info['track_condition']}"
    )

    # ── 着順テーブル ──────────────────────────────────────────
    result_table = (
        soup.select_one("table.race_table_01")
        or soup.select_one("div.ResultTableWrap table")
        or soup.select_one("table[summary*='結果']")
        or soup.select_one("table.nk_tb_common")
    )

    # db.netkeiba.com でテーブルが見つからなければ race.netkeiba.com を試す
    if result_table is None:
        alt_url = f"{RACE_RESULT_SITE_URL}?race_id={race_id}"
        logger.info(f"db.netkeiba でテーブル未検出 → race.netkeiba を試行: {alt_url}")
        alt_html = _get_result_html_with_playwright(alt_url)
        if alt_html:
            alt_soup = BeautifulSoup(alt_html, "html.parser")
            result_table = (
                alt_soup.select_one("table.race_table_01")
                or alt_soup.select_one("div.ResultTableWrap table")
                or alt_soup.select_one("table.Shutuba_Table")
                or alt_soup.select_one("table[summary*='結果']")
                or alt_soup.select_one("table.nk_tb_common")
            )
            if result_table:
                soup = alt_soup  # 以降の解析もこちらのsoupを使う
                logger.info("race.netkeiba から結果テーブル取得成功")
                # デバッグ保存
                try:
                    _debug_path = DATA_DIR / "debug" / f"result_race_{race_id}.html"
                    _debug_path.write_text(alt_soup.prettify(), encoding="utf-8")
                except Exception:
                    pass

    if result_table is None:
        logger.warning(f"Result table not found (both sites): {race_id}")
        return None

    # ── ヘッダからカラムインデックスをマッピング ───────────────
    # db.netkeiba.com のカラム名（表記が変わることがある）
    # 例: 着順,枠番,馬番,馬名,性齢,斤量,騎手,タイム,着差,通過,上り,単勝,人気,馬体重,調教師
    HEADER_ALIASES: dict[str, list[str]] = {
        "finish_position": ["着順"],
        "frame_number":    ["枠番", "枠"],
        "horse_number":    ["馬番"],
        "horse_name":      ["馬名"],
        "sex_age":         ["性齢"],
        "weight_carried":  ["斤量"],
        "jockey":          ["騎手"],
        "time":            ["タイム", "走破タイム"],
        "margin":          ["着差"],
        "odds":            ["単勝"],
        "popularity":      ["人気"],
        "horse_weight":    ["馬体重"],
        "last_3f":         ["上り", "上がり", "上り3F", "上がり3F"],
        "trainer":         ["調教師", "厩舎"],
    }

    # <thead> tr → <th> を優先。なければ最初の <tr> の <th> にフォールバック
    header_row = (
        result_table.select_one("thead tr")
        or result_table.select_one("tr")
    )
    raw_headers = []
    if header_row:
        raw_headers = [th.get_text(strip=True) for th in header_row.select("th")]
        # <th> が無く <td> のみの場合（一部旧フォーマット対応）
        if not raw_headers:
            raw_headers = [td.get_text(strip=True) for td in header_row.select("td")]
    logger.info(f"  [table headers] {raw_headers}")

    # インデックス辞書: フィールド名 -> カラムインデックス
    col_idx: dict[str, int] = {}
    for field, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            for i, h in enumerate(raw_headers):
                if alias in h:
                    col_idx[field] = i
                    break
            if field in col_idx:
                break

    # ヘッダが取れない場合はデフォルトのカラム順を使う
    # db.netkeiba.com race_table_01 実際の列順:
    # 0:着順 1:枠番 2:馬番 3:馬名 4:性齢 5:斤量 6:騎手 7:タイム
    # 8:着差 9:通過(コーナー通過順) 10:上り 11:単勝 12:人気
    # 13:馬体重(増減) 14:調教師
    # ※「通過」列が index=9 に入るため、単勝以降がひとつ後ろにズレる
    DEFAULT_IDX = {
        "finish_position": 0, "frame_number": 1, "horse_number": 2,
        "horse_name": 3, "sex_age": 4, "weight_carried": 5,
        "jockey": 6, "time": 7, "margin": 8,
        "last_3f": 10, "odds": 11, "popularity": 12,
        "horse_weight": 13, "trainer": 14,
    }
    for field, default in DEFAULT_IDX.items():
        if field not in col_idx:
            col_idx[field] = default

    def _td(tds: list, field: str) -> str:
        i = col_idx.get(field, -1)
        if 0 <= i < len(tds):
            return tds[i].get_text(strip=True)
        return ""

    rows = []
    for tr in result_table.select("tr")[1:]:
        tds = tr.select("td")
        if len(tds) < 8:
            continue

        row = dict(race_info)

        row["finish_position"] = _td(tds, "finish_position")
        row["frame_number"]    = _td(tds, "frame_number")
        row["horse_number"]    = _td(tds, "horse_number")

        # 馬名 + horse_id
        hi = col_idx.get("horse_name", 3)
        horse_el = tds[hi].select_one("a") if hi < len(tds) else None
        row["horse_name"] = (horse_el.get_text(strip=True) if horse_el
                             else _td(tds, "horse_name"))
        horse_href = horse_el.get("href", "") if horse_el else ""
        m = re.search(r"/horse/(\w+)", horse_href)
        row["horse_id"] = m.group(1) if m else ""

        row["sex_age"]        = _td(tds, "sex_age")
        row["weight_carried"] = _td(tds, "weight_carried")

        # 騎手 + jockey_id
        ji = col_idx.get("jockey", 6)
        jockey_el = tds[ji].select_one("a") if ji < len(tds) else None
        row["jockey"] = (jockey_el.get_text(strip=True) if jockey_el
                         else _td(tds, "jockey"))
        jockey_href = jockey_el.get("href", "") if jockey_el else ""
        m = (re.search(r"/jockey/result/recent/(\w+)", jockey_href)
             or re.search(r"/jockey/(\w+)", jockey_href))
        row["jockey_id"] = m.group(1) if m else ""

        row["time"]       = _td(tds, "time")
        row["margin"]     = _td(tds, "margin")
        row["odds"]       = _td(tds, "odds")
        row["popularity"] = _td(tds, "popularity")

        # 馬体重（例: "480(-4)"）
        weight_text = _td(tds, "horse_weight")
        m = re.match(r"(\d+)\(([+-]?\d+)\)", weight_text)
        if m:
            row["horse_weight"]      = int(m.group(1))
            row["horse_weight_diff"] = int(m.group(2))
        else:
            row["horse_weight"]      = None
            row["horse_weight_diff"] = None

        # 上がり3ハロン
        row["last_3f"] = _td(tds, "last_3f")

        # 通過順位（コーナー通過順: "1-1-1-1" 形式）
        passing_idx = 9  # デフォルトインデックス
        if passing_idx < len(tds):
            row["passing"] = tds[passing_idx].get_text(strip=True)
        else:
            row["passing"] = ""

        # 調教師 + trainer_id
        ti = col_idx.get("trainer", 13)
        trainer_el = tds[ti].select_one("a") if ti < len(tds) else None
        row["trainer"] = (trainer_el.get_text(strip=True) if trainer_el
                          else _td(tds, "trainer"))
        trainer_href = trainer_el.get("href", "") if trainer_el else ""
        m = (re.search(r"/trainer/result/recent/(\w+)", trainer_href)
             or re.search(r"/trainer/(\w+)", trainer_href))
        row["trainer_id"] = m.group(1) if m else ""

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


# ── NAR（地方競馬）スクレイピング ────────────────────────────────

def scrape_nar_kaisai_dates(year: int, month: int, session: requests.Session) -> list[str]:
    """NAR カレンダーページから指定年月の開催日リストを返す。

    nar.netkeiba.com のカレンダーは直近数年分しか提供しない。
    要求年月の日付が返ってこない場合（過去データ・リダイレクト）は
    race_list_sub.html を各日付で直接リクエストするフォールバックを使う。
    """
    import calendar as _calendar

    ym_prefix = f"{year}{month:02d}"  # "202101" など

    # ── Step1: カレンダーページを試みる ──────────────────────
    url = f"{NAR_CALENDAR_URL}?year={year}&month={month}"
    soup = _get(url, session, encoding="UTF-8")

    dates: list[str] = []
    if soup is not None:
        for a in soup.select("a[href*='kaisai_date=']"):
            href = a.get("href", "")
            m = re.search(r"kaisai_date=(\d{8})", href)
            if m:
                d = m.group(1)
                # 要求した年月と一致するもののみ収集（リダイレクト対策）
                if d.startswith(ym_prefix) and d not in dates:
                    dates.append(d)

    if dates:
        logger.info(f"  [NAR] カレンダー取得: {year}年{month}月 -> {len(dates)}開催日")
        _sleep()
        return dates

    # ── Step2: カレンダーが空/リダイレクト → 日別に race_list_sub.html を試みる ──
    # nar.netkeiba.com は直近データのみ対応のため、過去年分は
    # 各日付で race_list_sub.html をリクエストして開催日を特定する。
    logger.info(
        f"  [NAR] カレンダー空（過去データ非対応の可能性）"
        f" → 日別スキャン開始: {year}年{month}月"
    )
    _, days_in_month = _calendar.monthrange(year, month)
    for day in range(1, days_in_month + 1):
        d = f"{ym_prefix}{day:02d}"
        list_url = f"{NAR_RACE_LIST_URL}?kaisai_date={d}"
        list_soup = _get(list_url, session, encoding="UTF-8")
        if list_soup and list_soup.select("a[href*='race_id=']"):
            logger.info(f"    [NAR] 開催あり: {d}")
            if d not in dates:
                dates.append(d)
        _sleep()

    logger.info(f"  [NAR] 日別スキャン完了: {year}年{month}月 -> {len(dates)}開催日")
    return dates


def scrape_nar_race_ids_for_date(kaisai_date: str, session: requests.Session) -> list[str]:
    """NAR の1開催日のレースID一覧を取得する。"""
    url = f"{NAR_RACE_LIST_URL}?kaisai_date={kaisai_date}"
    soup = _get(url, session, encoding="UTF-8")
    if soup is None:
        return []

    race_ids: list[str] = []

    for a in soup.select("li.RaceList_DataItem a"):
        href = a.get("href", "")
        m = re.search(r"race_id=(\d{12})", href)
        if m:
            rid = m.group(1)
            if rid not in race_ids:
                race_ids.append(rid)

    if not race_ids:
        for a in soup.select("a[href*='race_id=']"):
            href = a.get("href", "")
            m = re.search(r"race_id=(\d{12})", href)
            if m:
                rid = m.group(1)
                if rid not in race_ids:
                    race_ids.append(rid)

    logger.info(f"    [NAR] {kaisai_date}: {len(race_ids)} races")
    _sleep()
    return race_ids


def scrape_nar_race_result(
    race_id: str,
    session: requests.Session,
    kaisai_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    nar.netkeiba.com/race/result.html?race_id={race_id} から
    地方競馬のレース結果を取得してDataFrameで返す。

    NAR の race_id は YYYY + 開催回数(2桁) + 開催日目(2桁) + 競馬場コード(2桁) + レース番号(2桁)
    の形式であり、[4:6] は月ではなく開催回数のため race_id から日付を直接算出できない。
    kaisai_date（カレンダーから取得した YYYYMMDD 文字列）を受け取って race_date に使用する。
    """
    url = f"{NAR_RESULT_URL}?race_id={race_id}"
    soup = _get(url, session, encoding="UTF-8")
    if soup is None:
        return None

    # ── NAR race_id フォーマット: YYYY + VV(競馬場) + MM(月) + DD(日) + RR(レース番号) ──
    # race_id[4:6] = 競馬場コード（30〜55）
    # race_id[6:8] = 月
    # race_id[8:10] = 日
    # race_id[10:12] = レース番号
    # kaisai_date が渡された場合はそちらを優先（より確実）
    if kaisai_date and len(kaisai_date) == 8:
        race_date = f"{kaisai_date[:4]}-{kaisai_date[4:6]}-{kaisai_date[6:8]}"
    elif len(race_id) >= 10:
        # race_id から直接日付を算出（NAR形式 YYYY+VV+MM+DD+RR）
        race_date = f"{race_id[:4]}-{race_id[6:8]}-{race_id[8:10]}"
        # 月・日の妥当性チェック
        try:
            mm, dd = int(race_id[6:8]), int(race_id[8:10])
            if not (1 <= mm <= 12 and 1 <= dd <= 31):
                raise ValueError(f"Invalid mm={mm} dd={dd}")
        except ValueError:
            logger.warning(f"[NAR] race_id {race_id} から日付算出失敗 → HTMLから取得を試みる")
            race_date = None
    else:
        race_date = None

    # HTMLから日付を取得（フォールバック）
    if race_date is None:
        for sel in ["div.RaceData02 span", "span.RaceData", "p.smalltxt", "div.RaceData01"]:
            el = soup.select_one(sel)
            if el:
                dm = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", el.get_text())
                if dm:
                    race_date = f"{dm.group(1)}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}"
                    break
    if race_date is None:
        logger.warning(f"[NAR] race_date not resolved for {race_id}")

    race_info: dict = {
        "race_id":         race_id,
        "race_name":       "",
        "race_date":       race_date,
        "course_type":     None,
        "distance":        None,
        "weather":         None,
        "track_condition": None,
    }

    # ── レース名 ──────────────────────────────────────────────
    name_el = (
        soup.select_one("h1.RaceName")
        or soup.select_one("div.race_head_inner h1")
        or soup.select_one("div.RaceMainColumn h1")
        or soup.select_one("h2.RaceName")
    )
    race_info["race_name"] = name_el.get_text(strip=True) if name_el else ""

    # ── コース・距離・天候・馬場 ────────────────────────────
    meta_candidates = [
        soup.select_one("div.RaceData01"),
        soup.select_one("div.data_intro p.smalltxt"),
        soup.select_one("p.smalltxt"),
        soup.select_one("div.data_intro"),
        soup.select_one("div.race_head_inner"),
    ]
    for el in meta_candidates:
        if el is None:
            continue
        text = el.get_text(" ", strip=True)
        _parse_course_distance(text, race_info)
        _fill_from_img_alts(el, race_info)

    race_data01 = soup.select_one("div.RaceData01")
    if race_data01:
        for span in race_data01.select("span"):
            _parse_course_distance(span.get_text(strip=True), race_info)
            _fill_from_img_alts(span, race_info)

    if race_info["distance"] is None:
        all_text = soup.get_text(" ", strip=True)
        _parse_course_distance(all_text, race_info)

    # ── 競馬場名・リーグ ────────────────────────────────────
    # NAR race_id[4:6] が競馬場コード（JRAは[8:10]と異なる）
    venue_code = race_id[4:6] if len(race_id) >= 6 else ""
    race_info["venue"]  = NAR_VENUE_CODE_MAP.get(venue_code, f"不明({venue_code})")
    race_info["league"] = "NAR"

    logger.info(
        f"  [NAR race_info] {race_id}: date={race_info['race_date']} "
        f"venue={race_info['venue']} course={race_info['course_type']} "
        f"dist={race_info['distance']}"
    )

    # ── 着順テーブル ──────────────────────────────────────────
    # NAR ページのテーブルクラスは JRA と異なる場合があるため複数セレクタを試みる
    result_table = (
        soup.select_one("table.race_table_01")
        or soup.select_one("table.nk_tb_common")
        or soup.select_one("table.ResultTableWrap")
        or soup.select_one("table#ResultTableBody")
        or soup.select_one("div#ResultTableWrap table")
        or soup.select_one("div.RaceTableWrap table")
        or soup.select_one("div.result_table_wrapper table")
    )
    # 上記で見つからない場合: 「着順」列ヘッダを持つ任意のテーブルを探す
    if result_table is None:
        for tbl in soup.select("table"):
            headers_text = tbl.get_text()
            if "着順" in headers_text and "馬名" in headers_text:
                result_table = tbl
                logger.info(f"[NAR] Table found by header keyword: class={tbl.get('class')}")
                break
    if result_table is None:
        logger.warning(f"[NAR] Result table not found: {race_id}")
        # デバッグ用: ページ内の全テーブルを記録
        all_tables = soup.select("table")
        logger.warning(f"[NAR] Tables on page: {[t.get('class') for t in all_tables]}")
        return None

    HEADER_ALIASES: dict[str, list[str]] = {
        "finish_position": ["着順"],
        "frame_number":    ["枠番", "枠"],
        "horse_number":    ["馬番"],
        "horse_name":      ["馬名"],
        "sex_age":         ["性齢"],
        "weight_carried":  ["斤量"],
        "jockey":          ["騎手"],
        "time":            ["タイム", "走破タイム"],
        "margin":          ["着差"],
        "odds":            ["単勝"],
        "popularity":      ["人気"],
        "horse_weight":    ["馬体重"],
        "last_3f":         ["上り", "上がり", "上り3F", "上がり3F"],
        "trainer":         ["調教師", "厩舎"],
    }

    header_row = (
        result_table.select_one("thead tr")
        or result_table.select_one("tr")
    )
    raw_headers = []
    if header_row:
        raw_headers = [th.get_text(strip=True) for th in header_row.select("th")]
        if not raw_headers:
            raw_headers = [td.get_text(strip=True) for td in header_row.select("td")]

    col_idx: dict[str, int] = {}
    for field, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            for i, h in enumerate(raw_headers):
                if alias in h:
                    col_idx[field] = i
                    break
            if field in col_idx:
                break

    DEFAULT_IDX = {
        "finish_position": 0, "frame_number": 1, "horse_number": 2,
        "horse_name": 3, "sex_age": 4, "weight_carried": 5,
        "jockey": 6, "time": 7, "margin": 8,
        "last_3f": 10, "odds": 11, "popularity": 12,
        "horse_weight": 13, "trainer": 14,
    }
    for field, default in DEFAULT_IDX.items():
        if field not in col_idx:
            col_idx[field] = default

    def _td(tds: list, field: str) -> str:
        i = col_idx.get(field, -1)
        if 0 <= i < len(tds):
            return tds[i].get_text(strip=True)
        return ""

    rows = []
    for tr in result_table.select("tr")[1:]:
        tds = tr.select("td")
        if len(tds) < 8:
            continue

        row = dict(race_info)
        row["finish_position"] = _td(tds, "finish_position")
        row["frame_number"]    = _td(tds, "frame_number")
        row["horse_number"]    = _td(tds, "horse_number")

        hi = col_idx.get("horse_name", 3)
        horse_el = tds[hi].select_one("a") if hi < len(tds) else None
        row["horse_name"] = horse_el.get_text(strip=True) if horse_el else _td(tds, "horse_name")
        horse_href = horse_el.get("href", "") if horse_el else ""
        m = re.search(r"/horse/(\w+)", horse_href)
        row["horse_id"] = m.group(1) if m else ""

        row["sex_age"]        = _td(tds, "sex_age")
        row["weight_carried"] = _td(tds, "weight_carried")

        ji = col_idx.get("jockey", 6)
        jockey_el = tds[ji].select_one("a") if ji < len(tds) else None
        row["jockey"] = jockey_el.get_text(strip=True) if jockey_el else _td(tds, "jockey")
        jockey_href = jockey_el.get("href", "") if jockey_el else ""
        m = (re.search(r"/jockey/result/recent/(\w+)", jockey_href)
             or re.search(r"/jockey/(\w+)", jockey_href))
        row["jockey_id"] = m.group(1) if m else ""

        row["time"]       = _td(tds, "time")
        row["margin"]     = _td(tds, "margin")
        row["odds"]       = _td(tds, "odds")
        row["popularity"] = _td(tds, "popularity")

        weight_text = _td(tds, "horse_weight")
        wm = re.match(r"(\d+)\(([+-]?\d+)\)", weight_text)
        if wm:
            row["horse_weight"]      = int(wm.group(1))
            row["horse_weight_diff"] = int(wm.group(2))
        else:
            row["horse_weight"]      = None
            row["horse_weight_diff"] = None

        row["last_3f"] = _td(tds, "last_3f")

        ti = col_idx.get("trainer", 13)
        trainer_el = tds[ti].select_one("a") if ti < len(tds) else None
        row["trainer"] = trainer_el.get_text(strip=True) if trainer_el else _td(tds, "trainer")
        trainer_href = trainer_el.get("href", "") if trainer_el else ""
        m = (re.search(r"/trainer/result/recent/(\w+)", trainer_href)
             or re.search(r"/trainer/(\w+)", trainer_href))
        row["trainer_id"] = m.group(1) if m else ""

        try:
            pos = int(row["finish_position"])
            row["top3"] = 1 if pos <= 3 else 0
        except (ValueError, TypeError):
            row["top3"] = None

        rows.append(row)

    _sleep()
    return pd.DataFrame(rows) if rows else None


def scrape_nar_races(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    指定期間の地方競馬レース結果をすべて取得してCSVに追記保存する。

    JRAデータと同じraw_races.csvに追記し、league列で区別する。
    """
    if output_path is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DATA_DIR / "raw_races.csv"

    session = requests.Session()
    all_dfs: list[pd.DataFrame] = []

    existing_ids: set = set()
    if output_path.exists():
        try:
            existing_df = pd.read_csv(output_path, usecols=["race_id"])
            existing_ids = set(existing_df["race_id"].astype(str))
            logger.info(f"[NAR] 既存データ: {len(existing_ids)} レース")
        except Exception:
            pass

    cur = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    months: list[tuple[int, int]] = []
    while cur <= end:
        months.append((cur.year, cur.month))
        cur = cur.replace(month=cur.month + 1) if cur.month < 12 else cur.replace(year=cur.year + 1, month=1)

    for year, month in months:
        logger.info(f"=== [NAR] {year}年{month}月 取得開始 ===")

        kaisai_dates = scrape_nar_kaisai_dates(year, month, session)
        if not kaisai_dates:
            logger.warning(f"  [NAR] 開催日なし: {year}年{month}月")
            continue

        for kaisai_date in kaisai_dates:
            day_race_ids = scrape_nar_race_ids_for_date(kaisai_date, session)

            for race_id in day_race_ids:
                if race_id in existing_ids:
                    logger.debug(f"    [NAR] SKIP (already exists): {race_id}")
                    continue

                logger.info(f"    [NAR] Scraping: {race_id}")
                df = scrape_nar_race_result(race_id, session, kaisai_date=kaisai_date)
                if df is not None and not df.empty:
                    all_dfs.append(df)
                    existing_ids.add(race_id)

    if not all_dfs:
        logger.warning("[NAR] 新規取得データがありませんでした")
        if output_path.exists():
            return pd.read_csv(output_path)
        return pd.DataFrame()

    new_df = pd.concat(all_dfs, ignore_index=True)

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["race_id", "horse_name"], inplace=True)
    else:
        combined = new_df

    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"[NAR] 保存完了: {output_path} ({len(combined)} rows)")
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
