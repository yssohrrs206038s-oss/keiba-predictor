"""
出馬表スクレイパー

【優先】Yahoo!競馬
  https://keiba.yahoo.co.jp/race/denma/{race_id_short}/
  race_id_short = race_id[2:]  例: 202607010611 → 2607010611

【フォールバック】netkeiba
  https://race.netkeiba.com/race/shutuba.html?race_id={race_id}
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

SHUTUBA_URL       = "https://race.netkeiba.com/race/shutuba.html"
YAHOO_DENMA_URL   = "https://keiba.yahoo.co.jp/race/denma/{race_id_short}/"
YAHOO_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
    "Referer":         "https://keiba.yahoo.co.jp/",
}


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


def _is_cancel_row(tr) -> bool:
    """tr が取消馬かどうかを判定する。"""
    tr_classes = tr.get("class", [])
    if "Cancel" in tr_classes:
        return True
    for td in tr.find_all("td"):
        td_cls = td.get("class") or []
        if isinstance(td_cls, list):
            td_cls_str = " ".join(td_cls)
        else:
            td_cls_str = str(td_cls)
        if "Cancel" in td_cls_str:
            return True
        if td.get_text(strip=True) == "取消":
            return True
    return False


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


def _scrape_yahoo_shutuba(race_id: str) -> Optional[dict]:
    """
    Yahoo!競馬の出馬表ページから馬情報・レース基本情報を取得する。

    取得フィールド: 馬番, 馬名, 騎手, オッズ, 人気
    (斤量・馬体重・騎手ID・調教師などは netkeiba フォールバック側で補完)

    Returns:
        scrape_shutuba() と同じ dict、または取得失敗時は None。
    """
    race_id_short = str(race_id)[2:]   # "202607010611" → "2607010611"
    url = YAHOO_DENMA_URL.format(race_id_short=race_id_short)
    logger.info(f"[Yahoo!] 出馬表を取得: {url}")

    session = requests.Session()
    session.headers.update(YAHOO_HEADERS)
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
    except Exception as e:
        logger.warning(f"[Yahoo!] 取得失敗: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # ── デバッグ: HTML をファイルに保存 ────────────────────────
    try:
        import pathlib
        _debug_dir = pathlib.Path(__file__).parent.parent / "data" / "debug"
        _debug_dir.mkdir(parents=True, exist_ok=True)
        _debug_path = _debug_dir / f"yahoo_shutuba_{race_id}.html"
        _debug_path.write_text(soup.prettify(), encoding="utf-8")
        print(f"[DEBUG][Yahoo!] HTML 保存: {_debug_path} ({_debug_path.stat().st_size} bytes)", flush=True)
    except Exception as _de:
        print(f"[DEBUG][Yahoo!] HTML 保存失敗: {_de}", flush=True)

    # ── レース名 ────────────────────────────────────────────
    race_name = ""
    for sel in ("h1.raceTitle", "h1", ".raceTtl", ".raceNameInner", ".RaceName"):
        el = soup.select_one(sel)
        if el:
            race_name = el.get_text(strip=True)
            break

    # ── 開催日 ──────────────────────────────────────────────
    race_date = ""
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", soup.get_text())
    if m:
        race_date = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # ── 発走時間 ─────────────────────────────────────────────
    start_time = ""
    mt = re.search(r"(\d{1,2}):(\d{2})発走", soup.get_text())
    if mt:
        start_time = f"{int(mt.group(1)):02d}:{mt.group(2)}"

    # ── 会場・コース・距離 ───────────────────────────────────
    venue = VENUE_CODE_MAP.get(str(race_id)[4:6], "")
    distance        = 0
    course_type_enc = 1
    course_info     = ""
    for sel in (".raceData", ".raceCourse", ".RaceData01", ".raceInfo"):
        el = soup.select_one(sel)
        if el:
            txt = el.get_text()
            m2 = re.search(r"(\d{3,4})m", txt)
            if m2:
                distance = int(m2.group(1))
            if "ダート" in txt or "ダ" in txt:
                course_type_enc = 0
                course_info = f"ダート{distance}m"
            else:
                course_type_enc = 1
                course_info = f"芝{distance}m"
            break

    # ── レース格 ────────────────────────────────────────────
    from keiba_predictor.features.feature_engineering import _encode_race_grade
    race_grade_enc = _encode_race_grade(race_name)

    # ── 出馬表行の取得 ───────────────────────────────────────
    # tr.HorseList で直接取得（テーブルセレクター不要）
    trs = soup.select("tr.HorseList")
    print(f"[DEBUG][Yahoo!] HorseList 行数: {len(trs)}", flush=True)
    if not trs:
        logger.warning("[Yahoo!] tr.HorseList が見つかりませんでした")
        return None

    # 全行のtr.classを出力して取消馬を確認
    for idx, tr in enumerate(trs):
        _hel = tr.select_one(".HorseInfo a") or tr.select_one(".HorseName a")
        _hn = _hel.get_text(strip=True) if _hel else "?"
        print(f"[DEBUG][Yahoo!] tr[{idx}] class={tr.get('class')} horse={_hn}", flush=True)
        if idx == 0:
            for j, td in enumerate(tr.find_all("td")):
                print(f"[DEBUG][Yahoo!]   td[{j:02d}] class={td.get('class')} text={td.get_text(strip=True)[:30]!r}", flush=True)

    rows = []
    for tr in trs:
        # 取消馬をスキップ
        tr_classes = tr.get("class", [])
        tds = tr.find_all("td")
        _hel = tr.select_one(".HorseInfo a") or tr.select_one(".HorseName a")
        _hn = _hel.get_text(strip=True) if _hel else "?"

        # 判定1: tr自体のクラス
        if "Cancel" in tr_classes:
            print(f"[SKIP][Yahoo] {_hn}: tr.classにCancel検出 → スキップ", flush=True)
            continue
        # 判定2: td内のクラス
        if any("Cancel" in str(td.get("class", "")) for td in tds):
            print(f"[SKIP][Yahoo] {_hn}: tdにCancel系クラス検出 → スキップ", flush=True)
            continue
        # 判定3: tdテキストに「取消」
        if any(td.get_text(strip=True) == "取消" for td in tds):
            print(f"[SKIP][Yahoo] {_hn}: tdに「取消」テキスト検出 → スキップ", flush=True)
            continue

        def _txt(*sels: str) -> str:
            for sel in sels:
                el = tr.select_one(sel)
                if el:
                    return el.get_text(strip=True)
            return ""

        # 馬番（必須）: Umaban1, Umaban2 ... → [class*='Umaban']
        try:
            horse_number = int(_txt("td[class*='Umaban']"))
        except ValueError:
            continue

        # 枠番: Waku1, Waku2 ... → [class*='Waku']
        try:
            frame_number = int(_txt("td[class*='Waku']"))
        except ValueError:
            frame_number = (horse_number - 1) // 2 + 1

        # 馬名（必須）
        horse_link = tr.select_one(".HorseInfo a") or tr.select_one(".HorseName a")
        horse_name = horse_link.get_text(strip=True) if horse_link else ""
        if not horse_name:
            continue

        # 性齢: .Barei
        sex, age = _parse_sex_age(_txt(".Barei"))
        sex_enc = _SEX_ENC.get(sex, 0) if sex else 0

        # 斤量: td[05]（クラス名なし・インデックス指定）
        try:
            weight_carried = float(tds[5].get_text(strip=True)) if len(tds) > 5 else None
        except (ValueError, IndexError):
            weight_carried = None

        # 騎手: .Jockey a
        jockey_link = tr.select_one(".Jockey a")
        jockey = jockey_link.get_text(strip=True) if jockey_link else ""

        # 調教師: .Trainer a
        trainer_link = tr.select_one(".Trainer a")
        trainer = trainer_link.get_text(strip=True) if trainer_link else ""

        # 馬体重: td.Weight
        horse_weight, horse_weight_diff = _parse_horse_weight(_txt("td.Weight"))

        # オッズ: クラス名ベースで取得（Popular, Txt_R, Odds など）
        odds = None
        odds_el = tr.select_one("td.Popular, td.Odds, td[class*='Odds']")
        if odds_el is None:
            # フォールバック: クラスに 'Txt_R' を含む td から数値を探す
            for td in tds:
                cls = " ".join(td.get("class", []))
                if "Txt_R" in cls or "Popular" in cls:
                    txt = td.get_text(strip=True)
                    if re.match(r"^\d+(\.\d+)?$", txt.replace(",", "")):
                        odds_el = td
                        break
        if odds_el is not None:
            try:
                odds_text = odds_el.get_text(strip=True).replace(",", "")
                print(f"[DEBUG odds] raw='{odds_text}' class={odds_el.get('class')}", flush=True)
                odds = float(odds_text)
            except (ValueError, TypeError) as e:
                print(f"[DEBUG odds] error: {e}", flush=True)
                odds = None
        else:
            # 最終フォールバック: tds[9] を試す（旧構造互換）
            try:
                odds_text = tds[9].get_text(strip=True) if len(tds) > 9 else ""
                if re.match(r"^\d+(\.\d+)?$", odds_text.replace(",", "")):
                    print(f"[DEBUG odds] fallback tds[9] raw='{odds_text}'", flush=True)
                    odds = float(odds_text.replace(",", ""))
            except (ValueError, IndexError):
                pass

        # 人気: .Popular_Ninki
        try:
            popularity = int(_txt(".Popular_Ninki"))
        except ValueError:
            popularity = None

        rows.append({
            "horse_number":       horse_number,
            "frame_number":       frame_number,
            "horse_name":         horse_name,
            "horse_id":           "",
            "sex":                sex or "",
            "sex_enc":            sex_enc,
            "age":                age,
            "weight_carried":     weight_carried,
            "horse_weight":       horse_weight,
            "horse_weight_diff":  horse_weight_diff,
            "jockey":             jockey,
            "jockey_id":          "",
            "trainer":            trainer,
            "trainer_id":         "",
            "odds":               odds,
            "popularity":         popularity,
        })

    if not rows:
        logger.warning(f"[Yahoo!] 馬データが 0 件です (race_id={race_id})")
        return None

    horses_df = pd.DataFrame(rows)
    logger.info(f"[Yahoo!] 取得完了: {race_name} {race_date} {start_time} {course_info} / {len(horses_df)}頭")

    return {
        "race_id":         race_id,
        "race_name":       race_name,
        "race_date":       race_date,
        "start_time":      start_time,
        "venue":           venue,
        "course_info":     course_info,
        "distance":        distance,
        "course_type_enc": course_type_enc,
        "race_grade_enc":  race_grade_enc,
        "horses":          horses_df,
    }


def _parse_shutuba_row(tr) -> Optional[dict]:
    """<tr class="HorseList"> 1行から馬情報を抽出する。取消馬は None を返す。"""
    # 取消馬をスキップ
    if _is_cancel_row(tr):
        _horse_el = tr.select_one(".HorseName a") or tr.select_one(".HorseInfo a")
        _name = _horse_el.get_text(strip=True) if _horse_el else "?"
        print(f"[CANCEL SKIP][netkeiba] {_name}: tr.class={tr.get('class', [])}", flush=True)
        logger.info(f"[netkeiba] 取消馬スキップ: {_name}")
        return None

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
    # 実際のクラスは "Umaban1" のように数字付き → [class*='Umaban'] で部分一致
    try:
        horse_number = int(_txt("td[class*='Umaban']", ".Umaban", "td.Umaban"))
    except ValueError:
        return None

    # 枠番
    # 実際のクラスは "Waku1" のように数字付き → [class*='Waku'] で部分一致
    try:
        frame_number = int(_txt("td[class*='Waku']", ".Waku", "td.Waku"))
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

    # 人気: 実際のクラスは "Popular_Ninki"
    try:
        popularity = int(_txt(".Popular_Ninki", "td.Popular_Ninki", ".Popular", "td.Popular", "td.popular_rank"))
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
    # ── 取得方法1: Yahoo!競馬（優先）──────────────────────────
    result = _scrape_yahoo_shutuba(race_id)
    if result is not None and not result["horses"].empty:
        logger.info(f"[Yahoo!] 出馬表取得成功: {result['race_name']} / {len(result['horses'])}頭")
        return result

    logger.warning(f"[Yahoo!] 取得失敗または馬データなし → netkeiba にフォールバック (race_id={race_id})")

    # ── 取得方法2: netkeiba Playwright ────────────────────────
    url = f"{SHUTUBA_URL}?race_id={race_id}"
    logger.info(f"[netkeiba] 出馬表を取得: {url}")

    soup = None
    html = _get_html_with_playwright(url)
    if html:
        soup = BeautifulSoup(html, "html.parser")
        logger.info("[netkeiba] Playwright で HTML 取得成功")

    # ── 取得方法3: netkeiba requests ──────────────────────────
    if soup is None:
        logger.info("[netkeiba] requests にフォールバック")
        session = requests.Session()
        session.headers.update(HEADERS)
        try:
            session.get("https://race.netkeiba.com/", headers=HEADERS, timeout=15)
            time.sleep(1.0)
        except Exception:
            pass
        soup = _get(url, session)

    if soup is None:
        logger.error(f"[netkeiba] 出馬表の取得に失敗: {race_id}")
        return None

    # ── デバッグ: 取得HTMLをファイルに保存 ─────────────────────
    try:
        import pathlib
        _debug_dir = pathlib.Path(__file__).parent.parent / "data" / "debug"
        _debug_dir.mkdir(parents=True, exist_ok=True)
        _debug_path = _debug_dir / f"shutuba_{race_id}.html"
        _debug_path.write_text(soup.prettify(), encoding="utf-8")
        print(f"[DEBUG][netkeiba] HTML 保存: {_debug_path} ({_debug_path.stat().st_size} bytes)", flush=True)
    except Exception as _de:
        print(f"[DEBUG][netkeiba] HTML 保存失敗: {_de}", flush=True)

    # ── レース基本情報 ─────────────────────────────────────
    race_name = ""
    for sel in (".RaceName", "h1.RaceName", ".RaceTitle"):
        el = soup.select_one(sel)
        if el:
            race_name = el.get_text(strip=True)
            break

    # 日付: ページHTMLから取得（race_id の構造は YYYY+場コード+開催回+日目+レース番号 のため
    # race_id から日付は導出できない）
    race_date = ""
    for sel in (".RaceData01", ".Race_Date", ".RaceInfo", ".RaceList_DataItem"):
        el = soup.select_one(sel)
        if el:
            m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", el.get_text())
            if m:
                race_date = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
                break
    if not race_date:
        # フォールバック: ページ全体から日付パターンを検索
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", soup.get_text())
        if m:
            race_date = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    if race_date:
        logger.info(f"開催日をHTMLから取得: {race_date}")
    else:
        logger.warning(f"開催日をHTMLから取得できませんでした (race_id={race_id})")

    # 発走時間
    start_time = ""
    mt = re.search(r"(\d{1,2}):(\d{2})発走", soup.get_text())
    if mt:
        start_time = f"{int(mt.group(1)):02d}:{mt.group(2)}"

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
    # デバッグ: テーブルとページ内の全テーブル一覧を出力
    all_tables = soup.find_all("table")
    print(f"[DEBUG] ページ内テーブル数: {len(all_tables)}", flush=True)
    for i, t in enumerate(all_tables[:10]):
        print(f"[DEBUG]   table[{i}] class={t.get('class')} id={t.get('id')}", flush=True)
    print(f"[DEBUG] Shutuba_Table 検出: {table is not None}", flush=True)

    rows = []
    if table:
        trs = table.select("tr.HorseList, tr[class*='HorseList']")
        # フォールバック: クラス名でマッチしない場合は Umaban 系クラスを持つ td を含む行を探す
        if not trs:
            trs = [tr for tr in table.find_all("tr")
                   if tr.select_one("td[class*='Umaban']") or tr.select_one("td.Umaban")]
        print(f"[DEBUG] HorseList 行数: {len(trs)}", flush=True)
        # 全行のtr.classと馬名を出力
        for idx, tr in enumerate(trs):
            _hel = tr.select_one(".HorseName a") or tr.select_one(".HorseInfo a")
            _hn = _hel.get_text(strip=True) if _hel else "?"
            print(f"[DEBUG][netkeiba] tr[{idx}] class={tr.get('class')} horse={_hn}", flush=True)
        for tr in trs:
            tr_classes = tr.get("class", [])
            _tds = tr.find_all("td")
            _hel = tr.select_one(".HorseName a") or tr.select_one(".HorseInfo a")
            _hn = _hel.get_text(strip=True) if _hel else "?"
            if "Cancel" in tr_classes:
                print(f"[SKIP][netkeiba] {_hn}: tr.classにCancel検出 → スキップ", flush=True)
                continue
            if any("Cancel" in str(td.get("class", "")) for td in _tds):
                print(f"[SKIP][netkeiba] {_hn}: tdにCancel系クラス検出 → スキップ", flush=True)
                continue
            if any(td.get_text(strip=True) == "取消" for td in _tds):
                print(f"[SKIP][netkeiba] {_hn}: tdに「取消」テキスト検出 → スキップ", flush=True)
                continue
            row = _parse_shutuba_row(tr)
            if row:
                rows.append(row)
    else:
        logger.warning("出馬表テーブルが見つかりませんでした（selector: table.Shutuba_Table）")
        # テーブルが見つからない場合: ページ全体から .HorseList 行を検索
        horse_trs = soup.select("tr.HorseList, tr[class*='HorseList']")
        print(f"[DEBUG] ページ全体の HorseList 行数: {len(horse_trs)}", flush=True)
        for idx, tr in enumerate(horse_trs):
            _hel = tr.select_one(".HorseName a") or tr.select_one(".HorseInfo a")
            _hn = _hel.get_text(strip=True) if _hel else "?"
            print(f"[DEBUG][netkeiba-fb] tr[{idx}] class={tr.get('class')} horse={_hn}", flush=True)
        for tr in horse_trs:
            tr_classes = tr.get("class", [])
            _tds = tr.find_all("td")
            _hel = tr.select_one(".HorseName a") or tr.select_one(".HorseInfo a")
            _hn = _hel.get_text(strip=True) if _hel else "?"
            if "Cancel" in tr_classes:
                print(f"[SKIP][netkeiba-fb] {_hn}: tr.classにCancel検出 → スキップ", flush=True)
                continue
            if any("Cancel" in str(td.get("class", "")) for td in _tds):
                print(f"[SKIP][netkeiba-fb] {_hn}: tdにCancel系クラス検出 → スキップ", flush=True)
                continue
            if any(td.get_text(strip=True) == "取消" for td in _tds):
                print(f"[SKIP][netkeiba-fb] {_hn}: tdに「取消」テキスト検出 → スキップ", flush=True)
                continue
            row = _parse_shutuba_row(tr)
            if row:
                rows.append(row)

    if not rows:
        logger.warning(f"出馬表の行データが 0 件です（race_id={race_id}）")

    horses_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    logger.info(
        f"出馬表取得完了: {race_name} {race_date} {start_time} {course_info} / {len(horses_df)}頭"
    )
    return {
        "race_id":         race_id,
        "race_name":       race_name,
        "race_date":       race_date,
        "start_time":      start_time,
        "venue":           venue,
        "course_info":     course_info,
        "distance":        distance,
        "course_type_enc": course_type_enc,
        "race_grade_enc":  race_grade_enc,
        "horses":          horses_df,
    }
