"""
週末重賞 自動予想・結果通知 → Discord

【機能1】毎週金曜 09:00 ── 週末重賞の予想を送信
    python -m keiba_predictor.main notify --mode predict

【機能2】毎週日曜 17:00 ── 重賞レースの結果・的中判定を送信
    python -m keiba_predictor.main notify --mode result

【環境変数】
    DISCORD_WEBHOOK_URL : Discord Incoming Webhook URL

【前提条件】
    学習済みモデル: keiba_predictor/model/xgb_model.pkl
    予想はpredict_live()で出馬表を直接スクレイピングするため
    featured_races.csvは不要（キャッシュ優先運用）
"""

import json
import logging
import os
import re
import time
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from keiba_predictor.scraper.netkeiba_scraper import (
    _get, _sleep, RACE_RESULT_URL,
)
from keiba_predictor.model.predict import load_model, predict_race, calc_ev_and_flags, format_prediction, _build_course_info
# ai_comment は簡素化のため無効化（将来の復活用にファイルは残す）
# from keiba_predictor.ai_comment import generate_comments

logger = logging.getLogger(__name__)

# ── パス定数 ────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model" / "xgb_model.pkl"
PRED_CACHE = DATA_DIR / "predictions_cache.json"   # 予想キャッシュ
MANUAL_RESULTS = DATA_DIR / "manual_results.json"  # 手動結果入力

# 重賞判定 (G1/G2/G3 を含む括弧表記)
GRADE_RE = re.compile(r"\(G[Ⅰ-Ⅲ1-3]\)|\(GI{1,3}\)")

MARK = {"honmei": "◎", "taikou": "○", "ana": "△", "hoshi": "☆"}

# JRA 競馬場コード → 場名
VENUE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# Discord 送信
# ══════════════════════════════════════════════════════════════

def send_discord(webhook_url: str, content: str) -> bool:
    """Discord Webhook にメッセージを送信する。2000 字超は自動分割。"""
    if not webhook_url:
        logger.error("Discord Webhook URL が未設定です")
        return False
    chunks = [content[i : i + 1900] for i in range(0, len(content), 1900)]
    ok = True
    for idx, chunk in enumerate(chunks):
        print(f"[Discord送信] chunk {idx + 1}/{len(chunks)} ({len(chunk)}文字):\n{chunk}", flush=True)
        try:
            # ensure_ascii=False で絵文字(📝等)をUTF-8のまま送信
            payload = json.dumps({"content": chunk}, ensure_ascii=False).encode("utf-8")
            r = requests.post(
                webhook_url,
                data=payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
                timeout=15,
            )
            if r.status_code not in (200, 204):
                logger.error(f"Discord 送信失敗: {r.status_code} {r.text[:200]}")
                ok = False
            else:
                logger.info(f"  Discord 送信OK chunk {idx + 1}/{len(chunks)} ({len(chunk)}文字)")
                time.sleep(1)
        except requests.RequestException as e:
            logger.error(f"Discord 送信エラー: {e}")
            ok = False
    return ok


# ══════════════════════════════════════════════════════════════
# 今週末のレース取得
# ══════════════════════════════════════════════════════════════

def _weekend_dates() -> list[str]:
    """今週末（土・日）の YYYYMMDD リストを返す。月〜土曜実行を想定。"""
    from datetime import datetime, timezone, timedelta as _td
    today   = (datetime.now(timezone.utc) + _td(hours=9)).date()  # JST基準
    wd      = today.weekday()          # 0=月 … 5=土 6=日
    if   wd == 5: d = 0                # 土 → 当日
    elif wd == 6: d = -1               # 日 → 昨日=土
    else:         d = 5 - wd          # 月(4)・火(3)・水(3)・木(2)・金(1) → 今週土
    sat = today + timedelta(days=d)
    sun = sat + timedelta(days=1)
    return [sat.strftime("%Y%m%d"), sun.strftime("%Y%m%d")]


def _detect_grade(el) -> str:
    """BeautifulSoup要素からグレード（"GI"/"GII"/"GIII"）を検出して返す。

    対象クラス（完全一致）:
      icon_gradetype1 → GI
      icon_gradetype2 → GII
      icon_gradetype3 → GIII
    Icon_GradeType16/17/18 などリステッド・オープン・地方重賞はスキップ（""を返す）。
    """
    # クラス名 → グレード文字列のマッピング
    CLASS_GRADE = {"icon_gradetype1": "GI", "icon_gradetype2": "GII", "icon_gradetype3": "GIII"}
    # テキスト/alt → グレード文字列のマッピング（正規表現でマッチ後に判定）
    TEXT_GRADE = {
        re.compile(r"G[Ⅰ1]|GI$"):   "GI",
        re.compile(r"G[Ⅱ2]|GII$"):  "GII",
        re.compile(r"G[Ⅲ3]|GIII$"): "GIII",
    }

    # 1. クラス名で判定（Icon_GradeType1/2/3 のみ。5以上はリステッド等で除外）
    GRADE_TYPE_RE = re.compile(r"^icon_gradetype(\d+)$", re.I)
    for child in el.find_all(True):
        for cls in child.get("class", []):
            m = GRADE_TYPE_RE.match(cls.lower())
            if m:
                num = int(m.group(1))
                if num == 1: return "GI"
                if num == 2: return "GII"
                if num == 3: return "GIII"
                # 4以上（リステッド・オープン等）は重賞ではない
                return ""

    # 2. 旧形式テキストアイコン: gradeicon-g1/g2/g3
    GRADE_CLS_RE = re.compile(r"\bgradeicon-g([123])\b", re.I)
    for child in el.find_all(True):
        cls_str = " ".join(child.get("class", []))
        m = GRADE_CLS_RE.search(cls_str)
        if m:
            return {"1": "GI", "2": "GII", "3": "GIII"}.get(m.group(1), "")

    # 3. 全テキストに括弧付きグレード表記 (G1)/(GⅠ)/(GII)/(GⅡ)/(GIII)/(GⅢ)
    text = el.get_text(" ", strip=True)
    m3 = re.search(r"\(G([Ⅰ1])\)|\(GI\)|\(G([Ⅱ2])\)|\(GII\)|\(G([Ⅲ3])\)|\(GIII\)", text)
    if m3:
        full = m3.group(0)
        if re.search(r"GI{3}|GⅢ|G3", full): return "GIII"
        if re.search(r"GI{2}|GⅡ|G2",  full): return "GII"
        return "GI"

    # 4. 単体テキストが "G1"/"GⅠ" 等の子孫要素
    for child in el.find_all(True):
        stext = child.get_text(strip=True)
        if re.fullmatch(r"G[Ⅲ3]|GIII", stext): return "GIII"
        if re.fullmatch(r"G[Ⅱ2]|GII",  stext): return "GII"
        if re.fullmatch(r"G[Ⅰ1]|GI",   stext): return "GI"

    # 5. 画像 alt 属性
    for img in el.find_all("img", alt=True):
        alt = img["alt"].strip()
        if re.fullmatch(r"G[Ⅲ3]|GIII", alt): return "GIII"
        if re.fullmatch(r"G[Ⅱ2]|GII",  alt): return "GII"
        if re.fullmatch(r"G[Ⅰ1]|GI",   alt): return "GI"

    return ""


def _is_grade_race(el) -> bool:
    """BeautifulSoup要素（<li>など）が GI/GII/GIII かどうかを判定する。"""
    return bool(_detect_grade(el))


def _dump_html_for_debug(soup, kaisai_date: str) -> None:
    """取得した soup の HTML をデバッグ用ファイルに保存する。"""
    try:
        debug_dir = DATA_DIR / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / f"race_list_{kaisai_date}.html"
        path.write_text(soup.prettify(), encoding="utf-8")
        logger.info(f"  [debug] HTML保存: {path}")
    except Exception as e:
        logger.debug(f"  [debug] HTML保存失敗: {e}")


def scrape_grade_race_ids(session: requests.Session) -> list[dict]:
    """今週末の重賞レース一覧 [{race_id, race_name, race_date}, ...] を返す。"""
    found: list[dict] = []
    seen:  set[str]   = set()

    dates = _weekend_dates()
    logger.info(f"検索対象日付: {dates[0]} (土) / {dates[1]} (日)")

    # race_list_sub.html（静的フラグメント）と race_list.html の両方を試みる
    LIST_PATHS = ["race_list_sub.html", "race_list.html"]

    for kaisai_date in dates:
        race_date_str = f"{kaisai_date[:4]}-{kaisai_date[4:6]}-{kaisai_date[6:]}"
        found_this_day: list[dict] = []

        for path in LIST_PATHS:
            url = f"https://race.netkeiba.com/top/{path}?kaisai_date={kaisai_date}"
            logger.info(f"取得中: {url}")
            soup = _get(url, session)
            if soup is None:
                logger.warning(f"  取得失敗: {url}")
                continue

            # ── <li class="RaceList_DataItem"> を起点に取得 ──────────
            # グレードアイコンは <a> タグの外 (<li> 直下) に置かれることが多いため
            # <a> ではなく <li> 全体を検査する
            items = soup.select("li.RaceList_DataItem")
            logger.info(f"  {kaisai_date}: {len(items)} RaceList_DataItem発見 ({path})")

            # 最初の取得時にHTMLをデバッグ保存（クラス構造確認用）
            if items:
                _dump_html_for_debug(soup, kaisai_date)

            for li in items:
                # race_id を li 内の a タグから取得
                a_tag = None
                for a in li.select("a[href]"):
                    if re.search(r"race_id=\d{12}", a.get("href", "")):
                        a_tag = a
                        break
                if a_tag is None:
                    continue
                m = re.search(r"race_id=(\d{12})", a_tag.get("href", ""))
                if not m:
                    continue
                race_id = m.group(1)
                if race_id in seen:
                    continue

                # JRA競馬場コード（race_id[4:6]）が01〜10のみ対象（NAR は30以上）
                if race_id[4:6] not in {"01","02","03","04","05","06","07","08","09","10"}:
                    logger.debug(f"    {race_id} スキップ（NAR venue={race_id[4:6]}）")
                    continue

                # レース名（<li> 全体から複数セレクタで試みる）
                name_el = (
                    li.select_one(".Race_Name")
                    or li.select_one(".RaceName")
                    or li.select_one(".RaceList_ItemTitle")
                    or li.select_one(".ItemTitle")
                )
                race_name = (
                    name_el.get_text(strip=True) if name_el
                    else a_tag.get_text(" ", strip=True)
                )

                # 重賞判定: <li> 全体を渡す（a タグ外のグレードアイコンも検査）
                is_grade = _is_grade_race(li)

                if is_grade:
                    seen.add(race_id)
                    found_this_day.append({
                        "race_id":   race_id,
                        "race_name": race_name,
                        "race_date": race_date_str,
                    })
                    logger.info(f"  ★重賞: {race_name} ({race_id})")

            # フォールバック: <li> がない場合は <a href*='race_id='> から全件取得して
            # <li> と同様に親要素を検査する
            if not items:
                logger.info(f"  RaceList_DataItem なし → href ベースにフォールバック ({path})")
                for a in soup.select("a[href*='race_id=']"):
                    m = re.search(r"race_id=(\d{12})", a.get("href", ""))
                    if not m:
                        continue
                    race_id = m.group(1)
                    if race_id in seen:
                        continue

                    # JRA競馬場コード（race_id[4:6]）が01〜10のみ対象（NAR は30以上）
                    if race_id[4:6] not in {"01","02","03","04","05","06","07","08","09","10"}:
                        logger.debug(f"    [fallback] {race_id} スキップ（NAR venue={race_id[4:6]}）")
                        continue

                    # <a> の最も近い block 祖先（<li>/<div>/<tr>）を検査対象にする
                    container = a
                    for anc in a.parents:
                        if anc.name in ("li", "div", "tr", "td"):
                            container = anc
                            break

                    name_el = (
                        container.select_one(".Race_Name")
                        or container.select_one(".RaceName")
                        or container.select_one(".RaceList_ItemTitle")
                    )
                    race_name = (
                        name_el.get_text(strip=True) if name_el
                        else a.get_text(" ", strip=True)
                    )

                    is_grade = _is_grade_race(container)
                    if is_grade:
                        seen.add(race_id)
                        found_this_day.append({
                            "race_id":   race_id,
                            "race_name": race_name,
                            "race_date": race_date_str,
                        })
                        logger.info(f"  ★重賞(fallback): {race_name} ({race_id})")

            # 重賞が見つかれば次のURLは試さない
            if found_this_day:
                break
            if items:
                # アイテムはあったが重賞なし → もう一方のURLも試す
                logger.info(f"  {path}: {len(items)}件あるが重賞0件 → 次URLを試みる")

        found.extend(found_this_day)
        _sleep()

    logger.info(f"重賞合計: {len(found)} レース")
    return found


# 除外キーワード
_FLAT_EXCLUDE = {"未勝利", "新馬", "障害"}


def scrape_flat_race_ids(session: requests.Session) -> list[dict]:
    """今週末の特別戦レース（1勝クラス以上の特別競走）を venue ごとに取得する。

    Returns:
        [{"race_id", "race_name", "race_date", "venue", "is_grade": False}, ...]
    """
    found: list[dict] = []
    seen: set[str] = set()

    dates = _weekend_dates()

    for kaisai_date in dates:
        race_date_str = f"{kaisai_date[:4]}-{kaisai_date[4:6]}-{kaisai_date[6:]}"
        url = f"https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={kaisai_date}"
        soup = _get(url, session)
        if soup is None:
            continue

        for li in soup.select("li.RaceList_DataItem"):
            a_tag = None
            for a in li.select("a[href]"):
                if re.search(r"race_id=\d{12}", a.get("href", "")):
                    a_tag = a
                    break
            if a_tag is None:
                continue
            m = re.search(r"race_id=(\d{12})", a_tag.get("href", ""))
            if not m:
                continue
            race_id = m.group(1)
            if race_id in seen:
                continue

            # JRA のみ
            venue_code = race_id[4:6]
            if venue_code not in {"01","02","03","04","05","06","07","08","09","10"}:
                continue

            # 重賞は除外（別途取得済み）
            if _is_grade_race(li):
                continue

            # レース名・条件を取得
            name_el = (
                li.select_one(".Race_Name") or li.select_one(".RaceName")
                or li.select_one(".RaceList_ItemTitle")
            )
            race_name = name_el.get_text(strip=True) if name_el else a_tag.get_text(" ", strip=True)

            # 除外キーワード
            li_text = li.get_text()
            if any(kw in li_text for kw in _FLAT_EXCLUDE):
                continue

            # 芝・ダート両方対象（障害は除外済み）

            seen.add(race_id)
            venue_name = VENUE_MAP.get(venue_code, "")
            found.append({
                "race_id": race_id,
                "race_name": race_name,
                "race_date": race_date_str,
                "venue": venue_name,
                "is_grade": False,
            })

        _sleep()

    logger.info(f"特別戦（1勝以上）: {len(found)} レース")
    return found


def update_featured_races_csv(
    path: Optional[Path] = None,
    session: Optional[requests.Session] = None,
) -> int:
    """翌週末（土日）の重賞レースを netkeiba からスクレイピングし、
    featured_races.csv（形式: race_id,race_name,grade）に上書き保存する。

    Returns:
        保存したレース数（0 の場合はスクレイピング失敗 or 重賞なし）
    """
    if path is None:
        path = DATA_DIR / "featured_races.csv"
    if session is None:
        session = requests.Session()

    dates = _weekend_dates()
    logger.info(f"[update_featured] 対象日付: {dates[0]} (土) / {dates[1]} (日)")

    LIST_PATHS = ["race_list_sub.html", "race_list.html"]
    found: list[dict] = []
    seen: set[str] = set()

    for kaisai_date in dates:
        found_this_day: list[dict] = []

        for list_path in LIST_PATHS:
            url = f"https://race.netkeiba.com/top/{list_path}?kaisai_date={kaisai_date}"
            logger.info(f"[update_featured] 取得中: {url}")
            soup = _get(url, session)
            if soup is None:
                logger.warning(f"[update_featured] 取得失敗: {url}")
                continue

            items = soup.select("li.RaceList_DataItem")
            logger.info(f"[update_featured] {kaisai_date}: {len(items)} アイテム ({list_path})")

            for li in items:
                a_tag = None
                for a in li.select("a[href]"):
                    if re.search(r"race_id=\d{12}", a.get("href", "")):
                        a_tag = a
                        break
                if a_tag is None:
                    continue
                m = re.search(r"race_id=(\d{12})", a_tag.get("href", ""))
                if not m:
                    continue
                race_id = m.group(1)
                if race_id in seen:
                    continue
                # JRA 競馬場コードのみ（NAR はスキップ）
                if race_id[4:6] not in {"01","02","03","04","05","06","07","08","09","10"}:
                    continue

                name_el = (
                    li.select_one(".Race_Name")
                    or li.select_one(".RaceName")
                    or li.select_one(".RaceList_ItemTitle")
                    or li.select_one(".ItemTitle")
                )
                race_name = (
                    name_el.get_text(strip=True) if name_el
                    else a_tag.get_text(" ", strip=True)
                )

                grade = _detect_grade(li)
                logger.debug(f"[update_featured]   {race_id} [{race_name!r}] grade={grade!r}")

                if grade:
                    seen.add(race_id)
                    found_this_day.append({
                        "race_id":   race_id,
                        "race_name": race_name,
                        "grade":     grade,
                    })
                    logger.info(f"[update_featured] ★ {grade} {race_name} ({race_id})")

            # フォールバック: RaceList_DataItem がない場合
            if not items:
                for a in soup.select("a[href*='race_id=']"):
                    m = re.search(r"race_id=(\d{12})", a.get("href", ""))
                    if not m:
                        continue
                    race_id = m.group(1)
                    if race_id in seen:
                        continue
                    if race_id[4:6] not in {"01","02","03","04","05","06","07","08","09","10"}:
                        continue
                    container = a
                    for anc in a.parents:
                        if anc.name in ("li", "div", "tr", "td"):
                            container = anc
                            break
                    name_el = (
                        container.select_one(".Race_Name")
                        or container.select_one(".RaceName")
                        or container.select_one(".RaceList_ItemTitle")
                    )
                    race_name = (
                        name_el.get_text(strip=True) if name_el
                        else a.get_text(" ", strip=True)
                    )
                    grade = _detect_grade(container)
                    if grade:
                        seen.add(race_id)
                        found_this_day.append({
                            "race_id":   race_id,
                            "race_name": race_name,
                            "grade":     grade,
                        })
                        logger.info(f"[update_featured] ★(fallback) {grade} {race_name} ({race_id})")

            if found_this_day:
                break

        found.extend(found_this_day)
        _sleep()

    if not found:
        logger.warning("[update_featured] 重賞レースが見つかりませんでした。featured_races.csv は更新しません。")
        return 0

    # CSV 保存
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"{r['race_id']},{r['race_name']},{r['grade']}" for r in found]
    path.write_text("race_id,race_name,grade\n" + "\n".join(rows) + "\n", encoding="utf-8-sig")
    logger.info(f"[update_featured] featured_races.csv 保存完了: {len(found)} レース → {path}")
    return len(found)


def _save_upcoming_to_cache() -> None:
    """featured_races.csv のレース情報を predictions_cache.json に upcoming として保存する。

    既存の予想データがあるレースは上書きしない。
    """
    featured_path = DATA_DIR / "featured_races.csv"
    if not featured_path.exists():
        logger.debug("featured_races.csv なし → キャッシュ優先運用のためスキップ")
        return

    try:
        df = pd.read_csv(featured_path, encoding="utf-8-sig", dtype={"race_id": str})
    except Exception as e:
        logger.debug(f"featured_races.csv 読み込み失敗（キャッシュ優先運用）: {e}")
        return

    cache = _load_cache()
    dates = _weekend_dates()

    added = 0
    for _, row in df.iterrows():
        race_id = str(row["race_id"])
        # 既に予想データがあるレースはスキップ
        if race_id in cache and cache[race_id].get("predicted_top3_nums"):
            continue

        race_date_str = ""
        if len(dates) >= 2:
            # race_id から土日を判定（末尾2桁がレース番号、その前が日次）
            race_date_str = f"{dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]}"

        venue_code = race_id[4:6] if len(race_id) >= 6 else ""
        venue = VENUE_MAP.get(venue_code, "")

        cache[race_id] = {
            "race_name":           str(row.get("race_name", race_id)),
            "race_date":           race_date_str,
            "start_time":          "",
            "venue":               venue,
            "course_info":         "",
            "honmei":              None,
            "taikou":              None,
            "ana":                 None,
            "predicted_top3_nums": [],
            "predicted_top5_nums": [],
            "predicted_top5":      [],
            "ev_top3":             [],
            "dangerous_horses":    [],
            "ai_comments":         {},
            "status":              "upcoming",
        }
        added += 1

    _save_cache(cache)
    logger.info(f"upcoming レース {added} 件をキャッシュに保存（既存 {len(cache) - added} 件は維持）")


def _load_featured_race_ids_for_weekend(
    featured_path: Optional[Path] = None,
) -> list[dict]:
    """
    featured_races.csv から今週末（土日）の日付に一致するレースIDを返す。

    scrape_grade_race_ids() のフォールバック用。
    今週末のレースIDを手動で featured_races.csv に登録しておくことで
    スクレイピング失敗時でも予想が動くようになる。

    Returns:
        [{"race_id": str, "race_name": str, "race_date": str}, ...]
    """
    if featured_path is None:
        featured_path = DATA_DIR / "featured_races.csv"
    if not featured_path.exists():
        logger.debug(f"featured_races.csv なし（キャッシュ優先運用）: {featured_path}")
        return []

    try:
        df = pd.read_csv(featured_path, encoding="utf-8-sig", dtype={"race_id": str})
    except Exception as e:
        logger.debug(f"featured_races.csv 読み込み失敗（キャッシュ優先運用）: {e}")
        return []

    if "race_id" not in df.columns:
        return []

    # race_date 列がない新フォーマット（race_id, race_name, grade）の場合は全件返す
    # キャッシュに日付情報があればそちらを優先する
    if "race_date" not in df.columns:
        dates = _weekend_dates()
        sat = f"{dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]}"
        try:
            pred_cache = json.loads(PRED_CACHE.read_text(encoding="utf-8")) if PRED_CACHE.exists() else {}
        except Exception:
            pred_cache = {}
        result = []
        for _, row in df.drop_duplicates(subset=["race_id"]).iterrows():
            rid = str(row["race_id"])
            cached_date = pred_cache.get(rid, {}).get("race_date", "")
            result.append({
                "race_id":   rid,
                "race_name": str(row.get("race_name", row["race_id"])),
                "race_date": cached_date or sat,
            })
        if result:
            logger.info(
                f"[featured fallback] {len(result)} レース "
                f"({', '.join(r['race_name'] for r in result)})"
            )
        return result

    dates = _weekend_dates()  # ["YYYYMMDD", "YYYYMMDD"]
    weekend_dates = {
        f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates
    }

    mask = df["race_date"].astype(str).str[:10].isin(weekend_dates)
    weekend_df = df[mask].drop_duplicates(subset=["race_id"])

    result = []
    for _, row in weekend_df.iterrows():
        result.append({
            "race_id":   str(row["race_id"]),
            "race_name": str(row.get("race_name", row["race_id"])),
            "race_date": str(row["race_date"])[:10],
        })

    if result:
        logger.info(
            f"[featured fallback] {len(result)} レース "
            f"({', '.join(r['race_name'] for r in result)})"
        )
    return result


# ══════════════════════════════════════════════════════════════
# 予想キャッシュ
# ══════════════════════════════════════════════════════════════

def _load_cache() -> dict:
    """予想キャッシュを読み込む。常に predictions_cache.json を使用。
    13時スナップショットは結果照合時のみ _load_cache_for_result() で使用する。"""
    if PRED_CACHE.exists():
        try:
            with open(PRED_CACHE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"キャッシュの読み込みに失敗: {e}")
    return {}


def _load_cache_for_result() -> dict:
    """結果照合用: 実キャッシュをベースに、13時スナップショットの予想データを補完（後出し防止）。
    result_notified 等のフラグは実キャッシュから取得し、予想内容はスナップショットを優先する。"""
    real = _load_cache()
    snapshot_13 = DATA_DIR / "predictions_snapshot_13.json"
    if not snapshot_13.exists():
        return real
    try:
        with open(snapshot_13, encoding="utf-8") as f:
            snap = json.load(f)
        snap.pop("_snapshot_time", None)
        logger.info(f"13時スナップショットを補完使用: {snapshot_13.name}")
        # スナップショットの予想データを使いつつ、実キャッシュのフラグを保持
        for rid, entry in snap.items():
            if rid.startswith("_"):
                continue
            if rid not in real:
                real[rid] = entry
            else:
                # 予想内容はスナップショット優先、フラグ（result_notified等）は実キャッシュ保持
                flags = {k: real[rid][k] for k in ("result_notified", "notified_predict", "result_settled", "wide_hit")
                         if k in real[rid]}
                real[rid].update({k: v for k, v in entry.items() if k not in flags})
                real[rid].update(flags)
        return real
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"スナップショットの読み込みに失敗: {e}")
        return real


def _save_cache(cache: dict) -> None:
    PRED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(PRED_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    # ── 書き込み確認ログ ──────────────────────────────────────
    size = PRED_CACHE.stat().st_size
    keys = list(cache.keys())
    print(f"[_save_cache] 書き込み完了: {PRED_CACHE.resolve()} ({size}bytes, {len(keys)}件: {keys})", flush=True)


def _ana_horse_info(result_df: pd.DataFrame, ana_horse_num: "Optional[int]") -> dict:
    """穴馬の詳細情報を返す（キャッシュ保存用）。"""
    if ana_horse_num is None or result_df.empty:
        return {}
    match = result_df[pd.to_numeric(result_df["horse_number"], errors="coerce") == ana_horse_num]
    if match.empty:
        return {}
    r = match.iloc[0]
    return {
        "horse_number": ana_horse_num,
        "horse_name": str(r.get("horse_name", "")),
        "prob": round(float(r.get("prob_top3", 0)), 4),
        "popularity": int(pd.to_numeric(r.get("popularity"), errors="coerce") or 0),
    }


def _load_cache_direct() -> dict:
    """predictions_cache.json を直接読み込む（スナップショット無視）。
    予想生成時に使用。スナップショット優先だと古いキャッシュで上書きされるため。"""
    if PRED_CACHE.exists():
        try:
            with open(PRED_CACHE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _store_prediction(race_id: str, race_name: str, race_date: str,
                      result_df: pd.DataFrame,
                      course_info: str = "",
                      start_time: str = "",
                      venue: str = "",
                      is_grade: bool = True) -> None:
    """予想結果をキャッシュに保存する（Discord通知・結果照合に使用）。"""
    cache = _load_cache_direct()  # スナップショットではなく実ファイルを読む

    def _horse(df: pd.DataFrame, idx: int) -> dict:
        if len(df) <= idx:
            return {}
        r = df.iloc[idx]
        return {
            "horse_number": int(r["horse_number"]) if pd.notna(r.get("horse_number")) else None,
            "horse_name":   str(r.get("horse_name", "")),
            "prob":         round(float(r["prob_top3"]), 4),
        }

    top3_nums = []
    for _, row in result_df.head(3).iterrows():
        v = row.get("horse_number")
        if pd.notna(v):
            top3_nums.append(int(v))

    top5_nums = []
    for _, row in result_df.head(6).iterrows():
        v = row.get("horse_number")
        if pd.notna(v):
            top5_nums.append(int(v))

    # EV
    if "ev_score" not in result_df.columns:
        result_df = calc_ev_and_flags(result_df)

    ev_top3: list[dict] = []
    for _, r in result_df[result_df["ev_score"].notna()].nlargest(3, "ev_score").iterrows():
        ev_top3.append({
            "horse_number": int(r["horse_number"]) if pd.notna(r.get("horse_number")) else None,
            "horse_name":   str(r.get("horse_name", "")),
            "ev_score":     round(float(r["ev_score"]), 3),
            "prob":         round(float(r["prob_top3"]), 4),
        })

    # 穴馬: TOP5外 & 8番人気以下 & EV>0 → EV最高の1頭
    ana_horse_num: Optional[int] = None
    top5_set = set(top5_nums[:5])
    if len(result_df) > 5:
        rest = result_df.iloc[5:]
        rest_pop = pd.to_numeric(rest.get("popularity", pd.Series(dtype=float)), errors="coerce")
        rest_ev = pd.to_numeric(rest.get("ev_score", pd.Series(dtype=float)), errors="coerce")
        cands = rest[rest_pop.notna() & (rest_pop >= 8) & (rest_ev > 0)]
        if not cands.empty:
            best = cands.nlargest(1, "ev_score").iloc[0]
            v = best.get("horse_number")
            if pd.notna(v) and int(v) not in top5_set:
                ana_horse_num = int(v)

    cache[race_id] = {
        "race_name":           race_name,
        "race_date":           race_date,
        "start_time":          start_time,
        "venue":               venue,
        "course_info":         course_info,
        "honmei":              _horse(result_df, 0),
        "taikou":              _horse(result_df, 1),
        "ana":                 _horse(result_df, 2),
        "predicted_top3_nums": top3_nums,
        "predicted_top5_nums": top5_nums,
        "ev_top3":             ev_top3,
        "ana_horse_num":       ana_horse_num,
        "is_grade":            is_grade,
    }

    # 自信度計算（買い目決定の前に実行）
    confidence_score = 0
    try:
        from keiba_predictor.model.predict import _calc_confidence
        confidence_score, stars = _calc_confidence(cache[race_id])
        cache[race_id]["confidence"] = confidence_score
        cache[race_id]["confidence_stars"] = stars
    except Exception as e:
        logger.warning(f"自信度計算失敗: {e}")

    # 買い目自動決定
    try:
        from keiba_predictor.model.predict import _decide_bet_strategy
        _is_vol = cache[race_id].get("simulation", {}).get("is_volatile_race", False)
        _ana = cache[race_id].get("ana_horse_num")
        _rname = cache[race_id].get("race_name", "")
        cache[race_id]["bet_strategy"] = _decide_bet_strategy(
            result_df, is_volatile_race=_is_vol, confidence=confidence_score,
            ana_horse_num=_ana, race_id=race_id, race_name=_rname)
    except Exception as e:
        logger.warning(f"買い目自動決定失敗: {e}")


    # モンテカルロシミュレーション
    try:
        from keiba_predictor.simulation import run_monte_carlo
        mc_horses = []
        for i in range(min(len(result_df), 18)):
            r = result_df.iloc[i]
            mc_horses.append({
                "horse_number": int(r["horse_number"]) if pd.notna(r.get("horse_number")) else i + 1,
                "horse_name": str(r.get("horse_name", "")),
                "prob": float(r["prob_top3"]),
                "running_style_enc": int(r.get("running_style_enc", 2)) if pd.notna(r.get("running_style_enc")) else 2,
            })
        mc_result = run_monte_carlo(mc_horses)
        # 上位5頭 + 波乱度を保存
        sim = {}
        for num in top5_nums[:5]:
            k = str(num)
            if k in mc_result:
                sim[k] = mc_result[k]
        sim["race_volatility"] = mc_result.get("race_volatility", 0)
        sim["is_volatile_race"] = mc_result.get("is_volatile_race", False)
        cache[race_id]["simulation"] = sim
    except Exception as e:
        logger.warning(f"モンテカルロシミュレーション失敗: {e}")

    _save_cache(cache)


# ══════════════════════════════════════════════════════════════
# 払戻金取得
# ══════════════════════════════════════════════════════════════

def scrape_payouts(race_id: str, session: requests.Session) -> dict:
    """レース払戻金を取得する。

    Returns:
        {"馬連": [{"combo": "3-5", "amount": 1450}], "ワイド": [...], ...}
    """
    # まず race.netkeiba.com（静的HTML・EUC-JP）を試す
    from keiba_predictor.scraper.netkeiba_scraper import RACE_RESULT_SITE_URL
    alt_url = f"{RACE_RESULT_SITE_URL}?race_id={race_id}"
    soup = _get(alt_url, session, encoding="euc-jp")
    # フォールバック: db.netkeiba.com
    if soup is None:
        url = RACE_RESULT_URL.format(race_id=race_id)
        soup = _get(url, session)
    if soup is None:
        return {}

    payouts: dict[str, list] = {}

    def _parse_yen(s: str) -> Optional[int]:
        s = re.sub(r"[¥￥,円\s]", "", s)
        try:
            return int(s)
        except ValueError:
            return None

    for table in soup.select("table.pay_table_01, table.Payout_Detail_Table"):
        current_type = None
        for tr in table.select("tr"):
            th = tr.select_one("th")
            tds = tr.select("td")
            if th:
                current_type = th.get_text(strip=True)
            if not current_type or len(tds) < 2:
                continue

            # brタグを改行に変換して分割（JRA: <br>区切り / NAR: span/div区切り）
            combo_parts = [p.strip() for p in tds[0].get_text("\n").split("\n") if p.strip()]
            amt_parts   = [p.strip() for p in tds[1].get_text("\n").split("\n") if p.strip()]
            amt_list = [_parse_yen(a) for a in amt_parts]

            n_amt = len(amt_list)
            n_combo = len(combo_parts)

            if n_combo == n_amt and n_amt > 0:
                # 1対1マッチ（複勝: 3馬番 vs 3金額）
                for combo, amt in zip(combo_parts, amt_list):
                    payouts.setdefault(current_type, []).append({
                        "combo": combo, "amount": amt,
                    })
            elif n_amt > 0 and n_combo > n_amt and n_combo % n_amt == 0:
                # combo を n_amt 個のグループに等分割（ワイド: 6馬番 → 3組×2馬番）
                group_size = n_combo // n_amt
                for i, amt in enumerate(amt_list):
                    group = combo_parts[i * group_size:(i + 1) * group_size]
                    combo_str = "-".join(group)
                    payouts.setdefault(current_type, []).append({
                        "combo": combo_str, "amount": amt,
                    })
            elif n_amt > 0:
                combo_all = "-".join(combo_parts)
                for amt in amt_list:
                    payouts.setdefault(current_type, []).append({
                        "combo": combo_all, "amount": amt,
                    })

    return payouts




def _record_manual_result(race_id: str, race_name: str, race_date: str,
                          pred: dict, manual: dict) -> None:
    """manual_results.json の的中フラグで results_history.csv に記録する。"""
    from keiba_predictor.history import HISTORY_PATH, DATA_DIR, _grade_label, _pred_row

    # レース名: manual 優先 → 引数 → race_id
    name = manual.get("race_name") or race_name or race_id
    grade = _grade_label(name)
    p1 = _pred_row(pred, "honmei")
    p2 = _pred_row(pred, "taikou")
    p3 = _pred_row(pred, "ana")

    result_nums = manual.get("result", [])
    manual_pay = manual.get("payouts", {})

    fukusho_hit = manual.get("fukusho_hit", False)
    umaren_hit  = manual.get("umaren_hit", False)
    sanren_hit  = manual.get("sanrenpuku_hit", False)

    fukusho_payout  = manual_pay.get("fukusho", 0)
    umaren_payout   = manual_pay.get("umaren", 0)
    sanren_payout   = manual_pay.get("sanrenpuku", 0)
    # 投資: 複勝1点 + 馬連3点 + 3連複10点 = 14点 × 100円
    bet_total       = 1400
    return_total    = fukusho_payout + umaren_payout + sanren_payout

    def _a(i):
        return {"name": "", "num": result_nums[i] if i < len(result_nums) else 0}

    row = {
        "date":       race_date,
        "race_id":    race_id,
        "race_name":  name,
        "race_grade": grade,
        "pred1_name": p1["name"], "pred1_num": p1["num"], "pred1_prob": p1["prob"],
        "pred2_name": p2["name"], "pred2_num": p2["num"], "pred2_prob": p2["prob"],
        "pred3_name": p3["name"], "pred3_num": p3["num"], "pred3_prob": p3["prob"],
        "actual1_name": _a(0)["name"], "actual1_num": _a(0)["num"],
        "actual2_name": _a(1)["name"], "actual2_num": _a(1)["num"],
        "actual3_name": _a(2)["name"], "actual3_num": _a(2)["num"],
        "fukusho_hit":     fukusho_hit,
        "umaren_hit":      umaren_hit,     "umaren_payout":   umaren_payout,
        "wide_hit":        False,          "wide_payout":     0,
        "sanrenpuku_hit":  sanren_hit,     "sanrenpuku_payout": sanren_payout,
        "bet_total":       bet_total,
        "return_total":    return_total,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new_row_df = pd.DataFrame([row])
    if HISTORY_PATH.exists():
        new_row_df.to_csv(HISTORY_PATH, mode="a", header=False,
                          index=False, encoding="utf-8-sig")
    else:
        new_row_df.to_csv(HISTORY_PATH, mode="w", header=True,
                          index=False, encoding="utf-8-sig")
    logger.info(f"  [history] 手動記録: {name} fukusho={fukusho_hit} umaren={umaren_hit} sanren={sanren_hit} return=¥{return_total:,}")


def _fmt_result(race_name: str, race_date: str,
                actual_df: pd.DataFrame,
                pred: dict,
                payouts: dict,
                manual: Optional[dict] = None,
                race_id: str = "",
                is_grade: bool = False) -> str:
    """結果メッセージを生成する。is_grade=True で特別戦/重賞の詳細版、False で簡易版。"""
    RULE = "─" * 16
    venue = pred.get("venue", "")
    race_num = ""
    if race_id and len(race_id) >= 12:
        try:
            race_num = f"{int(race_id[10:12])}R "
        except ValueError:
            pass
    if not venue and race_id and len(race_id) >= 10:
        venue = VENUE_MAP.get(race_id[8:10], "")
    header = f"{venue} {race_num}{race_name}".strip()
    lines = [f"🏆 【KEIBA EDGE】結果", f"📅 {race_date}　{header}", RULE]

    # 予想馬番→印 のマッピング
    pred_num_to_mark: dict[int, str] = {}
    for role, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "△")]:
        p = pred.get(role, {})
        num = p.get("horse_number")
        if num is not None:
            pred_num_to_mark[int(num)] = mark

    # manual_results.json の predicted_top3_nums があれば優先
    predicted_nums = pred.get("predicted_top3_nums", [])
    if manual and manual.get("predicted_top3_nums"):
        predicted_nums = manual["predicted_top3_nums"]

    # 馬番→馬名マップを予想キャッシュから構築（結果に馬名がない場合の補完用）
    num_to_name: dict[int, str] = {}
    for role in ("honmei", "taikou", "ana"):
        p = pred.get(role, {})
        pnum = p.get("horse_number")
        if pnum is not None:
            num_to_name[int(pnum)] = p.get("horse_name", "")
    for h in (pred.get("predicted_top5") or []):
        hnum = h.get("horse_number")
        if hnum is not None and int(hnum) not in num_to_name:
            num_to_name[int(hnum)] = h.get("horse_name", "")
    for e in (pred.get("ev_top3") or []):
        enum = e.get("horse_number")
        if enum is not None and int(enum) not in num_to_name:
            num_to_name[int(enum)] = e.get("horse_name", "")
    # actual_df からも馬名を補完
    for _, r in actual_df.iterrows():
        anum = r.get("horse_number")
        aname = str(r.get("horse_name", ""))
        if pd.notna(anum) and aname and int(anum) not in num_to_name:
            num_to_name[int(anum)] = aname

    # 確定 1〜3 着
    df_copy = actual_df.copy()
    df_copy["_fp"] = pd.to_numeric(df_copy["finish_position"], errors="coerce")
    top3 = df_copy[df_copy["_fp"].isin([1, 2, 3])].sort_values("_fp").head(3)

    actual_top3_nums: list[int] = []
    for _, r in top3.iterrows():
        fp   = int(r["_fp"])
        num  = int(r["horse_number"]) if pd.notna(r.get("horse_number")) else 0
        name = str(r.get("horse_name", ""))
        if not name:
            name = num_to_name.get(num, "")
        actual_top3_nums.append(num)
        mark = pred_num_to_mark.get(num, "　")
        icon = " ✅" if num in predicted_nums else ""
        lines.append(f"{fp}着 {mark} {num}番 {name}{icon}")

    lines.append(RULE)

    # 特別戦/重賞以外はあっさり版（着順 + 的中/不的中 のみ）
    if not is_grade and not manual:
        hits = check_hits_from_bet_strategy(pred, actual_top3_nums, payouts)
        any_hit = hits["fukusho_hit"] or hits["wide_hit"] or hits["sanren_hit"]
        parts = []
        if hits["fukusho_hit"]:
            parts.append("複勝✅")
        if hits["wide_hit"]:
            pay = hits["wide_pay"]
            parts.append(f"ワイド✅{f'({pay})' if pay else ''}")
        if hits["sanren_hit"]:
            pay = hits["sanren_pay"]
            parts.append(f"3連複✅{f'({pay})' if pay else ''}")
        if not any_hit:
            parts.append("❌")
        lines.append(" ".join(parts))
        return "\n".join(lines)

    # manual_results.json のフラグがあればそちらを優先
    if manual and "fukusho_hit" in manual:
        honmei_num = int(manual["honmei"]) if manual.get("honmei") is not None else (predicted_nums[0] if predicted_nums else None)
        honmei_name = num_to_name.get(honmei_num, "") if honmei_num else ""
        fukusho_hit = manual["fukusho_hit"]
        umaren_hit  = manual.get("umaren_hit", False)
        sanren_hit  = manual.get("sanrenpuku_hit", False)
        manual_pay  = manual.get("payouts", {})
        umaren_pay  = f"¥{manual_pay['umaren']:,}" if manual_pay.get("umaren") else ""
        sanren_pay  = f"¥{manual_pay['sanrenpuku']:,}" if manual_pay.get("sanrenpuku") else ""
        tansho_hit = False
        tansho_pay = ""
        use_wide = False
        wide_hit = False
        wide_pay = ""
        sanrentan_hit = False
        sanrentan_pay = ""
    else:
        # bet_strategy に基づく自動判定
        hits = check_hits_from_bet_strategy(pred, actual_top3_nums, payouts)
        honmei_num = hits["fukusho_num"]
        honmei_name = num_to_name.get(honmei_num, "") if honmei_num else ""
        tansho_hit = hits["tansho_hit"]
        tansho_pay = hits["tansho_pay"]
        fukusho_hit = hits["fukusho_hit"]
        umaren_hit = hits["umaren_hit"]
        umaren_pay = hits["umaren_pay"]
        sanren_hit = hits["sanren_hit"]
        sanren_pay = hits["sanren_pay"]
        sanrentan_hit = hits["sanrentan_hit"]
        sanrentan_pay = hits["sanrentan_pay"]
        use_wide = hits["use_wide"]
        wide_hit = hits["wide_hit"]
        wide_pay = hits["wide_pay"]

    # 単勝（買っていた場合のみ表示）
    if hits.get("tansho_num") if not manual else False:
        tansho_line = f"単勝 {'✅' if tansho_hit else '❌'}"
        if tansho_hit and tansho_pay:
            tansho_line += f"（{re.sub(r'[¥,]', '', str(tansho_pay))}円）"
        lines.append(tansho_line)

    # 複勝配当を取得
    fukusho_pay_str = ""
    if fukusho_hit and honmei_num:
        if manual and manual.get("payouts", {}).get("fukusho"):
            fukusho_pay_str = f"{manual['payouts']['fukusho']:,}"
        else:
            for _entry in payouts.get("複勝", []):
                _cnums = set(re.findall(r"\d+", str(_entry.get("combo", ""))))
                if str(honmei_num) in _cnums:
                    _amt = _entry.get("amount")
                    if _amt:
                        fukusho_pay_str = f"{_amt:,}" if isinstance(_amt, int) else re.sub(r"[¥￥,円\s]", "", str(_amt))
                    break
    if fukusho_hit and fukusho_pay_str:
        lines.append(f"複勝 ✅（◎{honmei_num}番 {honmei_name} 配当{fukusho_pay_str}円）")
    else:
        lines.append(f"複勝 {'✅' if fukusho_hit else '❌'}（◎{honmei_num}番 {honmei_name}）")

    if use_wide:
        wide_line = f"ワイド {'✅' if wide_hit else '❌'}"
        if wide_hit and wide_pay:
            wide_line += f"（{_payout_str_to_int(wide_pay) if isinstance(wide_pay, str) else wide_pay}円）"
        lines.append(wide_line)
    else:
        umaren_line = f"馬連 {'✅' if umaren_hit else '❌'}"
        if umaren_hit and umaren_pay:
            umaren_line += f"（{re.sub(r'[¥,]', '', str(umaren_pay))}円）"
        lines.append(umaren_line)

    sanren_line = f"3連複 {'✅' if sanren_hit else '❌'}"
    if sanren_hit and sanren_pay:
        sanren_line += f"（{re.sub(r'[¥,]', '', str(sanren_pay))}円）"
    lines.append(sanren_line)

    # 3連単（買っていた場合のみ表示）
    if hits.get("sanrentan_pay") is not None if not manual else False:
        bs = pred.get("bet_strategy", {})
        if bs.get("sanrentan"):
            sanrentan_line = f"3連単 {'✅' if sanrentan_hit else '❌'}"
            if sanrentan_hit and sanrentan_pay:
                sanrentan_line += f"（{re.sub(r'[¥,]', '', str(sanrentan_pay))}円）"
            lines.append(sanrentan_line)

    return "\n".join(lines)


def _get_payout(payouts: dict, bet_type: str, combo: str) -> str:
    """払戻金辞書から指定の組み合わせ・金額を文字列で返す。"""
    _ALIASES = {
        "三連複": ["三連複", "3連複"],
        "三連単": ["三連単", "3連単"],
        "馬連": ["馬連"], "馬単": ["馬単"],
        "ワイド": ["ワイド"], "単勝": ["単勝"], "複勝": ["複勝"],
    }
    keys = _ALIASES.get(bet_type, [bet_type])
    for key in keys:
        for entry in payouts.get(key, []):
            e_nums = set(re.findall(r"\d+", entry["combo"]))
            c_nums = set(re.findall(r"\d+", combo))
            if e_nums == c_nums:
                amt = entry["amount"]
                return f"¥{amt:,}" if amt else ""
    return ""


# ── 買い目判定（module-level: history.py からも呼び出せる） ────────────

def _check_umaren_raw(
    predicted_nums: list[int],
    actual_top3_nums: list[int],
    payouts: dict,
) -> tuple[bool, str]:
    """馬連的中判定。買い目はtop3の全組み合わせ(3点)。(hit, pay_str) を返す。"""
    if len(predicted_nums) < 2 or len(actual_top3_nums) < 2:
        return False, ""
    a1, a2 = actual_top3_nums[0], actual_top3_nums[1]
    actual_set = {a1, a2}
    # 買い目: predicted_top3_nums[:3] の全組み合わせ
    for pair in combinations(predicted_nums[:3], 2):
        if set(pair) == actual_set:
            combo = f"{pair[0]}-{pair[1]}"
            pay = _get_payout(payouts, "馬連", combo)
            return True, pay
    return False, ""


def _check_wide_pairs_raw(
    predicted_nums: list[int],
    actual_top3_nums: list[int],
    payouts: dict,
) -> list[tuple[str, bool, str]]:
    """ワイド全組み合わせ判定。[(combo, hit, pay_str), ...] を返す。"""
    results = []
    if len(predicted_nums) < 2 or len(actual_top3_nums) < 3:
        return results
    for a, b in combinations(predicted_nums[:3], 2):
        hit   = a in actual_top3_nums and b in actual_top3_nums
        combo = f"{a}-{b}"
        pay   = _get_payout(payouts, "ワイド", combo)
        results.append((combo, hit, pay))
    return results


def _check_sanrenpuku_raw(
    predicted_nums: list[int],
    actual_top3_nums: list[int],
    payouts: dict,
    ana_horse_num: Optional[int] = None,
    pred: Optional[dict] = None,
) -> tuple[bool, str]:
    """3連複的中判定。(hit, pay_str) を返す。

    predicted_top5_nums の全馬を相手候補として、
    軸(◎) × 相手2頭 の組合せが実際の3着以内と一致するか判定。
    """
    if len(actual_top3_nums) < 3:
        return False, ""

    # 軸 = honmei
    axis = None
    if pred:
        axis = (pred.get("honmei") or {}).get("horse_number")
    if axis is None and predicted_nums:
        axis = predicted_nums[0]
    if axis is None:
        return False, ""

    if int(axis) not in actual_top3_nums:
        return False, ""

    # 相手候補: predicted_top5_nums を優先
    partners = []
    if pred:
        top5 = pred.get("predicted_top5_nums", [])
        if top5:
            partners = [n for n in top5 if n != axis]
    if not partners:
        partners = list(predicted_nums[1:5]) if len(predicted_nums) > 1 else []
    if ana_horse_num and ana_horse_num not in partners:
        partners.append(ana_horse_num)

    actual_set = set(actual_top3_nums[:3])
    for pair in combinations(partners, 2):
        if {int(axis), pair[0], pair[1]} == actual_set:
            combo = "-".join(str(n) for n in sorted([int(axis), pair[0], pair[1]]))
            pay = _get_payout(payouts, "三連複", combo)
            return True, pay
    return False, ""


def _format_simple_message(race_id: str, entry: dict) -> str:
    """予想キャッシュから Discord通知メッセージを生成する。"""
    venue = entry.get("venue", "")
    if not venue and race_id and len(race_id) >= 10:
        venue = VENUE_MAP.get(race_id[4:6], "")
    race_num = ""
    if race_id and len(race_id) >= 12:
        try:
            race_num = f"{int(race_id[10:12])}R"
        except ValueError:
            pass
    race_name = entry.get("race_name", "")
    course_info = entry.get("course_info", "")

    honmei = entry.get("honmei", {})
    taikou = entry.get("taikou", {})
    ana = entry.get("ana", {})

    # EVマップ
    ev_map: dict[int, float] = {}
    for e in entry.get("ev_top3", []):
        num = e.get("horse_number")
        if num is not None:
            ev_map[int(num)] = e.get("ev_score", 0)

    # MC確率マップ
    sim = entry.get("simulation", {})

    def _horse_line(mark: str, h: dict) -> str:
        num = h.get("horse_number", "?")
        name = h.get("horse_name", "")
        # MC確率を優先
        mc = sim.get(str(num), {})
        mc_rate = mc.get("top3_rate")
        prob = (mc_rate * 100) if mc_rate is not None else (h.get("prob", 0) * 100)
        ev = ev_map.get(int(num) if num != "?" else 0, 0)
        ev_str = f" EV{ev:.2f}" if ev else ""
        return f"{mark} {num}番 {name}　{prob:.1f}%{ev_str}"

    flag_sep = "🏁━━━━━━━━━━━━━━━━━━🏁"
    title = f"　　{venue} {race_num} {race_name}".rstrip()
    meta_parts = []
    if venue:
        meta_parts.append(f"📍{venue}")
    start_time = entry.get("start_time", "")
    if start_time:
        meta_parts.append(f"🕐{start_time}発走")
    if course_info:
        meta_parts.append(course_info)
    meta_line = f"　　{' | '.join(meta_parts)}" if meta_parts else ""
    lines = [flag_sep, title]
    if meta_line:
        lines.append(meta_line)
    lines.append(flag_sep)
    lines.append("")
    lines.append(_horse_line("◎", honmei))
    lines.append(_horse_line("○", taikou))
    lines.append(_horse_line("△", ana))

    # 波乱度表示
    if sim.get("is_volatile_race"):
        lines.append(f"🌀 波乱注意（波乱度{sim.get('race_volatility', 0):.2f}）")

    # 買い目（bet_strategy ベース）
    bs = entry.get("bet_strategy", {})
    if bs and bs.get("total_points", 0) > 0:
        lines.append("")
        lines.append("💰 買い目")
        if bs.get("fukusho"):
            f = bs["fukusho"][0]
            lines.append(f"複勝 {f['num']}番 {f.get('name', '')}  1,000円")
        if bs.get("use_wide") and bs.get("wide"):
            wide_str = " / ".join(f"{w['nums'][0]}-{w['nums'][1]}" for w in bs["wide"])
            lines.append(f"ワイド {wide_str}  各300円")
        if bs.get("umaren"):
            umaren_str = " / ".join(f"{u['nums'][0]}-{u['nums'][1]}" for u in bs["umaren"])
            lines.append(f"馬連 {umaren_str}  各100円")
        sr = bs.get("sanrenpuku", {})
        if sr and sr.get("jiku") and sr.get("aite"):
            from itertools import combinations
            sr_pt = len(list(combinations(sr["aite"], 2)))
            lines.append(f"3連複 軸{sr['jiku'][0]} × {'/'.join(str(n) for n in sr['aite'])}  各100円")
        total_cost = bs.get("total_cost", bs["total_points"] * 100)
        lines.append(f"────────────────")
        lines.append(f"合計投資額: {total_cost:,}円")

    return "\n".join(lines)


def check_hits_from_bet_strategy(
    pred: dict,
    actual_top3_nums: list[int],
    payouts: dict,
) -> dict:
    """bet_strategy の実際の買い目に基づいて的中判定を行��。

    bet_strategy がなければ従来のフォールバックロジックを使用する。

    Returns:
        {
            "fukusho_hit": bool,
            "fukusho_num": int,
            "umaren_hit": bool, "umaren_pay": str,
            "wide_hit": bool, "wide_pay": int,
            "sanren_hit": bool, "sanren_pay": str,
            "use_wide": bool,
            "bet_total": int,
        }
    """
    bs = pred.get("bet_strategy")
    predicted_nums = pred.get("predicted_top3_nums", [])
    ana_horse_num = pred.get("ana_horse_num")

    if not bs or not bs.get("total_points"):
        # フォールバック: 従来ロジック
        honmei_num = predicted_nums[0] if predicted_nums else None
        fukusho_hit = honmei_num is not None and int(honmei_num) in actual_top3_nums
        umaren_hit, umaren_pay = _check_umaren_raw(predicted_nums, actual_top3_nums, payouts)
        sanren_hit, sanren_pay = _check_sanrenpuku_raw(
            predicted_nums, actual_top3_nums, payouts, ana_horse_num, pred=pred)
        wide_pairs = _check_wide_pairs_raw(predicted_nums, actual_top3_nums, payouts)
        wide_hit = any(h for _, h, _ in wide_pairs)
        wide_pay_total = sum(_payout_str_to_int(p) for _, h, p in wide_pairs if h)
        return {
            "tansho_hit": False, "tansho_num": None, "tansho_pay": "",
            "fukusho_hit": fukusho_hit, "fukusho_num": honmei_num,
            "umaren_hit": umaren_hit, "umaren_pay": umaren_pay,
            "wide_hit": wide_hit, "wide_pay": wide_pay_total,
            "sanren_hit": sanren_hit, "sanren_pay": sanren_pay,
            "sanrentan_hit": False, "sanrentan_pay": "",
            "use_wide": False, "bet_total": 2300,
        }

    # ── bet_strategy に基づく判定 ──

    # 単勝
    tansho_hit = False
    tansho_pay = ""
    tansho_list = bs.get("tansho", [])
    tansho_num = tansho_list[0]["num"] if tansho_list else None
    if tansho_num and len(actual_top3_nums) >= 1 and int(tansho_num) == actual_top3_nums[0]:
        tansho_hit = True
        tansho_pay = _get_payout(payouts, "単勝", str(tansho_num))

    # 複勝
    fukusho_list = bs.get("fukusho", [])
    fukusho_num = fukusho_list[0]["num"] if fukusho_list else (predicted_nums[0] if predicted_nums else None)
    fukusho_hit = fukusho_num is not None and int(fukusho_num) in actual_top3_nums

    # 馬連
    umaren_hit = False
    umaren_pay = ""
    if bs.get("umaren") and not bs.get("use_wide"):
        actual_12 = set(actual_top3_nums[:2]) if len(actual_top3_nums) >= 2 else set()
        for u in bs["umaren"]:
            if set(u["nums"]) == actual_12:
                combo = f"{u['nums'][0]}-{u['nums'][1]}"
                umaren_pay = _get_payout(payouts, "馬連", combo)
                umaren_hit = True
                break

    # ワイド
    wide_hit = False
    wide_pay_total = 0
    use_wide = bs.get("use_wide", False)
    if use_wide and bs.get("wide"):
        actual_top3_set = set(actual_top3_nums[:3]) if len(actual_top3_nums) >= 3 else set()
        for w in bs["wide"]:
            a, b = w["nums"]
            if a in actual_top3_set and b in actual_top3_set:
                combo = f"{a}-{b}"
                pay_str = _get_payout(payouts, "ワイド", combo)
                wide_pay_total += _payout_str_to_int(pay_str)
                wide_hit = True

    # 3連複
    sanren_hit = False
    sanren_pay = ""
    sr = bs.get("sanrenpuku", {})
    if sr and sr.get("jiku") and sr.get("aite"):
        jiku = sr["jiku"]
        aite = sr["aite"]
        actual_set = set(actual_top3_nums[:3]) if len(actual_top3_nums) >= 3 else set()
        if len(jiku) == 1:
            axis = jiku[0]
            if axis in actual_set:
                for pair in combinations(aite, 2):
                    if {axis, pair[0], pair[1]} == actual_set:
                        combo = "-".join(str(n) for n in sorted([axis, pair[0], pair[1]]))
                        sanren_pay = _get_payout(payouts, "三連複", combo)
                        sanren_hit = True
                        break
        elif len(jiku) == 2:
            if jiku[0] in actual_set and jiku[1] in actual_set:
                for a in aite:
                    if {jiku[0], jiku[1], a} == actual_set:
                        combo = "-".join(str(n) for n in sorted([jiku[0], jiku[1], a]))
                        sanren_pay = _get_payout(payouts, "三連複", combo)
                        sanren_hit = True
                        break

    # 3連単
    sanrentan_hit = False
    sanrentan_pay = ""
    st = bs.get("sanrentan", {})
    if st and st.get("first") and st.get("aite") and len(actual_top3_nums) >= 3:
        first = st["first"]
        st_aite = st["aite"]
        a1, a2, a3 = actual_top3_nums[0], actual_top3_nums[1], actual_top3_nums[2]
        # 1着固定: first が1着 & 2着・3着が aite に含まれる
        if a1 == first and a2 in st_aite and a3 in st_aite:
            combo = f"{a1}-{a2}-{a3}"
            sanrentan_pay = _get_payout(payouts, "三連単", combo)
            sanrentan_hit = True

    return {
        "tansho_hit": tansho_hit, "tansho_num": tansho_num, "tansho_pay": tansho_pay,
        "fukusho_hit": fukusho_hit, "fukusho_num": fukusho_num,
        "umaren_hit": umaren_hit, "umaren_pay": umaren_pay,
        "wide_hit": wide_hit, "wide_pay": wide_pay_total,
        "sanren_hit": sanren_hit, "sanren_pay": sanren_pay,
        "sanrentan_hit": sanrentan_hit, "sanrentan_pay": sanrentan_pay,
        "use_wide": use_wide, "bet_total": bs.get("total_cost", bs.get("total_points", 0) * 100),
    }


def _payout_str_to_int(s) -> int:
    """'¥1,234' のような文字列を int に変換する。"""
    if not s:
        return 0
    return int(re.sub(r"[¥,]", "", str(s))) if re.search(r"\d", str(s)) else 0


def _format_prediction_from_cache(race_name: str, entry: dict, race_id: str = "") -> tuple[str, str]:
    """predictions_cache.json のエントリからDiscord用メッセージ(予想・買い目)を生成する。"""
    sep = "─" * 16
    course_info = entry.get("course_info", "")
    ai_comments = entry.get("ai_comments", {})

    # ── Message 1: 予想 ───────────────────────────────────────
    venue = entry.get("venue", "")
    race_num = ""
    if race_id and len(race_id) >= 12:
        try:
            race_num = f"{int(race_id[10:12])}R "
        except ValueError:
            pass
    # JRA: race_id[8:10] が競馬場コード
    if not venue and race_id and len(race_id) >= 10:
        venue = VENUE_MAP.get(race_id[8:10], "")
    conf_stars = entry.get("confidence_stars", "")
    header = f"{venue} {race_num}{race_name}".strip()
    lines1 = [sep, f"🏇 {header}"]
    if course_info:
        lines1.append(course_info)
    if conf_stars:
        lines1.append(f"🎯 自信度: {conf_stars}")
    lines1.append(sep)

    MARKS = ["◎", "○", "▲", "△", " "]
    top5_nums = entry.get("predicted_top5_nums", [])

    # predicted_top5（上位5頭の詳細情報）を馬番→infoのマップに変換
    top5_detail: dict[int, dict] = {}
    for h in (entry.get("predicted_top5") or []):
        num = h.get("horse_number")
        if num is not None:
            top5_detail[int(num)] = h

    # honmei/taikou/ana からも補完
    for role in ("honmei", "taikou", "ana"):
        p = entry.get(role, {})
        num = p.get("horse_number")
        if num is not None and int(num) not in top5_detail:
            top5_detail[int(num)] = p

    ev_map: dict[int, dict] = {}
    for e in (entry.get("ev_top3") or []):
        num = e.get("horse_number")
        if num is not None:
            ev_map[int(num)] = e

    # モンテカルロ3着以内率マップ（あればXGBoostのprobより優先）
    sim = entry.get("simulation", {})

    for rank, num in enumerate(top5_nums):
        mark = MARKS[rank] if rank < len(MARKS) else "　"
        info = top5_detail.get(num, ev_map.get(num, {}))
        name = info.get("horse_name", "")
        if not name:
            name = f"{num}番"
        # MC確率を優先、なければXGBoost prob
        mc_data = sim.get(str(num), {})
        mc_rate = mc_data.get("top3_rate")
        prob = (mc_rate * 100) if mc_rate is not None else (info.get("prob", 0) * 100)
        ev_entry = ev_map.get(num, {})
        ev_val = ev_entry.get("ev_score")
        has_real_odds = ev_entry.get("odds") is not None
        ev_str = f" EV{ev_val:.2f}" if ev_val and has_real_odds else ""
        if prob > 0.01:
            lines1.append(f"{mark}{num}番 {name} {prob:.1f}%{ev_str}")
        else:
            lines1.append(f"{mark}{num}番 {name}")

    lines1.append(sep)

    # ★穴馬
    ana_num = entry.get("ana_horse_num")
    ana_info = entry.get("ana_horse_info", {})
    if ana_num and ana_num not in top5_nums[:5]:
        name = ana_info.get("horse_name", "")
        # MC確率を優先
        mc_ana = sim.get(str(ana_num), {})
        mc_ana_rate = mc_ana.get("top3_rate")
        prob = (mc_ana_rate * 100) if mc_ana_rate is not None else (ana_info.get("prob", 0) * 100)
        pop = ana_info.get("popularity", "?")
        if not name:
            for e in entry.get("ev_top3", []):
                if e.get("horse_number") == ana_num:
                    name = e.get("horse_name", "")
                    if mc_ana_rate is None:
                        prob = e.get("prob", 0) * 100
                    break
        if name:
            lines1.append(f"★穴 {ana_num}番 {name}（{prob:.1f}% {pop}番人気）")

    # ⚠危険馬（MC確率15%以上の馬は表示しない）
    for d in entry.get("dangerous_horses", []):
        num = d.get("horse_number", 0)
        mc_d = sim.get(str(num), {})
        mc_rate = mc_d.get("top3_rate")
        if mc_rate is not None and mc_rate >= 0.15:
            continue
        name = d.get("horse_name", "")
        reasons = d.get("reasons", [])
        reason = reasons[0] if reasons else "要注意"
        lines1.append(f"⚠ {num}番 {name}（{reason}）")

    lines1.append(sep)
    msg1 = "\n".join(lines1)

    # ── Message 2: 買い目（bet_strategy があれば使用）──────────
    _SEP = "─" * 16
    bs = entry.get("bet_strategy")

    if bs and bs.get("total_points", 0) > 0:
        header = f"💰 {race_name} 買い目"
        lines2 = [_SEP, header, _SEP]

        # 単勝
        if bs.get("tansho"):
            t = bs["tansho"][0]
            lines2.append(f"■ 単勝: {t['num']}番 {t.get('name', '')}")

        # 複勝
        if bs.get("fukusho"):
            f = bs["fukusho"][0]
            lines2.append(f"■ 複勝: {f['num']}番 {f.get('name', '')}")

        # 馬連 or ワイド
        if bs.get("use_wide") and bs.get("wide"):
            wide_str = " / ".join(f"{w['nums'][0]}-{w['nums'][1]}" for w in bs["wide"])
            lines2.append(f"■ ワイド({len(bs['wide'])}点): {wide_str}")
        if bs.get("umaren"):
            umaren_str = " / ".join(f"{u['nums'][0]}-{u['nums'][1]}" for u in bs["umaren"])
            lines2.append(f"■ 馬連({len(bs['umaren'])}点): {umaren_str}")

        # 3連複
        sr = bs.get("sanrenpuku", {})
        if sr:
            jiku = sr.get("jiku", [])
            aite = sr.get("aite", [])
            if len(jiku) == 1:
                sr_pt = len(list(combinations(aite, 2)))
                lines2.append(f"■ 3連複({sr_pt}点): 軸{jiku[0]}番 x {'/'.join(str(n) for n in aite)}")
            elif len(jiku) == 2:
                sr_pt = len(aite)
                lines2.append(f"■ 3連複({sr_pt}点): 軸{jiku[0]}-{jiku[1]}番 x {'/'.join(str(n) for n in aite)}")

        # 3連単
        st = bs.get("sanrentan", {})
        if st and st.get("first") and st.get("aite"):
            from itertools import permutations as _perm
            st_pt = len(list(_perm(st["aite"], 2)))
            lines2.append(f"■ 3連単({st_pt}点): 1着{st['first']}番 x {'/'.join(str(n) for n in st['aite'])}")

        total_cost = bs.get('total_cost', bs['total_points'] * 100)
        lines2 += [_SEP, f"合計 {bs['total_points']}点 / {total_cost:,}円"]
        if bs.get("strategy_note"):
            lines2.append(f"💡 {bs['strategy_note']}")
    else:
        # フォールバック: 従来の固定買い目
        nums = top5_nums
        if len(nums) < 2:
            return msg1, ""
        hon = nums[0]
        hon_name = entry.get("honmei", {}).get("horse_name", "")
        header = f"💰 {race_name}  買い目" if race_name else "💰 買い目"

        # フィルタ判定
        venue_code = race_id[4:6] if len(race_id) >= 6 else ""
        is_fukushima = venue_code == "03"
        is_old_upper = any(kw in race_name for kw in ("2勝クラス", "3勝クラス", "オープン"))
        if is_fukushima or is_old_upper:
            filter_label = "福島" if is_fukushima else "古馬上級"
            lines2 = [
                _SEP, header, _SEP,
                "■ 複勝（1点）", f"  {hon}番 {hon_name}",
                _SEP, f"合計 1点（{filter_label}フィルタ: 複勝のみ）",
            ]
        else:
            wide_pairs = list(combinations(nums[:3], 2))
            wide_str = " / ".join(f"{a}-{b}" for a, b in wide_pairs)
            partners = nums[1:5]
            ana_buy = entry.get("ana_horse_num")
            if ana_buy and ana_buy not in partners:
                partners = partners + [ana_buy]
            sanren_pt = len(list(combinations(partners, 2)))
            partners_str = "/".join(str(n) for n in partners)
            total = 1 + len(wide_pairs) + sanren_pt
            lines2 = [
                _SEP, header, _SEP,
                "■ 複勝（1点）", f"  {hon}番 {hon_name}",
                f"■ ワイド（{len(wide_pairs)}点）", f"  {wide_str}",
                f"■ 3連複（{sanren_pt}点）", f"  軸{hon}番 x {partners_str}",
                _SEP, f"合計 {total}点",
            ]

    msg2 = "\n".join(lines2)
    return msg1, msg2


# ══════════════════════════════════════════════════════════════
# 機能0: 金曜予告
# ══════════════════════════════════════════════════════════════

def run_preview_notify() -> None:
    """金曜21時: 今週末の重賞予告をXに投稿する。"""
    if os.environ.get("ENABLE_X_POST", "false").lower() != "true":
        logger.info("[preview] ENABLE_X_POST=false → スキップ")
        return

    cache = _load_cache()
    if not cache:
        # キャッシュがなければスクレイピングで取得
        session = requests.Session()
        grade_races = scrape_grade_race_ids(session)
    else:
        # キャッシュから重賞を抽出
        grade_races = []
        for race_id, entry in cache.items():
            if race_id.startswith("_"):
                continue
            if entry.get("is_grade") is False:
                continue
            grade_races.append({
                "race_id": race_id,
                "race_name": entry.get("race_name", race_id),
                "race_date": entry.get("race_date", ""),
                "venue": entry.get("venue", ""),
                "course_info": entry.get("course_info", ""),
            })

    if not grade_races:
        logger.info("[preview] 今週末の重賞が見つかりませんでした")
        return

    logger.info(f"[preview] 重賞 {len(grade_races)} レースの予告を投稿")
    try:
        from keiba_predictor.x_post import post_preview_tweet
        post_preview_tweet(grade_races)
    except Exception as e:
        logger.warning(f"[preview] X投稿失敗: {e}")


# ══════════════════════════════════════════════════════════════
# 機能1: 土日予想
# ══════════════════════════════════════════════════════════════

def run_predict_notify(
    webhook_url: Optional[str] = None,
    featured_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    test_race_id: Optional[str] = None,
    use_live: bool = False,
) -> None:
    """週末特別戦/重賞を予想してDiscordに送信する。

    特別戦/重賞: Discord通知 + ダッシュボード表示
    クラス戦/未勝利: 見送り（買い目なし）
    """
    webhook_url = _resolve_webhook(webhook_url)

    if model_path is None:
        model_path = MODEL_PATH
    if not model_path.exists():
        send_discord(webhook_url, "⚠️ モデルファイルが見つかりません。")
        return

    # GitHub Actions は UTC で動作するため JST に変換
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td2
    _jst = _tz(_td2(hours=9))
    today_str = _dt.now(_jst).date().isoformat()
    cache = _load_cache()

    # 先週のキャッシュを除外
    dates = _weekend_dates()
    weekend_set = {f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates}
    stale_ids = [
        rid for rid, entry in cache.items()
        if isinstance(entry, dict) and entry.get("race_date") and entry["race_date"] not in weekend_set
    ]
    if stale_ids:
        for rid in stale_ids:
            del cache[rid]
        _save_cache(cache)

    # ── 重賞レース ──────────────────────────────────────
    if test_race_id:
        grade_races = [{"race_id": test_race_id, "race_name": str(test_race_id), "race_date": ""}]
    else:
        session = requests.Session()
        logger.info("週末重賞を検索中...")
        grade_races = scrape_grade_race_ids(session)
        if not grade_races:
            grade_races = _load_featured_race_ids_for_weekend(featured_path)

    # キャッシュから当日の重賞を補完
    scraped_ids = {r["race_id"] for r in grade_races}
    for rid, entry in cache.items():
        if not isinstance(entry, dict) or rid in scraped_ids:
            continue
        if entry.get("race_date") == today_str and entry.get("predicted_top3_nums") and entry.get("is_grade") is not False:
            grade_races.append({"race_id": rid, "race_name": entry.get("race_name", rid), "race_date": today_str})

    # 発走時刻順にソート
    grade_races = sorted(grade_races, key=lambda r: cache.get(r["race_id"], {}).get("start_time", "99:99"))

    notified = 0
    for race in grade_races:
        race_id = race["race_id"]
        race_name = race.get("race_name", race_id)
        race_date = race.get("race_date", "")

        cached_date = cache.get(race_id, {}).get("race_date", race_date)
        if cached_date and cached_date != today_str and not test_race_id:
            logger.info(f"  スキップ（{cached_date} ≠ {today_str}）: {race_name}")
            continue

        cached_entry = cache.get(race_id, {})
        if not (cached_entry and cached_entry.get("predicted_top3_nums")):
            logger.info(f"  predict_live 実行: {race_name} ({race_id})")
            try:
                from keiba_predictor.model.predict import predict_live
                predict_live(race_id, notify=False, model_path=model_path)
                cache = _load_cache()
                cached_entry = cache.get(race_id, {})
            except Exception as e:
                logger.warning(f"  predict_live 失敗: {e}")
                send_discord(webhook_url, f"⚠️ {race_name} の予想生成に失敗: {e}")
                continue

        # 厳選（実買い）レースのみDiscord通知
        _bs = cached_entry.get("bet_strategy", {}) or {}
        if _bs.get("total_cost", 0) <= 0:
            logger.info(f"  見送りレース（予想通知なし）: {race_name}")
            continue

        msg = _format_simple_message(race_id, cached_entry)
        print(msg, flush=True)
        if send_discord(webhook_url, msg):
            notified += 1
            logger.info(f"  送信完了: {race_name}")

        # X（Twitter）に予想を投稿
        if os.environ.get("ENABLE_X_POST", "false").lower() == "true":
            try:
                from keiba_predictor.x_post import post_predict_tweet
                post_predict_tweet(race_name, cached_entry)
            except Exception as e:
                logger.warning(f"  [X] 予想投稿エラー: {e}")

    # ── 特別戦レース → ダッシュボード用にキャッシュのみ（Discord通知なし） ──
    if not test_race_id:
        flat_count = 0
        cache = _load_cache()
        for rid, entry in cache.items():
            if isinstance(entry, dict) and entry.get("is_grade") is False and entry.get("race_date") == today_str:
                flat_count += 1
        logger.info(f"特別戦/重賞以外: 当日 {flat_count} レースがキャッシュ済み（ダッシュボードのみ）")

    send_discord(webhook_url, f"✅ 特別戦/重賞 {notified} レース送信完了")


# ══════════════════════════════════════════════════════════════
# 機能2: 日曜結果
# ══════════════════════════════════════════════════════════════

def run_result_notify(
    webhook_url: Optional[str] = None,
    model_path: Optional[Path] = None,
    race_id: Optional[str] = None,
) -> None:
    """週末重賞の結果をスクレイピングし、予想との比較をDiscordに送信する。"""
    # 結果は DISCORD_RESULT_WEBHOOK_URL 優先（未設定ならデフォルト）
    result_url = os.environ.get("DISCORD_RESULT_WEBHOOK_URL", "")
    webhook_url = result_url if result_url else _resolve_webhook(webhook_url)

    session = requests.Session()
    cache   = _load_cache_for_result()  # 結果照合は13時スナップショット優先（後出し防止）

    # --race-id 指定時はそのレースのみ対象
    if race_id:
        cached = cache.get(race_id, {})
        race_name = cached.get("race_name", race_id)
        race_date = cached.get("race_date", "")
        grade_races = [{"race_id": race_id, "race_name": race_name, "race_date": race_date}]
        logger.info(f"指定レースID: {race_id} ({race_name})")
    else:
        # キャッシュ内の今週末レース（特別戦/重賞）を対象にする
        logger.info("キャッシュ内の今週末レースを結果照合対象にします...")
        weekend_set = set()
        for d in _weekend_dates():
            weekend_set.add(f"{d[:4]}-{d[4:6]}-{d[6:]}")
        logger.info(f"  今週末: {weekend_set}")
        grade_races = []
        for rid, entry in cache.items():
            if rid.startswith("_") or not isinstance(entry, dict):
                continue
            rd = entry.get("race_date", "")
            if rd and rd not in weekend_set:
                continue
            grade_races.append({
                "race_id": rid,
                "race_name": entry.get("race_name", rid),
                "race_date": rd,
            })
    if not grade_races:
        logger.warning("結果照合対象のレースがありません")
        return

    from keiba_predictor.scraper.netkeiba_scraper import scrape_race_result
    from keiba_predictor.history import (
        record_result, load_history,
        weekly_summary, cumulative_summary, hit_streak, format_summary_message,
    )
    from datetime import date as _date

    # results_history.csv に記録済みの race_id を取得（二重通知防止）
    history_ids: set[str] = set()
    try:
        hist = load_history()
        if hist is not None and not hist.empty and "race_id" in hist.columns:
            history_ids = set(hist["race_id"].astype(str))
            logger.info(f"results_history.csv: {len(history_ids)}件の記録済みレース")
    except Exception as e:
        logger.warning(f"results_history.csv 読み込み失敗: {e}")

    # 対象レース全通知済みなら何も送信せず終了（race_id指定時はスキップしない）
    if race_id:
        pending = grade_races
    else:
        pending = [r for r in grade_races
                   if not cache.get(r["race_id"], {}).get("result_notified")
                   and r["race_id"] not in history_ids]
    if not pending:
        logger.info("対象レース全通知済み → スキップ")
        return

    # 手動結果を読み込む
    manual_results: dict = {}
    if MANUAL_RESULTS.exists():
        try:
            manual_results = json.loads(MANUAL_RESULTS.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"manual_results.json 読み込み失敗: {e}")

    notified = 0
    flat_results: list[str] = []  # 特別戦まとめ用（将来拡張用に保持）
    seen_ids: set[str] = set()    # 重複チェック
    for race in grade_races:
        race_id   = race["race_id"]
        if race_id in seen_ids:
            continue
        seen_ids.add(race_id)
        # キャッシュのrace_nameを優先（スクレイプ由来は文字化けの可能性あり）
        cached_name = cache.get(race_id, {}).get("race_name", "")
        race_name = cached_name or race.get("race_name", race_id)
        race_date = race.get("race_date", "")

        # 結果通知済みならスキップ（--race-id指定時は強制再送）
        _forced = len(grade_races) == 1 and grade_races[0].get("race_id") == race_id
        if not _forced and (cache.get(race_id, {}).get("result_notified") or race_id in history_ids):
            logger.info(f"  結果通知済みスキップ: {race_name} ({race_id})")
            continue

        # 手動結果があればスクレイピングをスキップ
        manual = manual_results.get(race_id)
        if manual:
            logger.info(f"  手動結果を使用: {race_id} ({manual.get('race_name', race_name)})")
            result_nums = manual.get("result", [])
            # 手動結果から簡易DataFrameを構築
            actual_rows = []
            for i, num in enumerate(result_nums):
                actual_rows.append({
                    "finish_position": i + 1,
                    "horse_number": num,
                    "horse_name": "",
                })
            actual_df = pd.DataFrame(actual_rows) if actual_rows else None
            # 払戻金
            manual_pay = manual.get("payouts", {})
            payouts = {}
            if manual_pay.get("umaren"):
                payouts["馬連"] = [{"combo": "-".join(str(n) for n in result_nums[:2]),
                                    "amount": manual_pay["umaren"]}]
            if manual_pay.get("sanrenpuku"):
                payouts["三連複"] = [{"combo": "-".join(str(n) for n in sorted(result_nums[:3])),
                                      "amount": manual_pay["sanrenpuku"]}]
        else:
            # 結果スクレイピング
            actual_df = scrape_race_result(race_id, session)
            payouts = scrape_payouts(race_id, session) if actual_df is not None else {}

        if actual_df is None or actual_df.empty:
            send_discord(webhook_url, f"⚠️ **{race_name}** の結果が取得できませんでした。")
            continue

        # 予想キャッシュ取得
        pred = cache.get(race_id, {})
        if not pred:
            logger.warning(f"  予想キャッシュなし: {race_id}")
            pred = {"race_name": race_name, "race_date": race_date,
                    "honmei": {}, "taikou": {}, "ana": {}, "predicted_top3_nums": []}

        # manual の honmei / predicted_top3_nums で pred を上書き
        if manual:
            if manual.get("honmei") is not None:
                pred["honmei"] = {"horse_number": manual["honmei"],
                                  "horse_name": pred.get("honmei", {}).get("horse_name", ""),
                                  "prob": pred.get("honmei", {}).get("prob", 0)}
            if manual.get("predicted_top3_nums"):
                pred["predicted_top3_nums"] = manual["predicted_top3_nums"]

        # Discord通知は厳選（実買い）レースのみ
        bs = pred.get("bet_strategy", {}) or {}
        is_betted = bs.get("total_cost", 0) > 0

        if is_betted:
            msg = _fmt_result(race_name, race_date, actual_df, pred, payouts, manual=manual, race_id=race_id, is_grade=True)
            if send_discord(webhook_url, msg):
                notified += 1
                logger.info(f"  送信: {race_name}")
        else:
            logger.info(f"  見送りレース（通知なし）: {race_name}")

        # 結果通知済みフラグをキャッシュに保存
        if race_id in cache:
            cache[race_id]["result_notified"] = True
            _save_cache(cache)

        # 的中実績を CSV に記録
        if manual and "fukusho_hit" in manual:
            try:
                _record_manual_result(race_id, race_name, race_date, pred, manual)
            except Exception as e:
                logger.warning(f"  [history] 手動記録失敗 ({race_name}): {e}")
        else:
            try:
                record_result(race_id, race_name, race_date, pred, actual_df, payouts)
            except Exception as e:
                logger.warning(f"  [history] 記録失敗 ({race_name}): {e}")

        # X（Twitter）に結果を投稿
        if os.environ.get("ENABLE_X_POST", "false").lower() == "true":
            try:
                from keiba_predictor.x_post import post_result_tweet
                post_result_tweet(race_name, actual_df, pred, payouts)
            except Exception as e:
                logger.warning(f"  [X] 結果投稿エラー: {e}")

    # 特別戦まとめ送信（無効化: 特別戦/重賞のみ通知）
    # if flat_results:
    #     flat_hits = sum(1 for r in flat_results if r.startswith("✅"))
    #     flat_msg = f"📋 **特別戦結果** {flat_hits}/{len(flat_results)}的中\n" + "\n".join(flat_results)
    #     send_discord(webhook_url, flat_msg)

    if notified > 0:
        send_discord(webhook_url, f"✅ {notified}レース結果送信完了")

    # 週次サマリーを Discord に送信（日曜のみ）
    try:
        today = _date.today()
        if today.weekday() == 6:  # 日曜日
            from keiba_predictor.analysis.loss_analysis import analyze_week
            summary_msg = analyze_week()
            if summary_msg:
                send_discord(webhook_url, summary_msg)
    except Exception as e:
        logger.warning(f"  [history] サマリー送信失敗: {e}")

    # 日曜日に週次サマリーを X に投稿
    if os.environ.get("ENABLE_X_POST", "false").lower() == "true":
        try:
            today = _date.today()
            if today.weekday() == 6:  # 日曜日
                from datetime import timedelta
                hist_df = load_history()
                week_start = today - timedelta(days=today.weekday())  # 月曜
                week_end = today
                ws = pd.Timestamp(week_start)
                we = pd.Timestamp(week_end)
                mask = (hist_df["date"] >= ws) & (
                    hist_df["date"] <= we + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
                wdf = hist_df[mask]
                if not wdf.empty:
                    results = []
                    for _, row in wdf.iterrows():
                        results.append({
                            "race_name": str(row.get("race_name", "")),
                            "fukusho": bool(row.get("fukusho_hit", False)),
                            "umaren": bool(row.get("umaren_hit", False)),
                            "sanren": bool(row.get("sanrenpuku_hit", False)),
                            "bet": int(row.get("bet_total", 0)),
                            "return_total": int(row.get("return_total", 0)),
                        })
                    from keiba_predictor.x_post import post_weekly_summary_tweet
                    post_weekly_summary_tweet(results)
        except Exception as e:
            logger.warning(f"  [X] 週次サマリー投稿エラー: {e}")


# ══════════════════════════════════════════════════════════════
# ユーティリティ
# ══════════════════════════════════════════════════════════════

def _resolve_webhook(url: Optional[str]) -> str:
    """引数 → 環境変数 → エラー の順で Webhook URL を解決する。"""
    if url:
        return url
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        raise ValueError(
            "Discord Webhook URL が未設定です。\n"
            "環境変数 DISCORD_WEBHOOK_URL を設定するか "
            "--webhook-url オプションを使用してください。"
        )
    return url


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="週末重賞 Discord 通知")
    p.add_argument("--mode", choices=["predict", "result"], required=True,
                   help="predict=金曜予想 / result=日曜結果")
    p.add_argument("--webhook-url", help="Discord Webhook URL（未指定=環境変数）")
    args = p.parse_args()

    if args.mode == "predict":
        run_predict_notify(webhook_url=args.webhook_url)
    else:
        run_result_notify(webhook_url=args.webhook_url)


if __name__ == "__main__":
    main()
