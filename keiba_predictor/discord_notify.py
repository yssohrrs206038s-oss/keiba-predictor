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
    特徴量 CSV  : keiba_predictor/data/featured_races.csv
    ない場合は先に: python -m keiba_predictor.main all --start 2023-01 --end YYYY-MM
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
from keiba_predictor.ai_comment import generate_comments

logger = logging.getLogger(__name__)

# ── パス定数 ────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model" / "xgb_model.pkl"
PRED_CACHE = DATA_DIR / "predictions_cache.json"   # 予想キャッシュ
MANUAL_RESULTS = DATA_DIR / "manual_results.json"  # 手動結果入力

# 重賞判定 (G1/G2/G3 を含む括弧表記)
GRADE_RE = re.compile(r"\(G[Ⅰ-Ⅲ1-3]\)|\(GI{1,3}\)")

MARK = {"honmei": "◎", "taikou": "○", "ana": "△", "hoshi": "☆"}


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
    today   = date.today()
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
        logger.warning("featured_races.csv が見つかりません")
        return

    try:
        df = pd.read_csv(featured_path, encoding="utf-8-sig", dtype={"race_id": str})
    except Exception as e:
        logger.warning(f"featured_races.csv 読み込み失敗: {e}")
        return

    cache = _load_cache()
    dates = _weekend_dates()

    VENUE_MAP = {
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
        "05": "東京", "06": "中山", "07": "中京", "08": "京都",
        "09": "阪神", "10": "小倉",
    }

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
        logger.warning(f"featured_races.csv が見つかりません: {featured_path}")
        return []

    try:
        df = pd.read_csv(featured_path, encoding="utf-8-sig", dtype={"race_id": str})
    except Exception as e:
        logger.warning(f"featured_races.csv 読み込み失敗: {e}")
        return []

    if "race_id" not in df.columns:
        return []

    # race_date 列がない新フォーマット（race_id, race_name, grade）の場合は全件返す
    if "race_date" not in df.columns:
        dates = _weekend_dates()
        sat = f"{dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]}"
        result = []
        for _, row in df.drop_duplicates(subset=["race_id"]).iterrows():
            result.append({
                "race_id":   str(row["race_id"]),
                "race_name": str(row.get("race_name", row["race_id"])),
                "race_date": sat,
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
    """予想キャッシュを読み込む。当日のスナップショットがあればそちらを優先する。"""
    snapshot_path = DATA_DIR / f"predictions_snapshot_{date.today().strftime('%Y%m%d')}.json"
    if snapshot_path.exists():
        logger.info(f"スナップショットを使用: {snapshot_path.name}")
        with open(snapshot_path, encoding="utf-8") as f:
            return json.load(f)
    if PRED_CACHE.exists():
        with open(PRED_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


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


def _store_prediction(race_id: str, race_name: str, race_date: str,
                      result_df: pd.DataFrame,
                      ai_comments: Optional[dict] = None,
                      course_info: str = "",
                      start_time: str = "",
                      venue: str = "") -> None:
    """予想結果をキャッシュに保存する（日曜結果比較・note レポート生成に使用）。"""
    cache = _load_cache()

    # オッズが全馬同一（仮オッズ）かチェック
    _odds_ser = pd.to_numeric(result_df["odds"], errors="coerce").dropna()
    _all_same = len(_odds_ser) > 1 and _odds_ser.nunique() == 1

    def _row(df: pd.DataFrame, idx: int) -> dict:
        if len(df) <= idx:
            return {}
        r = df.iloc[idx]
        raw_o = pd.to_numeric(r.get("odds"), errors="coerce")
        o_val = None if (not pd.notna(raw_o) or _all_same) else round(float(raw_o), 1)
        entry = {
            "horse_number": int(r["horse_number"]) if pd.notna(r.get("horse_number")) else None,
            "horse_name":   str(r.get("horse_name", "")),
            "prob":         float(r["prob_top3"]),
            "odds":         o_val,
        }
        # SHAP値の上位要因を追加
        shap_top = r.get("shap_top")
        if isinstance(shap_top, list) and shap_top:
            entry["shap_top"] = shap_top
        return entry

    # 穴馬: AI確率35%以上 & 6番人気以下 & TOP3外 → AI確率最高の1頭
    ana: dict = {}
    try:
        top3_set = set()
        for _, r in result_df.head(3).iterrows():
            v = r.get("horse_number")
            if pd.notna(v):
                top3_set.add(int(v))
        rest = result_df.iloc[3:] if len(result_df) > 3 else pd.DataFrame()
        if not rest.empty:
            rest_prob = pd.to_numeric(rest["prob_top3"], errors="coerce")
            rest_pop = pd.to_numeric(rest.get("popularity", pd.Series(dtype=float)), errors="coerce")
            cands = rest[(rest_prob >= 0.35) & (rest_pop >= 6)]
            if not cands.empty:
                best_idx = cands["prob_top3"].idxmax()
                ana = _row(result_df, result_df.index.get_loc(best_idx))
    except Exception as e:
        logger.warning(f"穴馬検出失敗: {e}")
    if not ana:
        ana = _row(result_df, 2)

    top3_nums = []
    for _, row in result_df.head(3).iterrows():
        v = row.get("horse_number")
        if pd.notna(v):
            top3_nums.append(int(v))

    top5_nums = []
    for _, row in result_df.head(5).iterrows():
        v = row.get("horse_number")
        if pd.notna(v):
            top5_nums.append(int(v))

    # 穴馬（3連複用）: AI確率35%以上 & 6番人気以下 & TOP5外 → AI確率最高
    ana_horse_num: Optional[int] = None
    if len(result_df) > 5:
        rest = result_df.iloc[5:]
        rest_prob = pd.to_numeric(rest["prob_top3"], errors="coerce")
        rest_pop = pd.to_numeric(rest.get("popularity", pd.Series(dtype=float)), errors="coerce")
        cands = rest[(rest_prob >= 0.35) & (rest_pop >= 6)]
        if not cands.empty:
            best = cands.nlargest(1, "prob_top3").iloc[0]
            v = best.get("horse_number")
            if pd.notna(v):
                ana_horse_num = int(v)

    # EV・危険馬データ（_calc_ev_and_flags 済みならそのまま使う）
    if "ev_score" not in result_df.columns:
        result_df = calc_ev_and_flags(result_df)

    # オッズが全馬同一（仮オッズ）かチェック
    odds_vals = pd.to_numeric(result_df["odds"], errors="coerce").dropna()
    all_same_odds = len(odds_vals) > 1 and odds_vals.nunique() == 1
    if all_same_odds:
        logger.warning(f"全馬同一オッズ({odds_vals.iloc[0]}) → 仮オッズのため odds=None として保存")

    ev_top3: list[dict] = []
    for _, r in result_df[result_df["ev_score"].notna()].nlargest(3, "ev_score").iterrows():
        raw_odds = pd.to_numeric(r.get("odds"), errors="coerce")
        odds_val = None if (not pd.notna(raw_odds) or all_same_odds) else round(float(raw_odds), 1)
        ev_top3.append({
            "horse_number": int(r["horse_number"]) if pd.notna(r.get("horse_number")) else None,
            "horse_name":   str(r.get("horse_name", "")),
            "ev_score":     round(float(r["ev_score"]), 3),
            "prob":         round(float(r["prob_top3"]), 4),
            "odds":         odds_val,
        })

    dangerous: list[dict] = []
    for _, r in result_df[result_df["is_dangerous"]].iterrows():
        dangerous.append({
            "horse_number": int(r["horse_number"]) if pd.notna(r.get("horse_number")) else None,
            "horse_name":   str(r.get("horse_name", "")),
            "popularity":   int(pd.to_numeric(r.get("popularity"), errors="coerce") or 0),
            "prob":         round(float(r["prob_top3"]), 4),
            "reasons":      list(r.get("danger_reasons", [])),
        })

    # 上位5頭の詳細情報（Discord通知用）
    predicted_top5: list[dict] = []
    for i in range(min(5, len(result_df))):
        r = result_df.iloc[i]
        raw_o = pd.to_numeric(r.get("odds"), errors="coerce")
        o_val = None if (not pd.notna(raw_o) or _all_same) else round(float(raw_o), 1)
        predicted_top5.append({
            "horse_number": int(r["horse_number"]) if pd.notna(r.get("horse_number")) else None,
            "horse_name":   str(r.get("horse_name", "")),
            "prob":         round(float(r["prob_top3"]), 4),
            "odds":         o_val,
        })

    cache[race_id] = {
        "race_name":           race_name,
        "race_date":           race_date,
        "start_time":          start_time,
        "venue":               venue,
        "course_info":         course_info,
        "honmei":              _row(result_df, 0),
        "taikou":              _row(result_df, 1),
        "ana":                 ana,
        "predicted_top3_nums": top3_nums,
        "predicted_top5_nums": top5_nums,
        "ana_horse_num":       ana_horse_num,
        "ana_horse_info":      _ana_horse_info(result_df, ana_horse_num),
        "predicted_top5":      predicted_top5,
        "ev_top3":             ev_top3,
        "dangerous_horses":    dangerous,
        "ai_comments":         ai_comments or {},
    }

    # 買い目自動決定
    try:
        from keiba_predictor.model.predict import _decide_bet_strategy
        cache[race_id]["bet_strategy"] = _decide_bet_strategy(result_df)
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
        # 上位5頭のみ保存
        top5_mc = {}
        for num in top5_nums[:5]:
            k = str(num)
            if k in mc_result:
                top5_mc[k] = mc_result[k]
        cache[race_id]["simulation"] = top5_mc
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
    url  = RACE_RESULT_URL.format(race_id=race_id)
    soup = _get(url, session)
    if soup is None:
        return {}

    payouts: dict[str, list] = {}

    for table in soup.select("table.pay_table_01"):
        current_type = None
        for tr in table.select("tr"):
            th = tr.select_one("th")
            tds = tr.select("td")
            if th:
                current_type = th.get_text(strip=True)
            if not current_type or len(tds) < 2:
                continue

            combos  = tds[0].get_text(" ", strip=True)
            amounts = tds[1].get_text(" ", strip=True)

            # 金額を数値に（"¥1,450" → 1450）
            def _parse_yen(s: str) -> Optional[int]:
                s = re.sub(r"[¥,\s]", "", s)
                try:
                    return int(s)
                except ValueError:
                    return None

            amt = _parse_yen(amounts)
            payouts.setdefault(current_type, []).append({
                "combo":  combos,
                "amount": amt,
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
                manual: Optional[dict] = None) -> str:
    """日曜結果メッセージを生成する。"""
    RULE = "━" * 24
    lines = [f"🏆 【KEIBA EDGE】{race_name} 結果  {race_date}", RULE]

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

    # honmei: manual > predicted_nums[0] > pred["honmei"]
    if manual and manual.get("honmei") is not None:
        honmei_num = int(manual["honmei"])
    elif predicted_nums:
        honmei_num = predicted_nums[0]
    else:
        honmei_num = pred.get("honmei", {}).get("horse_number")
    honmei_name = num_to_name.get(honmei_num, "") if honmei_num else ""

    # manual_results.json のフラグがあればそちらを優先
    if manual and "fukusho_hit" in manual:
        fukusho_hit = manual["fukusho_hit"]
        umaren_hit  = manual.get("umaren_hit", False)
        sanren_hit  = manual.get("sanrenpuku_hit", False)
        manual_pay  = manual.get("payouts", {})
        umaren_pay  = f"¥{manual_pay['umaren']:,}" if manual_pay.get("umaren") else ""
        sanren_pay  = f"¥{manual_pay['sanrenpuku']:,}" if manual_pay.get("sanrenpuku") else ""
    else:
        # 自動判定
        fukusho_hit = (honmei_num is not None) and (int(honmei_num) in actual_top3_nums)
        umaren_hit, umaren_pay = _check_umaren_raw(predicted_nums, actual_top3_nums, payouts)
        ana_horse_num = pred.get("ana_horse_num")
        sanren_hit, sanren_pay = _check_sanrenpuku_raw(predicted_nums, actual_top3_nums, payouts, ana_horse_num)

    lines.append(f"複勝  {'✅ 的中' if fukusho_hit else '❌ ハズレ'}（◎{honmei_num}番{honmei_name}）")

    umaren_line = f"馬連  {'✅ 的中' if umaren_hit else '❌ ハズレ'}"
    if umaren_hit and umaren_pay:
        umaren_line += f"（配当{re.sub(r'[¥,]', '', str(umaren_pay))}円）"
    lines.append(umaren_line)

    sanren_line = f"3連複 {'✅ 的中' if sanren_hit else '❌ ハズレ'}"
    if sanren_hit and sanren_pay:
        sanren_line += f"（配当{re.sub(r'[¥,]', '', str(sanren_pay))}円）"
    lines.append(sanren_line)

    return "\n".join(lines)


def _get_payout(payouts: dict, bet_type: str, combo: str) -> str:
    """払戻金辞書から指定の組み合わせ・金額を文字列で返す。"""
    for entry in payouts.get(bet_type, []):
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
) -> tuple[bool, str]:
    """3連複的中判定。買い目は軸(top5[0])×相手(top5[1:5]+穴馬)。(hit, pay_str) を返す。"""
    if len(predicted_nums) < 2 or len(actual_top3_nums) < 3:
        return False, ""
    # 買い目: 軸 = predicted_nums[0], 相手 = predicted_nums[1:5] + 穴馬
    axis = predicted_nums[0]
    partners = list(predicted_nums[1:5])
    if ana_horse_num and ana_horse_num not in partners:
        partners.append(ana_horse_num)
    # 軸が3着以内に含まれることが前提
    if axis not in actual_top3_nums:
        return False, ""
    # 相手2頭が3着以内に含まれるか
    actual_set = set(actual_top3_nums[:3])
    for pair in combinations(partners, 2):
        if {axis, pair[0], pair[1]} == actual_set:
            combo = "-".join(str(n) for n in sorted([axis, pair[0], pair[1]]))
            pay = _get_payout(payouts, "三連複", combo)
            return True, pay
    return False, ""


def _format_prediction_from_cache(race_name: str, entry: dict) -> tuple[str, str]:
    """predictions_cache.json のエントリからDiscord用メッセージ(予想・買い目)を生成する。"""
    sep = "━" * 20
    course_info = entry.get("course_info", "")
    ai_comments = entry.get("ai_comments", {})

    # ── Message 1: 予想 ───────────────────────────────────────
    lines1 = [sep, f"🏇 {race_name}"]
    if course_info:
        lines1.append(course_info)
    lines1.append(sep)

    MARKS = ["◎", "○", "▲", "△", "　"]
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

    for rank, num in enumerate(top5_nums):
        mark = MARKS[rank] if rank < len(MARKS) else "　"
        info = top5_detail.get(num, ev_map.get(num, {}))
        name = info.get("horse_name", "")
        if not name:
            name = f"{num}番"
        prob = info.get("prob", 0) * 100
        ev_entry = ev_map.get(num, {})
        ev_val = ev_entry.get("ev_score")
        has_real_odds = ev_entry.get("odds") is not None
        ev_str = f" EV{ev_val:.2f}" if ev_val and has_real_odds else ""
        # predicted_top5にデータがない馬は確率非表示（馬番のみ）
        if prob > 0.01:
            lines1.append(f"{mark} {num}番 {name}　{prob:.1f}%{ev_str}")
        else:
            lines1.append(f"{mark} {num}番 {name}")

    lines1.append(sep)

    # ★穴馬
    ana_num = entry.get("ana_horse_num")
    ana_info = entry.get("ana_horse_info", {})
    if ana_num and ana_num not in top5_nums[:5]:
        name = ana_info.get("horse_name", "")
        prob = ana_info.get("prob", 0) * 100
        pop = ana_info.get("popularity", "?")
        if not name:
            # フォールバック: ev_top3 から取得
            for e in entry.get("ev_top3", []):
                if e.get("horse_number") == ana_num:
                    name = e.get("horse_name", "")
                    prob = e.get("prob", 0) * 100
                    break
        if name:
            lines1.append(f"★穴 {ana_num}番{name}（AI確率{prob:.1f}% {pop}番人気）")
            lines1.append(f"　→ AIが高評価も市場は低評価！")

    # ⚠危険馬
    for d in entry.get("dangerous_horses", []):
        num = d.get("horse_number", 0)
        name = d.get("horse_name", "")
        reasons = d.get("reasons", [])
        reason = reasons[0] if reasons else "要注意"
        lines1.append(f"⚠危険 {num}番{name}（{reason}）")

    # 📊 モンテカルロ分析
    sim = entry.get("simulation", {})
    if sim:
        MARKS_MC = ["◎", "○", "▲", "△", "　"]
        lines1.append(sep)
        lines1.append("📊 モンテカルロ分析（1万回）")
        lines1.append(sep)
        for rank, num in enumerate(top5_nums[:5]):
            mc = sim.get(str(num))
            if not mc:
                continue
            mark = MARKS_MC[rank] if rank < len(MARKS_MC) else "　"
            rate = mc.get("top3_rate", 0) * 100
            is_stable = mc.get("is_stable", False)
            tag = "🔒安定軸" if is_stable else "⚡展開依存"
            sc = mc.get("scenario", {})
            hi = sc.get("high_pace", 0) * 100
            sl = sc.get("slow_pace", 0) * 100
            lines1.append(f"{mark}{num}番 3着以内{rate:.1f}% {tag}")
            lines1.append(f"　ハイペース{hi:.0f}% / スロー{sl:.0f}%")

    lines1.append(sep)
    msg1 = "\n".join(lines1)

    # ── Message 2: 買い目（bet_strategy があれば使用）──────────
    _SEP = "━" * 20
    bs = entry.get("bet_strategy")

    if bs and bs.get("total_points", 0) > 0:
        header = f"💰 {race_name}  買い目（AI自動決定）" if race_name else "💰 買い目（AI自動決定）"
        lines2 = [_SEP, header, _SEP]

        # 複勝
        if bs.get("fukusho"):
            f = bs["fukusho"][0]
            lines2 += [
                f"■ 複勝（{len(bs['fukusho'])}点）",
                f"　{f['num']}番 {f.get('name', '')}",
            ]

        # 馬連 or ワイド
        if bs.get("use_wide") and bs.get("wide"):
            wide_str = " / ".join(f"{w['nums'][0]}-{w['nums'][1]}" for w in bs["wide"])
            lines2 += [f"■ ワイド（{len(bs['wide'])}点）", f"　{wide_str}"]
        if bs.get("umaren"):
            umaren_str = " / ".join(f"{u['nums'][0]}-{u['nums'][1]}" for u in bs["umaren"])
            lines2 += [f"■ 馬連（{len(bs['umaren'])}点）", f"　{umaren_str}"]

        # 3連複
        sr = bs.get("sanrenpuku", {})
        if sr:
            jiku = sr.get("jiku", [])
            aite = sr.get("aite", [])
            if len(jiku) == 1:
                sr_pt = len(list(combinations(aite, 2)))
                lines2 += [
                    f"■ 3連複（{sr_pt}点）",
                    f"　軸 {jiku[0]}番",
                    f"　× {'/'.join(str(n) for n in aite)}",
                ]
            elif len(jiku) == 2:
                sr_pt = len(aite)
                lines2 += [
                    f"■ 3連複（{sr_pt}点）",
                    f"　軸 {jiku[0]}-{jiku[1]}番",
                    f"　× {'/'.join(str(n) for n in aite)}",
                ]

        lines2 += [_SEP, f"合計 {bs['total_points']}点", _SEP]
        if bs.get("strategy_note"):
            lines2.append(f"💡 {bs['strategy_note']}")
    else:
        # フォールバック: 従来の固定買い目
        nums = top5_nums
        if len(nums) < 2:
            return msg1, ""
        hon = nums[0]
        hon_name = entry.get("honmei", {}).get("horse_name", "")
        umaren_pairs = list(combinations(nums[:3], 2))
        umaren_str = " / ".join(f"{a}-{b}" for a, b in umaren_pairs)
        partners = nums[1:5]
        ana_buy = entry.get("ana_horse_num")
        if ana_buy and ana_buy not in partners:
            partners = partners + [ana_buy]
        sanren_pt = len(list(combinations(partners, 2)))
        partners_str = "/".join(str(n) for n in partners)
        total = 1 + len(umaren_pairs) + sanren_pt
        header = f"💰 {race_name}  買い目" if race_name else "💰 買い目"
        lines2 = [
            _SEP, header, _SEP,
            "■ 複勝（1点）", f"　{hon}番 {hon_name}",
            f"■ 馬連（{len(umaren_pairs)}点）", f"　{umaren_str}",
            f"■ 3連複（{sanren_pt}点）", f"　軸 {hon}番", f"　× {partners_str}",
            _SEP, f"合計 {total}点", _SEP,
        ]

    msg2 = "\n".join(lines2)
    return msg1, msg2


# ══════════════════════════════════════════════════════════════
# 機能1: 金曜予想
# ══════════════════════════════════════════════════════════════

def run_predict_notify(
    webhook_url: Optional[str] = None,
    featured_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    test_race_id: Optional[str] = None,
    use_live: bool = False,
) -> None:
    """週末重賞を予想して Discord に送信し、結果をキャッシュに保存する。

    Args:
        test_race_id: 指定時は週末重賞検索をスキップして該当race_idのみテスト送信する。
        use_live:     True のとき出馬表をリアルタイム取得して予測する。
                      出馬表未確定・取得失敗の場合は featured_races.csv にフォールバック。
    """
    webhook_url = _resolve_webhook(webhook_url)

    if featured_path is None:
        featured_path = DATA_DIR / "featured_races.csv"
    if model_path is None:
        model_path = MODEL_PATH

    # 前提ファイル確認
    if not featured_path.exists():
        send_discord(webhook_url,
            "⚠️ 特徴量データが見つかりません。\n"
            "```\npython -m keiba_predictor.main all --start 2023-01 --end YYYY-MM\n```")
        return
    if not model_path.exists():
        send_discord(webhook_url,
            "⚠️ モデルファイルが見つかりません。\n"
            "```\npython -m keiba_predictor.main train\n```")
        return

    model_bundle = load_model(model_path)
    df_all = pd.read_csv(featured_path, encoding="utf-8-sig")

    # --test-race-id が指定された場合は重賞検索をスキップ
    from_featured = False
    if test_race_id:
        race_name = str(test_race_id)
        grade_races = [{"race_id": test_race_id, "race_name": race_name, "race_date": "（テスト）"}]
        logger.info(f"テストモード: race_id={test_race_id} race_name={race_name}")
        send_discord(webhook_url, f"🧪 **テスト送信** race_id={test_race_id}  {race_name}")
    else:
        session = requests.Session()
        logger.info("週末重賞を検索中...")
        grade_races = scrape_grade_race_ids(session)
        if not grade_races:
            dates = _weekend_dates()
            sat = f"{dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]}"
            sun = f"{dates[1][:4]}-{dates[1][4:6]}-{dates[1][6:]}"
            logger.warning(f"スクレイピングで重賞0件 ({sat}/{sun}) → featured_races.csv にフォールバック")
            grade_races = _load_featured_race_ids_for_weekend(featured_path)
            if not grade_races:
                msg = (
                    f"🏇 今週末（{sat} / {sun}）の重賞レース情報が取得できませんでした。\n"
                    "以下のいずれかが原因の可能性があります:\n"
                    "• netkeibaへのアクセス失敗（ネットワーク/レート制限）\n"
                    "• 今週末に重賞レースがない\n"
                    "• HTMLの構造変更によりレース名が取得できていない\n\n"
                    "💡 featured_races.csv に今週のレースIDを手動登録することで\n"
                    "スクレイピング失敗時でも予想を実行できます。\n\n"
                    "デバッグ用ログ確認:\n"
                    "```\n"
                    "python -m keiba_predictor.main notify --mode predict --debug\n"
                    "```"
                )
                send_discord(webhook_url, msg)
                return
            from_featured = True
            send_discord(webhook_url,
                f"⚠️ スクレイピング失敗 → featured_races.csv から {len(grade_races)} レースを使用")
        dates_str = " / ".join(sorted({r["race_date"] for r in grade_races}))
        send_discord(webhook_url,
            f"🏇 **今週末の重賞予想** ({dates_str})  全{len(grade_races)}レース")

    notified = 0
    cache = _load_cache()

    # 当週土日以外のレースをキャッシュから除外
    dates = _weekend_dates()
    weekend_set = {
        f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates
    }
    stale_ids = [
        rid for rid, entry in cache.items()
        if entry.get("race_date") and entry["race_date"] not in weekend_set
    ]
    if stale_ids:
        for rid in stale_ids:
            name = cache[rid].get("race_name", rid)
            logger.info(f"  先週レース除外: {name} ({cache[rid].get('race_date')})")
            del cache[rid]
        _save_cache(cache)
        logger.info(f"  {len(stale_ids)} 件の先週レースをキャッシュから削除")

    # 当日のレースのみ通知（土曜実行→土曜レース、日曜実行→日曜レース）
    today_str = date.today().isoformat()  # "YYYY-MM-DD"
    logger.info(f"本日: {today_str}")

    # 発走時刻順にソート（キャッシュの start_time を使用）
    grade_races = sorted(
        grade_races,
        key=lambda r: cache.get(r["race_id"], {}).get("start_time", "99:99"),
    )

    for race in grade_races:
        race_id   = race["race_id"]
        race_name = race.get("race_name", race_id)
        race_date = race.get("race_date", "")

        # キャッシュの race_date も確認
        cached_date = cache.get(race_id, {}).get("race_date", race_date)
        effective_date = cached_date or race_date

        if effective_date and effective_date != today_str:
            logger.info(f"  スキップ（{effective_date} ≠ {today_str}）: {race_name}")
            continue

        # ── キャッシュ優先: predictions_cache.json にデータがあればそれを使う ──
        # predict_live() を再実行するとcleaned_races.csvが無い環境で確率が壊れるため、
        # キャッシュに予想データがあれば常にそちらを使う
        cached_entry = cache.get(race_id, {})
        has_cache = bool(cached_entry and cached_entry.get("predicted_top3_nums"))

        if has_cache:
            race_name = cached_entry.get("race_name", race_name)
            logger.info(f"  キャッシュから予想を読み込み: {race_name} ({race_id})")

            # AI解説が空の場合は再生成を試みる
            if not cached_entry.get("ai_comments"):
                logger.info(f"  AI解説が空 → 再生成を試みます: {race_name}")
                try:
                    # キャッシュのpredicted_top5から簡易DataFrameを構築してAI解説生成
                    top5_data = cached_entry.get("predicted_top5", [])
                    if top5_data:
                        import pandas as _pd
                        ai_df = _pd.DataFrame(top5_data)
                        ai_df["prob_top3"] = ai_df["prob"]
                        if "ev_score" not in ai_df.columns:
                            ai_df["ev_score"] = ai_df.get("odds", _pd.Series(dtype=float)) * ai_df["prob"]
                        course_info = cached_entry.get("course_info", "")
                        ai_comments = generate_comments(
                            ai_df, race_name=race_name, course_info=course_info)
                        if ai_comments:
                            cached_entry["ai_comments"] = ai_comments
                            cache[race_id] = cached_entry
                            _save_cache(cache)
                            logger.info(f"  AI解説再生成成功: {len(ai_comments)} 頭分")
                except Exception as e:
                    logger.warning(f"  AI解説再生成失敗（続行）: {e}")

            msg1, msg2 = _format_prediction_from_cache(race_name, cached_entry)
        else:
            # キャッシュになければ predict_live() で生成
            logger.info(f"  キャッシュなし → predict_live 実行: {race_name} ({race_id})")
            try:
                from keiba_predictor.model.predict import predict_live
                result = predict_live(race_id, notify=False, model_path=model_path)
                cached_entry = _load_cache().get(race_id, {})
                ai_comments = cached_entry.get("ai_comments", {})
                course_info = cached_entry.get("course_info", "")
                race_name   = cached_entry.get("race_name", race_name)
                logger.info(f"  predict_live 成功: {race_name} ({race_id})")
                msg1, msg2 = format_prediction(result, race_name=race_name,
                                               ai_comments=ai_comments, course_info=course_info)
            except Exception as e:
                import traceback
                logger.warning(f"  predict_live 失敗: {traceback.format_exc()}")
                send_discord(webhook_url,
                    f"⚠️ **{race_name}** の予想生成に失敗しました: {e}")
                continue

        print(msg1, flush=True)
        print(msg2, flush=True)

        # Discord に送信
        ok = send_discord(webhook_url, msg1)
        if ok:
            send_discord(webhook_url, msg2)
            notified += 1
            logger.info(f"  送信完了: {race_name}")

        # X（Twitter）に予想を投稿
        if os.environ.get("ENABLE_X_POST", "false").lower() == "true":
            try:
                from keiba_predictor.x_post import post_predict_tweet
                post_predict_tweet(race_name, cached_entry)
            except Exception as e:
                logger.warning(f"  [X] 予想投稿エラー: {e}")

    send_discord(webhook_url, f"✅ {notified}/{len(grade_races)} レース送信完了")


# ══════════════════════════════════════════════════════════════
# 機能2: 日曜結果
# ══════════════════════════════════════════════════════════════

def run_result_notify(
    webhook_url: Optional[str] = None,
    model_path: Optional[Path] = None,
    race_id: Optional[str] = None,
) -> None:
    """週末重賞の結果をスクレイピングし、予想との比較をDiscordに送信する。"""
    webhook_url = _resolve_webhook(webhook_url)

    session = requests.Session()
    cache   = _load_cache()

    # --race-id 指定時はそのレースのみ対象
    if race_id:
        cached = cache.get(race_id, {})
        race_name = cached.get("race_name", race_id)
        race_date = cached.get("race_date", "")
        grade_races = [{"race_id": race_id, "race_name": race_name, "race_date": race_date}]
        logger.info(f"指定レースID: {race_id} ({race_name})")
    else:
        # 今週末の重賞IDを取得
        logger.info("今週末の重賞を検索中...")
        grade_races = scrape_grade_race_ids(session)
    if not grade_races:
        dates = _weekend_dates()
        sat = f"{dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]}"
        sun = f"{dates[1][:4]}-{dates[1][4:6]}-{dates[1][6:]}"
        logger.warning(f"スクレイピングで重賞0件 ({sat}/{sun}) → featured_races.csv にフォールバック")
        grade_races = _load_featured_race_ids_for_weekend()
        if not grade_races:
            msg = (
                f"🏆 今週末（{sat} / {sun}）の重賞レース情報が取得できませんでした。\n"
                "netkeibaへのアクセス失敗または重賞レースなしの可能性があります。"
            )
            logger.warning(msg)
            send_discord(webhook_url, msg)
            return
        send_discord(webhook_url,
            f"⚠️ スクレイピング失敗 → featured_races.csv から {len(grade_races)} レースを使用")

    dates_str = " / ".join(sorted({r["race_date"] for r in grade_races}))
    send_discord(webhook_url,
        f"🏆 **今週末の重賞結果** ({dates_str})  全{len(grade_races)}レース")

    from keiba_predictor.scraper.netkeiba_scraper import scrape_race_result
    from keiba_predictor.history import (
        record_result, load_history,
        weekly_summary, cumulative_summary, hit_streak, format_summary_message,
    )
    from datetime import date as _date

    # 手動結果を読み込む
    manual_results: dict = {}
    if MANUAL_RESULTS.exists():
        try:
            manual_results = json.loads(MANUAL_RESULTS.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"manual_results.json 読み込み失敗: {e}")

    notified = 0
    for race in grade_races:
        race_id   = race["race_id"]
        race_name = race.get("race_name", race_id)
        race_date = race.get("race_date", "")

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

        msg = _fmt_result(race_name, race_date, actual_df, pred, payouts, manual=manual)
        if send_discord(webhook_url, msg):
            notified += 1
            logger.info(f"  送信: {race_name}")

        # 的中実績を CSV に記録
        if manual and "fukusho_hit" in manual:
            # manual_results.json の的中フラグで直接記録
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

    send_discord(webhook_url, f"✅ {notified}/{len(grade_races)} レース結果送信完了")

    # 週次・累計サマリーを Discord に送信
    try:
        today   = _date.today()
        hist_df = load_history()
        w_stats = weekly_summary(hist_df, today)
        c_stats = cumulative_summary(hist_df)
        streak  = hit_streak(hist_df)
        if w_stats["n_races"] > 0:
            summary_msg = format_summary_message(w_stats, c_stats, streak)
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
