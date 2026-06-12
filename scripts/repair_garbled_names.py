"""
results_history.csv の文字化け馬名を修復するスクリプト。

化けた行（name系カラムに U+FFFD を含む行）の race_id で結果ページを
再取得し、馬番→馬名の対応で pred1-3_name / actual1-3_name を引き直す。
的中判定は馬番ベースなので名前カラムのみ更新する。

使い方:
    python scripts/repair_garbled_names.py            # dry-run（確認のみ）
    python scripts/repair_garbled_names.py --apply    # CSV保存
    python scripts/repair_garbled_names.py --csv path/to/results_history.csv --apply
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# プロジェクトルートを sys.path に追加
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from keiba_predictor.scraper.netkeiba_scraper import (
    HEADERS,
    NAR_RESULT_URL,
    RACE_RESULT_URL,
    _get,
    _best_encoding_decode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPLACEMENT = "�"
NAME_COLS = ["pred1_name", "pred2_name", "pred3_name",
             "actual1_name", "actual2_name", "actual3_name"]
NUM_COLS  = ["pred1_num", "pred2_num", "pred3_num",
             "actual1_num", "actual2_num", "actual3_num"]
DEFAULT_CSV = ROOT / "keiba_predictor" / "data" / "results_history.csv"


def _is_garbled(val) -> bool:
    return isinstance(val, str) and REPLACEMENT in val


def _fetch_horse_map(race_id: str, session: requests.Session) -> dict[str, str]:
    """race_id の結果ページから {馬番: 馬名} を返す。取得失敗時は空辞書。"""
    # NAR かどうかを race_id 長さで判定（どちらも12桁だが league 列がないためURLで試みる）
    # まず JRA URL で試み、テーブルが空なら NAR URL にフォールバック
    from bs4 import BeautifulSoup

    def _parse_table(soup) -> dict[str, str]:
        if soup is None:
            return {}
        for selector in [
            "table.race_table_01",
            "table.ResultMain",
            "table.RaceTable01",
            "table.nk_tb_common",
        ]:
            tbl = soup.select_one(selector)
            if tbl is None:
                continue
            rows = [tr for tr in tbl.select("tr") if tr.select("td")]
            if not rows:
                continue
            # ヘッダから馬番・馬名の列インデックスを特定
            header_row = tbl.select_one("thead tr") or tbl.select_one("tr")
            headers = [th.get_text(strip=True) for th in header_row.select("th")]
            if not headers:
                headers = [td.get_text(strip=True) for td in header_row.select("td")]
            num_idx  = next((i for i, h in enumerate(headers) if "馬番" in h), 2)
            name_idx = next((i for i, h in enumerate(headers) if "馬名" in h), 3)
            horse_map: dict[str, str] = {}
            for tr in rows:
                tds = tr.select("td")
                if max(num_idx, name_idx) >= len(tds):
                    continue
                num  = tds[num_idx].get_text(strip=True)
                name_el = tds[name_idx].select_one("a")
                name = name_el.get_text(strip=True) if name_el else tds[name_idx].get_text(strip=True)
                if num and name:
                    horse_map[num] = name
            if horse_map:
                return horse_map
        return {}

    # JRA db.netkeiba.com
    url_jra = RACE_RESULT_URL.format(race_id=race_id)
    soup = _get(url_jra, session)
    horse_map = _parse_table(soup)
    if horse_map:
        logger.info(f"  JRA結果取得成功: {race_id} ({len(horse_map)}頭)")
        return horse_map

    # NAR nar.netkeiba.com
    nar_url = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"
    soup = _get(nar_url, session, encoding="euc-jp")
    horse_map = _parse_table(soup)
    if horse_map:
        logger.info(f"  NAR結果取得成功: {race_id} ({len(horse_map)}頭)")
        return horse_map

    logger.warning(f"  馬番マップ取得失敗: {race_id}")
    return {}


def repair(csv_path: Path, apply: bool) -> None:
    df = pd.read_csv(csv_path, dtype=str)

    # 化けている行を特定
    garbled_mask = df[NAME_COLS].apply(
        lambda col: col.map(_is_garbled)
    ).any(axis=1)
    garbled_ids = df.loc[garbled_mask, "race_id"].dropna().unique().tolist()

    if not garbled_ids:
        logger.info("文字化け行なし — 修復不要")
        return

    logger.info(f"文字化け行: {garbled_mask.sum()} 行 / {len(garbled_ids)} race_id")

    session = requests.Session()
    updated_rows = 0

    for race_id in garbled_ids:
        logger.info(f"再取得: {race_id}")
        horse_map = _fetch_horse_map(race_id, session)
        if not horse_map:
            logger.warning(f"  スキップ（馬番マップ取得不可）: {race_id}")
            time.sleep(2)
            continue

        rows_for_race = df.index[df["race_id"] == race_id].tolist()
        for idx in rows_for_race:
            changed = False
            for name_col, num_col in zip(NAME_COLS, NUM_COLS):
                name_val = df.at[idx, name_col]
                num_val  = str(df.at[idx, num_col]).strip() if pd.notna(df.at[idx, num_col]) else ""
                if _is_garbled(name_val) and num_val in horse_map:
                    new_name = horse_map[num_val]
                    logger.info(
                        f"  [{race_id}] row {idx}: {name_col} '{name_val}' → '{new_name}' (馬番{num_val})"
                    )
                    if apply:
                        df.at[idx, name_col] = new_name
                    changed = True
            if changed:
                updated_rows += 1

        time.sleep(2)

    logger.info(f"修復対象行数: {updated_rows}")

    if apply:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"保存完了: {csv_path}")
    else:
        logger.info("dry-run モード — CSVは変更しません（--apply で保存）")


def main():
    parser = argparse.ArgumentParser(description="results_history.csv の文字化け馬名を修復する")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="対象CSVパス")
    parser.add_argument("--apply", action="store_true", help="実際にCSVを上書き保存する")
    args = parser.parse_args()

    if not args.csv.exists():
        logger.error(f"CSVが見つかりません: {args.csv}")
        sys.exit(1)

    repair(args.csv, apply=args.apply)


if __name__ == "__main__":
    main()
