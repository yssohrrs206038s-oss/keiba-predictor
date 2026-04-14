"""
netkeiba.com から血統データをスクレイピングするモジュール

取得先: https://db.netkeiba.com/horse/ped/{horse_id}/
blood_table の td要素から父(sire)・母(dam)・母父(bms)を取得する。
"""

import time
import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PED_URL = "https://db.netkeiba.com/horse/ped/{horse_id}/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
DATA_DIR = Path(__file__).parent.parent / "data"


def scrape_pedigree(horse_id: str, session: requests.Session | None = None) -> dict:
    """
    単一horse_idの血統情報を取得する。

    Returns:
        dict: {"horse_id": str, "sire": str, "dam": str, "bms": str}
    """
    empty = {"horse_id": horse_id, "sire": "", "dam": "", "bms": ""}
    url = PED_URL.format(horse_id=horse_id)
    _get = session.get if session else requests.get

    try:
        resp = _get(url, headers=HEADERS, timeout=10)
        resp.encoding = "euc-jp"

        if resp.status_code != 200:
            return empty

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", class_="blood_table")
        if table is None:
            return empty

        rs16 = table.find_all("td", attrs={"rowspan": "16"})
        rs8 = table.find_all("td", attrs={"rowspan": "8"})

        def _name(td):
            a = td.find("a")
            return a.get_text(strip=True) if a else td.get_text(strip=True).split("\n")[0].strip()

        sire = _name(rs16[0]) if len(rs16) >= 1 else ""
        dam = _name(rs16[1]) if len(rs16) >= 2 else ""
        bms = _name(rs8[2]) if len(rs8) >= 3 else ""

        return {"horse_id": horse_id, "sire": sire, "dam": dam, "bms": bms}

    except Exception:
        return empty


def build_pedigree_db(
    horse_ids: list[str],
    output_csv: str | Path | None = None,
    workers: int = 5,
) -> pd.DataFrame:
    """
    複数horse_idの血統データを並列取得してCSVに保存する。

    既存CSVがあればスキップ済みのhorse_idは再取得しない。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if output_csv is None:
        output_csv = DATA_DIR / "pedigree_db.csv"
    output_csv = Path(output_csv)

    # 既存CSVを読み込んでスキップ対象を確認
    existing_ids = set()
    rows = []
    if output_csv.exists():
        existing_df = pd.read_csv(output_csv, dtype=str)
        existing_ids = set(existing_df["horse_id"].astype(str).tolist())
        rows = existing_df.to_dict("records")
        logger.info(f"既存CSV読み込み: {len(existing_ids)} 件スキップ")

    # スクレイピング対象を絞り込み
    todo = [h for h in horse_ids if str(h) not in existing_ids]
    logger.info(f"スクレイピング対象: {len(todo)} / 全{len(horse_ids)} 件 (workers={workers})")

    if not todo:
        logger.info("全horse_idがスクレイピング済みです")
        return pd.DataFrame(rows)

    session = requests.Session()
    session.headers.update(HEADERS)
    done = 0

    def _fetch(hid):
        time.sleep(0.1)  # 軽いrate limit
        return scrape_pedigree(str(hid), session=session)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, h): h for h in todo}
        for future in as_completed(futures):
            result = future.result()
            rows.append(result)
            done += 1

            if done % 500 == 0:
                logger.info(f"進捗: {done}/{len(todo)} 件完了")
                pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")

    # 最終保存
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"血統DB保存完了: {output_csv} ({len(df)} 件)")
    return df


if __name__ == "__main__":
    # テスト: 単一horse_idの血統取得
    import sys
    if len(sys.argv) > 1:
        horse_id = sys.argv[1]
    else:
        horse_id = "2021105821"  # サンディブロンド

    result = scrape_pedigree(horse_id)
    print(result)
