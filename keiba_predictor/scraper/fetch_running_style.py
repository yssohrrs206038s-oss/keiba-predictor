"""
既存の cleaned_races.csv に running_style_enc 列を追加する。

通過順位（passing 列）から脚質を推定する。
passing 列がない場合は、レース結果を再スクレイピングして追加する。

使い方:
    python -m keiba_predictor.scraper.fetch_running_style
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CLEANED_PATH = DATA_DIR / "cleaned_races.csv"
CACHE_PATH = DATA_DIR / "running_style_cache.json"


def _estimate_running_style(passing: str) -> float:
    """通過順位文字列（例: "1-1-1-1"）から脚質を推定する。

    Returns:
        0=逃, 1=先, 2=差, 3=追, NaN=不明
    """
    if not isinstance(passing, str) or not passing.strip():
        return float("nan")
    # 最初のコーナー通過順位を取得
    m = re.match(r"(\d+)", passing.strip())
    if not m:
        return float("nan")
    pos = int(m.group(1))
    if pos <= 2:
        return 0.0   # 逃げ
    elif pos <= 5:
        return 1.0   # 先行
    elif pos <= 10:
        return 2.0   # 差し
    else:
        return 3.0   # 追い込み


def add_running_style_from_passing(df: pd.DataFrame) -> pd.DataFrame:
    """passing 列から running_style_enc を推定して追加する。"""
    if "passing" in df.columns:
        df["running_style_enc"] = df["passing"].apply(_estimate_running_style)
        valid = df["running_style_enc"].notna().sum()
        logger.info(f"脚質推定完了: {valid}/{len(df)} 行")
    else:
        logger.warning("passing 列がありません。再スクレイピングが必要です。")
        df["running_style_enc"] = float("nan")
    return df


def fetch_passing_for_races(df: pd.DataFrame) -> pd.DataFrame:
    """passing 列がないレースに対して、結果ページから通過順位を取得する。"""
    if "passing" in df.columns and df["passing"].notna().sum() > len(df) * 0.5:
        logger.info("passing 列は十分なデータがあります。スキップ。")
        return df

    # キャッシュ読み込み
    cache: dict[str, dict] = {}
    if CACHE_PATH.exists():
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            logger.info(f"キャッシュ読み込み: {len(cache)} 件")
        except Exception:
            pass

    # race_id一覧（2024-2025年）
    if "race_id" not in df.columns:
        logger.warning("race_id 列がありません")
        return df

    race_ids = df["race_id"].astype(str).unique()
    race_ids_2y = [rid for rid in race_ids if rid[:4] in ("2024", "2025", "2026")]
    logger.info(f"対象レース数: {len(race_ids_2y)} (2024-2026年)")

    import requests
    from keiba_predictor.scraper.netkeiba_scraper import _get, HEADERS

    session = requests.Session()
    session.headers.update(HEADERS)

    processed = 0
    for rid in race_ids_2y:
        if rid in cache:
            continue

        url = f"https://db.netkeiba.com/race/{rid}/"
        try:
            soup = _get(url, session)
            if soup is None:
                continue

            table = soup.select_one("table.race_table_01")
            if table is None:
                continue

            race_passing: dict[str, str] = {}
            for tr in table.select("tr")[1:]:
                tds = tr.select("td")
                if len(tds) < 10:
                    continue
                # 馬番
                horse_num = tds[2].get_text(strip=True)
                # 通過順位 (index=9)
                passing = tds[9].get_text(strip=True)
                if horse_num and passing:
                    race_passing[horse_num] = passing

            cache[rid] = race_passing
            processed += 1

            if processed % 100 == 0:
                print(f"[進捗] {processed} レース取得完了", flush=True)
                # 中間保存
                CACHE_PATH.write_text(
                    json.dumps(cache, ensure_ascii=False), encoding="utf-8"
                )

            time.sleep(0.5)

        except Exception as e:
            logger.warning(f"  {rid} 取得失敗: {e}")
            continue

    # キャッシュ保存
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"キャッシュ保存: {len(cache)} レース ({processed} 件新規取得)")

    # DataFrame に passing 列を追加
    if "passing" not in df.columns:
        df["passing"] = ""

    for idx, row in df.iterrows():
        rid = str(row["race_id"])
        hnum = str(int(row["horse_number"])) if pd.notna(row.get("horse_number")) else ""
        race_data = cache.get(rid, {})
        if hnum in race_data:
            df.at[idx, "passing"] = race_data[hnum]

    return df


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not CLEANED_PATH.exists():
        print(f"エラー: {CLEANED_PATH} が見つかりません", file=sys.stderr)
        print("先に python -m keiba_predictor.main clean を実行してください")
        sys.exit(1)

    df = pd.read_csv(CLEANED_PATH, encoding="utf-8-sig")
    print(f"読み込み: {len(df)} 行, カラム: {list(df.columns)}")

    # passing 列がなければスクレイピングして追加
    if "passing" not in df.columns or df["passing"].isna().sum() > len(df) * 0.5:
        print("通過順位データを取得中...")
        df = fetch_passing_for_races(df)

    # running_style_enc を推定
    df = add_running_style_from_passing(df)

    # 保存
    df.to_csv(CLEANED_PATH, index=False, encoding="utf-8-sig")
    print(f"保存完了: {CLEANED_PATH}")

    # 統計
    style_counts = df["running_style_enc"].value_counts()
    style_names = {0: "逃", 1: "先", 2: "差", 3: "追"}
    print("\n脚質分布:")
    for enc, count in style_counts.items():
        name = style_names.get(int(enc), "不明")
        print(f"  {name}({int(enc)}): {count}")
    print(f"  不明(NaN): {df['running_style_enc'].isna().sum()}")


if __name__ == "__main__":
    main()
