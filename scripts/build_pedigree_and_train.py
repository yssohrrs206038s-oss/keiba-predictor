"""血統DB構築 → 特徴量生成 → モデル再学習を一括実行するスクリプト"""
import sys
import os
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# プロジェクトルートに移動
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

def main():
    # Step 1: 血統スクレイピング
    logger.info("=" * 50)
    logger.info("Step 1: 血統DB構築")
    logger.info("=" * 50)
    try:
        import pandas as pd
        from keiba_predictor.scraper.pedigree_scraper import build_pedigree_db
        ids = pd.read_csv(
            "keiba_predictor/data/raw_races.csv",
            usecols=["horse_id"], dtype=str
        )["horse_id"].dropna().unique().tolist()
        build_pedigree_db(ids, "keiba_predictor/data/pedigree_db.csv")
        logger.info("血統DB構築完了")
    except Exception as e:
        logger.error(f"血統DB構築失敗: {e}")
        return

    # Step 2: 特徴量生成
    logger.info("=" * 50)
    logger.info("Step 2: 特徴量生成")
    logger.info("=" * 50)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "keiba_predictor.main", "features"],
            capture_output=True, text=True, timeout=1800
        )
        logger.info(result.stdout[-500:] if result.stdout else "(no output)")
        if result.returncode != 0:
            logger.error(f"features失敗: {result.stderr[-500:]}")
            return
        logger.info("特徴量生成完了")
    except Exception as e:
        logger.error(f"特徴量生成失敗: {e}")
        return

    # Step 3: モデル再学習
    logger.info("=" * 50)
    logger.info("Step 3: モデル再学習")
    logger.info("=" * 50)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "keiba_predictor.main", "train"],
            capture_output=True, text=True, timeout=3600
        )
        logger.info(result.stdout[-500:] if result.stdout else "(no output)")
        if result.returncode != 0:
            logger.error(f"train失敗: {result.stderr[-500:]}")
            return
        logger.info("モデル再学習完了")
    except Exception as e:
        logger.error(f"モデル再学習失敗: {e}")
        return

    logger.info("=" * 50)
    logger.info("全工程完了！")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
