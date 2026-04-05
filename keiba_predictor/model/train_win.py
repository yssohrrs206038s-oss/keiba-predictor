"""
WIN5用 1着予測モデルの学習モジュール

既存の3着以内予測モデル(xgb_model.pkl)とは完全に独立。
ラベル: finish_position == 1 (1着なら1, それ以外0)
保存先: keiba_predictor/model/xgb_win_model.pkl

使い方:
    python -m keiba_predictor.model.train_win
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from keiba_predictor.features.feature_engineering import FEATURE_COLS

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "xgb_win_model.pkl"
BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"

# 1着は全体の約7%（18頭立てなら1/18≒5.6%）→ scale_pos_weight高め
DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 6.0,
    "random_state": 42,
    "n_jobs": -1,
    "use_label_encoder": False,
    "verbosity": 0,
}


def win_accuracy(y_true: np.ndarray, y_prob: np.ndarray, top_n: int = 1) -> float:
    """上位top_n頭のうち実際に1着の馬が含まれる割合。"""
    if len(y_true) == 0:
        return 0.0
    sorted_idx = np.argsort(-y_prob)[:top_n]
    return float(y_true[sorted_idx].sum() > 0)


def evaluate_per_race(
    df_val: pd.DataFrame,
    y_prob: np.ndarray,
) -> dict:
    """バリデーションセットをレースごとに分割して1着的中率を計算。"""
    df_val = df_val.copy()
    df_val["_prob"] = y_prob
    df_val["_is_winner"] = df_val["is_winner"].values

    top1_hits = []
    top3_hits = []
    for _, group in df_val.groupby("race_id"):
        y_t = group["_is_winner"].values.astype(int)
        y_p = group["_prob"].values
        top1_hits.append(win_accuracy(y_t, y_p, top_n=1))
        top3_hits.append(win_accuracy(y_t, y_p, top_n=3))

    return {
        "top1_rate": float(np.mean(top1_hits)) if top1_hits else 0.0,
        "top3_rate": float(np.mean(top3_hits)) if top3_hits else 0.0,
        "n_races": len(top1_hits),
    }


def train_win(
    featured_path: Path | None = None,
    model_path: Path | None = None,
    n_splits: int = 5,
) -> xgb.XGBClassifier:
    """
    1着予測モデルを学習する。

    既存の featured_races.csv (特徴量付き) を使い、
    ラベルを top3 → is_winner (finish_position == 1) に変更して学習。
    """
    if featured_path is None:
        # 特徴量付きCSVがなければ cleaned_races.csv から生成
        featured_path = DATA_DIR / "featured_races.csv"
        if not featured_path.exists() or featured_path.stat().st_size < 10000:
            logger.info("特徴量付きCSVが見つかりません → cleaned_races.csv から生成します")
            from keiba_predictor.features.feature_engineering import load_and_build
            load_and_build()
    if model_path is None:
        model_path = MODEL_PATH

    params = DEFAULT_PARAMS.copy()

    # ── データ読み込み ───────────────────────────────────────
    logger.info(f"データ読み込み: {featured_path}")
    df = pd.read_csv(featured_path, encoding="utf-8-sig", parse_dates=["race_date"])

    # finish_position から 1着ラベルを作成
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df = df.dropna(subset=["finish_position"]).reset_index(drop=True)
    df["is_winner"] = (df["finish_position"] == 1).astype(int)
    df = df.sort_values("race_date").reset_index(drop=True)

    # JRAのみ（WIN5はJRA限定）
    if "league" in df.columns:
        before = len(df)
        df = df[df["league"].str.upper() == "JRA"].reset_index(drop=True)
        logger.info(f"JRAフィルタ: {before} → {len(df)} rows")

    n_total = len(df)
    n_winners = df["is_winner"].sum()
    logger.info(f"学習データ: {n_total} rows")
    logger.info(f"1着: {n_winners} ({n_winners/n_total*100:.1f}%)")
    logger.info(f"期間: {df['race_date'].min()} ~ {df['race_date'].max()}")

    # ── 特徴量 ───────────────────────────────────────────────
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].astype(float)
    y = df["is_winner"].astype(int)

    logger.info(f"使用特徴量: {len(available_cols)} 個")

    # ── TimeSeriesSplit 交差検証 ─────────────────────────────
    tscv = TimeSeriesSplit(n_splits=n_splits)
    auc_scores = []
    top1_scores = []
    top3_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        df_val = df.iloc[val_idx].copy()

        model_fold = xgb.XGBClassifier(**params)
        model_fold.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_prob = model_fold.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        stats = evaluate_per_race(df_val, y_prob)

        auc_scores.append(auc)
        top1_scores.append(stats["top1_rate"])
        top3_scores.append(stats["top3_rate"])

        logger.info(
            f"Fold {fold}: AUC={auc:.4f}, "
            f"1着的中率={stats['top1_rate']*100:.1f}%, "
            f"TOP3含有率={stats['top3_rate']*100:.1f}% "
            f"(n_races={stats['n_races']})"
        )

    logger.info(
        f"\n=== 交差検証結果 ===\n"
        f"AUC:        {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}\n"
        f"1着的中率:  {np.mean(top1_scores)*100:.1f}% ± {np.std(top1_scores)*100:.1f}%\n"
        f"TOP3含有率: {np.mean(top3_scores)*100:.1f}% ± {np.std(top3_scores)*100:.1f}%"
    )

    # ── 全データで最終モデルを学習 ───────────────────────────
    logger.info("全データで最終モデルを学習中...")
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y, verbose=False)

    # ── Feature Importance ────────────────────────────────────
    importance = pd.DataFrame({
        "feature": available_cols,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info("\n=== Feature Importance (Top 15) ===")
    logger.info(importance.head(15).to_string(index=False))

    # ── モデル保存 ────────────────────────────────────────────
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": final_model,
        "feature_cols": available_cols,
        "cv_auc_mean": float(np.mean(auc_scores)),
        "cv_top1_rate": float(np.mean(top1_scores)),
        "cv_top3_rate": float(np.mean(top3_scores)),
        "task": "win_prediction",
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    logger.info(f"\nモデル保存完了: {model_path}")
    logger.info(f"AUC: {np.mean(auc_scores):.4f}")
    logger.info(f"1着的中率: {np.mean(top1_scores)*100:.1f}%")
    logger.info(f"TOP3含有率: {np.mean(top3_scores)*100:.1f}%")

    return final_model


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    train_win()


if __name__ == "__main__":
    main()
