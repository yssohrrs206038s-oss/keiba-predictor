"""
XGBoostによる3着以内予測モデルの学習モジュール

- TimeSeriesSplit で時系列を考慮した交差検証
- 評価指標: AUC, 複勝的中率
- Feature Importance の表示・保存
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI不要環境向け
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from keiba_predictor.features.feature_engineering import FEATURE_COLS

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
IMPORTANCE_PATH = DATA_DIR / "feature_importance.csv"
IMPORTANCE_PLOT_PATH = DATA_DIR / "feature_importance.png"

# XGBoost デフォルトハイパーパラメータ
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
    "scale_pos_weight": 2.0,  # クラス不均衡を考慮（3着以内は全体の約30%）
    "random_state": 42,
    "n_jobs": -1,
    "use_label_encoder": False,
    "verbosity": 0,
}


def fukusho_accuracy(y_true: np.ndarray, y_prob: np.ndarray, top_n: int = 3) -> float:
    """
    複勝的中率: 上位top_n頭のうち実際に3着以内に入った馬の割合。
    1レースあたりの平均を返す。

    ※この関数はレース単位の情報が必要なため、評価時はレース別に呼ぶこと。
    """
    if len(y_true) == 0:
        return 0.0
    sorted_idx = np.argsort(-y_prob)[:top_n]
    hits = y_true[sorted_idx].sum()
    return hits / top_n


def evaluate_per_race(
    df_val: pd.DataFrame,
    y_prob: np.ndarray,
    top_n: int = 3,
) -> dict:
    """
    バリデーションセットをレースごとに分割して複勝的中率を計算する。
    """
    df_val = df_val.copy()
    df_val["_prob"] = y_prob
    df_val["_top3"] = df_val["top3"].values

    results = []
    for race_id, group in df_val.groupby("race_id"):
        y_t = group["_top3"].values.astype(int)
        y_p = group["_prob"].values
        acc = fukusho_accuracy(y_t, y_p, top_n=top_n)
        results.append(acc)

    return {
        "fukusho_accuracy_mean": float(np.mean(results)) if results else 0.0,
        "fukusho_accuracy_std": float(np.std(results)) if results else 0.0,
        "n_races": len(results),
    }


def train(
    featured_path: Path | None = None,
    model_path: Path | None = None,
    params: dict | None = None,
    n_splits: int = 5,
) -> xgb.XGBClassifier:
    """
    特徴量付きCSVを読み込んで XGBoost モデルを学習する。

    Args:
        featured_path: 特徴量付きCSVのパス
        model_path:    モデル保存先
        params:        XGBoost ハイパーパラメータ（省略時はデフォルト）
        n_splits:      TimeSeriesSplit の分割数

    Returns:
        学習済み XGBClassifier
    """
    if featured_path is None:
        featured_path = DATA_DIR / "featured_races.csv"
    if model_path is None:
        model_path = MODEL_PATH
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # ── データ読み込み ───────────────────────────────────────
    df = pd.read_csv(featured_path, encoding="utf-8-sig", parse_dates=["race_date"])
    df = df.dropna(subset=["top3"]).reset_index(drop=True)
    df = df.sort_values("race_date").reset_index(drop=True)

    logger.info(f"学習データ: {len(df)} rows, 期間: {df['race_date'].min()} ~ {df['race_date'].max()}")

    # ── 特徴量・ラベル ───────────────────────────────────────
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].astype(float)
    y = df["top3"].astype(int)

    logger.info(f"使用特徴量: {available_cols}")

    # ── TimeSeriesSplit 交差検証 ─────────────────────────────
    tscv = TimeSeriesSplit(n_splits=n_splits)
    auc_scores = []
    fukusho_scores = []

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
        fukusho_stats = evaluate_per_race(df_val, y_prob, top_n=3)

        auc_scores.append(auc)
        fukusho_scores.append(fukusho_stats["fukusho_accuracy_mean"])

        logger.info(
            f"Fold {fold}: AUC={auc:.4f}, "
            f"複勝的中率={fukusho_stats['fukusho_accuracy_mean']:.4f} "
            f"(n_races={fukusho_stats['n_races']})"
        )

    logger.info(
        f"\n=== 交差検証結果 ===\n"
        f"AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}\n"
        f"複勝的中率: {np.mean(fukusho_scores):.4f} ± {np.std(fukusho_scores):.4f}"
    )

    # ── 全データで最終モデルを学習 ───────────────────────────
    logger.info("全データで最終モデルを学習中...")
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y, verbose=False)

    # ── Feature Importance ───────────────────────────────────
    importance = pd.DataFrame({
        "feature": available_cols,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(IMPORTANCE_PATH, index=False, encoding="utf-8-sig")

    # プロット
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance["feature"][::-1], importance["importance"][::-1])
        ax.set_xlabel("Importance")
        ax.set_title("XGBoost Feature Importance")
        plt.tight_layout()
        fig.savefig(IMPORTANCE_PLOT_PATH, dpi=150)
        plt.close(fig)
        logger.info(f"Feature importance plot saved: {IMPORTANCE_PLOT_PATH}")
    except Exception as e:
        logger.warning(f"プロット保存失敗: {e}")

    logger.info("\n=== Feature Importance (Top 15) ===")
    logger.info(importance.head(15).to_string(index=False))

    # ── モデル保存 ───────────────────────────────────────────
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": final_model,
                "feature_cols": available_cols,
                "cv_auc_mean": float(np.mean(auc_scores)),
                "cv_fukusho_mean": float(np.mean(fukusho_scores)),
            },
            f,
        )
    logger.info(f"モデル保存: {model_path}")

    return final_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()
