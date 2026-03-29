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

try:
    import shap
except ImportError:
    shap = None

from keiba_predictor.features.feature_engineering import FEATURE_COLS

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
IMPORTANCE_PATH = DATA_DIR / "feature_importance.csv"
IMPORTANCE_PLOT_PATH = DATA_DIR / "feature_importance.png"

# 距離帯の定義
DISTANCE_BANDS = {
    "sprint": (0, 1400),       # 1400m以下
    "mile":   (1401, 1800),    # 1401-1800m
    "middle": (1801, 2200),    # 1801-2200m
    "long":   (2201, 99999),   # 2201m以上
}

DISTANCE_BAND_LABELS = {
    "sprint": "短距離",
    "mile":   "マイル",
    "middle": "中距離",
    "long":   "長距離",
}


def classify_distance_band(distance: float) -> str:
    """距離(m)から距離帯名を返す。"""
    for band, (lo, hi) in DISTANCE_BANDS.items():
        if lo <= distance <= hi:
            return band
    return "middle"  # fallback

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
    league: str = "jra",
) -> xgb.XGBClassifier:
    """
    特徴量付きCSVを読み込んで XGBoost モデルを学習する。

    Args:
        featured_path: 特徴量付きCSVのパス
        model_path:    モデル保存先
        params:        XGBoost ハイパーパラメータ（省略時はデフォルト）
        n_splits:      TimeSeriesSplit の分割数
        league:        学習対象リーグ ("jra" / "nar" / "all")

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

    # ── JRA/NAR データ件数を表示 ──────────────────────────────
    if "league" in df.columns:
        counts = df["league"].value_counts()
        for lg, cnt in counts.items():
            logger.info(f"  {lg}: {cnt} rows")
        logger.info(f"  合計: {len(df)} rows")
    else:
        logger.info(f"  合計: {len(df)} rows (league列なし)")

    # ── リーグフィルタ ────────────────────────────────────────
    if league.lower() != "all" and "league" in df.columns:
        before = len(df)
        df = df[df["league"].str.upper() == league.upper()].reset_index(drop=True)
        logger.info(f"リーグフィルタ ({league.upper()}): {before} → {len(df)} rows")
    elif league.lower() != "all" and "league" not in df.columns:
        logger.warning("league列が存在しないためフィルタをスキップします")

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

    # ── SHAP値による Feature Importance ────────────────────────
    shap_values_array = None
    if shap is not None:
        try:
            logger.info("SHAP値を計算中...")
            explainer = shap.TreeExplainer(final_model)
            shap_values_array = explainer.shap_values(X)
            shap_importance = pd.DataFrame({
                "feature": available_cols,
                "importance": np.abs(shap_values_array).mean(axis=0),
            }).sort_values("importance", ascending=False)
            logger.info("SHAP値ベースの特徴量重要度を使用します")
        except Exception as e:
            logger.warning(f"SHAP値計算に失敗、XGBoost標準のimportanceを使用: {e}")
            shap_importance = None
    else:
        logger.warning("shapパッケージ未インストール: pip install shap")
        shap_importance = None

    if shap_importance is not None:
        importance = shap_importance
    else:
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
        ax.set_xlabel("Mean |SHAP value|" if shap_importance is not None else "Importance")
        ax.set_title("Feature Importance (SHAP)" if shap_importance is not None else "XGBoost Feature Importance")
        plt.tight_layout()
        fig.savefig(IMPORTANCE_PLOT_PATH, dpi=150)
        plt.close(fig)
        logger.info(f"Feature importance plot saved: {IMPORTANCE_PLOT_PATH}")
    except Exception as e:
        logger.warning(f"プロット保存失敗: {e}")

    logger.info("\n=== Feature Importance (Top 15) ===")
    logger.info(importance.head(15).to_string(index=False))

    # ── モデル保存（統合モデル） ────────────────────────────────
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": final_model,
        "feature_cols": available_cols,
        "cv_auc_mean": float(np.mean(auc_scores)),
        "cv_fukusho_mean": float(np.mean(fukusho_scores)),
    }
    if shap_values_array is not None:
        bundle["shap_values"] = shap_values_array
        logger.info("SHAP値をモデルバンドルに保存しました")
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(f"モデル保存: {model_path}")

    # ── 距離帯別モデル学習 ──────────────────────────────────────
    if "distance" in df.columns:
        logger.info("\n=== 距離帯別モデル学習 ===")
        df["_distance_band"] = pd.to_numeric(df["distance"], errors="coerce").apply(
            lambda d: classify_distance_band(d) if pd.notna(d) else None
        )
        for band, (lo, hi) in DISTANCE_BANDS.items():
            label = DISTANCE_BAND_LABELS[band]
            df_band = df[df["_distance_band"] == band].reset_index(drop=True)
            if len(df_band) < 100:
                logger.warning(f"{label}モデル: データ不足 ({len(df_band)} rows) → スキップ")
                continue

            X_band = df_band[available_cols].astype(float)
            y_band = df_band["top3"].astype(int)

            # 交差検証でAUC算出
            tscv_band = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(df_band) // 500)))
            band_auc_scores = []
            for fold, (tr_idx, va_idx) in enumerate(tscv_band.split(X_band), 1):
                X_tr_b, X_va_b = X_band.iloc[tr_idx], X_band.iloc[va_idx]
                y_tr_b, y_va_b = y_band.iloc[tr_idx], y_band.iloc[va_idx]
                m = xgb.XGBClassifier(**params)
                m.fit(X_tr_b, y_tr_b, eval_set=[(X_va_b, y_va_b)], verbose=False)
                prob_b = m.predict_proba(X_va_b)[:, 1]
                band_auc_scores.append(roc_auc_score(y_va_b, prob_b))

            band_auc_mean = float(np.mean(band_auc_scores))
            logger.info(f"{label}モデル AUC: {band_auc_mean:.3f} ({len(df_band)} rows)")

            # 全データで学習・保存
            band_model = xgb.XGBClassifier(**params)
            band_model.fit(X_band, y_band, verbose=False)

            band_path = MODEL_DIR / f"xgb_model_{band}.pkl"
            with open(band_path, "wb") as f:
                pickle.dump(
                    {
                        "model": band_model,
                        "feature_cols": available_cols,
                        "cv_auc_mean": band_auc_mean,
                        "distance_band": band,
                    },
                    f,
                )
            logger.info(f"  保存: {band_path}")
        df.drop(columns=["_distance_band"], inplace=True)
    else:
        logger.warning("distance列がないため距離帯別モデルをスキップ")

    return final_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train()
