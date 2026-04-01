"""
AUC自動改善ループ

毎週火曜深夜に実行し、以下の改善案をランダムに試行して
AUCが向上した場合のみモデルを更新する。

改善案:
1. 特徴量の重要度下位5つを削除して再学習
2. Optunaで50試行のハイパーパラメータ探索
3. scale_pos_weightの調整
4. n_estimatorsの増加（+200）
"""

import json
import logging
import os
import pickle
import random
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"
DATA_DIR = Path(__file__).parent.parent / "data"
FEATURED_PATH = DATA_DIR / "featured_races.csv"

TARGET_AUC = 0.87
MAX_TRIALS = 5


def _load_current_auc() -> float:
    """現在のモデルのAUCを返す。"""
    if not MODEL_PATH.exists():
        return 0.0
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return float(bundle.get("cv_auc_mean", 0.0))


def _backup_model() -> Path:
    """モデルをバックアップして一時パスを返す。"""
    backup = MODEL_PATH.with_suffix(".pkl.bak")
    if MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, backup)
    return backup


def _restore_model(backup: Path) -> None:
    """バックアップからモデルを復元する。"""
    if backup.exists():
        shutil.copy2(backup, MODEL_PATH)
        backup.unlink()


def _try_drop_low_importance() -> str:
    """重要度下位5特徴量を削除して再学習。"""
    import pandas as pd
    from keiba_predictor.features.feature_engineering import FEATURE_COLS
    from keiba_predictor.model.train import train

    if not MODEL_PATH.exists():
        return "モデルなし"

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    importances = model.feature_importances_

    # 下位5特徴量を特定
    bottom_idx = np.argsort(importances)[:5]
    drop_cols = [feature_cols[i] for i in bottom_idx]
    logger.info(f"削除候補: {drop_cols}")

    # 一時的にFEATURE_COLSを書き換えて再学習
    import keiba_predictor.features.feature_engineering as fe
    original_cols = list(fe.FEATURE_COLS)
    fe.FEATURE_COLS = [c for c in original_cols if c not in drop_cols]

    try:
        train()
    finally:
        fe.FEATURE_COLS = original_cols

    return f"低重要度削除: {', '.join(drop_cols)}"


def _try_optuna_tune() -> str:
    """Optunaで50試行のハイパーパラメータ探索。"""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return "optuna未インストール"

    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    from keiba_predictor.features.feature_engineering import FEATURE_COLS

    if not FEATURED_PATH.exists():
        return "featured_races.csvなし"

    df = pd.read_csv(FEATURED_PATH, encoding="utf-8-sig", parse_dates=["race_date"])
    df = df.dropna(subset=["top3"]).sort_values("race_date").reset_index(drop=True)
    if "league" in df.columns:
        df = df[df["league"].str.upper() == "JRA"].reset_index(drop=True)

    available = [c for c in FEATURE_COLS if c in df.columns]
    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["top3"] = pd.to_numeric(df["top3"], errors="coerce")
    df = df.dropna(subset=["top3"]).reset_index(drop=True)
    X = df[available].astype(float)
    y = df["top3"].astype(int)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 1000, step=50),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 15),
            "gamma": trial.suggest_float("gamma", 0, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.5, 3.0),
        }
        tscv = TimeSeriesSplit(n_splits=3)
        aucs = []
        for train_idx, val_idx in tscv.split(X):
            m = xgb.XGBClassifier(
                **params, random_state=42, n_jobs=-1, verbosity=0,
                use_label_encoder=False, eval_metric="auc",
            )
            m.fit(X.iloc[train_idx], y.iloc[train_idx],
                  eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], verbose=False)
            pred = m.predict_proba(X.iloc[val_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[val_idx], pred))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    best = study.best_params
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2), encoding="utf-8")
    logger.info(f"Optuna best: AUC={study.best_value:.4f} params={best}")

    # ベストパラメータで再学習
    from keiba_predictor.model.train import train
    train()

    return f"Optuna50試行: best_auc={study.best_value:.4f}"


def _try_increase_estimators() -> str:
    """n_estimatorsを+200して再学習。"""
    if BEST_PARAMS_PATH.exists():
        params = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    else:
        params = {}

    current = params.get("n_estimators", 500)
    params["n_estimators"] = current + 200
    BEST_PARAMS_PATH.write_text(json.dumps(params, indent=2), encoding="utf-8")
    logger.info(f"n_estimators: {current} → {params['n_estimators']}")

    from keiba_predictor.model.train import train
    train()

    return f"n_estimators増加: {current}→{params['n_estimators']}"


def _try_adjust_pos_weight() -> str:
    """scale_pos_weightを微調整して再学習。"""
    if BEST_PARAMS_PATH.exists():
        params = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    else:
        params = {}

    current = params.get("scale_pos_weight", 2.0)
    new_val = current + random.uniform(-0.3, 0.3)
    new_val = max(1.0, min(4.0, new_val))
    params["scale_pos_weight"] = round(new_val, 2)
    BEST_PARAMS_PATH.write_text(json.dumps(params, indent=2), encoding="utf-8")
    logger.info(f"scale_pos_weight: {current} → {params['scale_pos_weight']}")

    from keiba_predictor.model.train import train
    train()

    return f"scale_pos_weight調整: {current}→{params['scale_pos_weight']}"


STRATEGIES = [
    ("低重要度特徴量削除", _try_drop_low_importance),
    ("Optunaチューニング", _try_optuna_tune),
    ("n_estimators増加", _try_increase_estimators),
    ("scale_pos_weight調整", _try_adjust_pos_weight),
]


def run_auto_improve() -> dict:
    """自動改善ループを実行する。

    Returns:
        {"trials": int, "initial_auc": float, "final_auc": float,
         "adopted": list[str], "log": list[str]}
    """
    initial_auc = _load_current_auc()
    current_auc = initial_auc
    adopted = []
    log_lines = []

    logger.info(f"自動改善ループ開始: 初期AUC={initial_auc:.4f} 目標={TARGET_AUC}")

    for trial in range(1, MAX_TRIALS + 1):
        if current_auc >= TARGET_AUC:
            log_lines.append(f"試行{trial}: 目標AUC達成 → 終了")
            break

        name, func = random.choice(STRATEGIES)
        logger.info(f"試行{trial}/{MAX_TRIALS}: {name}")
        log_lines.append(f"試行{trial}: {name}")

        backup = _backup_model()
        try:
            detail = func()
            new_auc = _load_current_auc()
            diff = new_auc - current_auc

            if new_auc > current_auc:
                logger.info(f"  採用: AUC {current_auc:.4f} → {new_auc:.4f} (+{diff:.4f})")
                log_lines.append(f"  → 採用 AUC +{diff:.4f} ({detail})")
                adopted.append(f"{name}: +{diff:.4f}")
                current_auc = new_auc
                if backup.exists():
                    backup.unlink()
            else:
                logger.info(f"  棄却: AUC {current_auc:.4f} → {new_auc:.4f} ({diff:+.4f})")
                log_lines.append(f"  → 棄却 AUC {diff:+.4f}")
                _restore_model(backup)

        except Exception as e:
            logger.warning(f"  エラー: {e}")
            log_lines.append(f"  → エラー: {e}")
            _restore_model(backup)

    final_auc = _load_current_auc()
    result = {
        "trials": min(trial, MAX_TRIALS) if 'trial' in dir() else 0,
        "initial_auc": initial_auc,
        "final_auc": final_auc,
        "adopted": adopted,
        "log": log_lines,
    }

    logger.info(
        f"自動改善ループ完了: {initial_auc:.4f} → {final_auc:.4f} "
        f"(差: {final_auc - initial_auc:+.4f}) 採用{len(adopted)}件"
    )
    return result


def notify_discord(result: dict) -> None:
    """結果をDiscordに通知する。"""
    import requests as req

    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not webhook:
        return

    diff = result["final_auc"] - result["initial_auc"]
    adopted_str = "\n".join(f"  - {a}" for a in result["adopted"]) if result["adopted"] else "  なし"

    msg = (
        f"🔬 **自動改善ループ完了**\n"
        f"試行回数: {result['trials']}/{MAX_TRIALS}\n"
        f"AUC: {result['initial_auc']:.4f} → {result['final_auc']:.4f} ({diff:+.4f})\n"
        f"採用した改善:\n{adopted_str}"
    )

    try:
        req.post(webhook, json={"content": msg}, timeout=15)
    except Exception as e:
        logger.warning(f"Discord通知失敗: {e}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    result = run_auto_improve()
    notify_discord(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
