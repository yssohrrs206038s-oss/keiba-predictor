"""
JRA特徴量効果測定スクリプト。

train < 2024-07-01 / 検証 2024-07〜2025-12 の時系列分割で
新旧特徴量セットのAUCを比較し、feature_eval_report.md を生成する。

使い方:
    python -m keiba_predictor.feature_eval
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR = Path(__file__).parent / "data"
REPORT_PATH = DATA_DIR / "feature_eval_report.md"

# 旧特徴量（52列）
OLD_FEATURE_COLS = [
    "distance", "course_type_enc", "track_condition_enc", "weather_enc",
    "frame_number", "horse_number", "weight_carried", "odds", "popularity",
    "sex_enc", "age", "horse_weight", "horse_weight_diff", "last_3f",
    "avg_time_3", "avg_time_5", "avg_time_3_any", "avg_time_5_any",
    "jockey_fukusho_rate", "trainer_fukusho_rate", "dist_diff_prev",
    "days_since_last_race", "prev_finish_pos", "prev_odds",
    "prev2_finish_pos", "prev2_odds", "prev3_finish_pos",
    "prev2_last_3f", "prev3_last_3f", "finish_pos_trend",
    "horse_course_fukusho_rate", "horse_dist_fukusho_rate",
    "race_grade_enc", "jockey_horse_fukusho_rate", "horse_track_fukusho_rate",
    "running_style_enc", "pace_pressure",
    "jockey_course_fukusho_rate", "jockey_dist_fukusho_rate",
    "weeks_since_last_race", "is_fresh", "is_continuous",
    "jockey_trainer_fukusho_rate", "weight_carried_diff", "is_weight_increase",
    "same_day_rank", "prob_vs_avg",
    "sire_win_rate", "bms_win_rate", "sire_course_win_rate",
    "sire_dist_win_rate", "bms_course_win_rate",
]

NEW_FEATURE_COLS = OLD_FEATURE_COLS + [
    "race_class_level", "prev_class_level", "class_diff",
    "is_class_up", "is_class_down",
    "horse_elo", "elo_minus_field_avg", "prev_race_opp_elo",
    "is_jockey_change",
]

TRAIN_CUTOFF = pd.Timestamp("2024-07-01")
VAL_END = pd.Timestamp("2025-12-31")
TOP_N_IMPORTANCE = 20


def run_eval(featured_path: Path | None = None) -> None:
    try:
        from sklearn.metrics import roc_auc_score
        import xgboost as xgb
    except ImportError as e:
        logger.error(f"依存パッケージが不足しています: {e}")
        sys.exit(1)

    if featured_path is None:
        featured_path = DATA_DIR / "featured_races.csv"
    if not featured_path.exists():
        logger.error(
            f"特徴量CSV が見つかりません: {featured_path}\n"
            "先に `python -m keiba_predictor.main features` を実行してください。"
        )
        sys.exit(1)

    logger.info(f"読み込み: {featured_path}")
    df = pd.read_csv(featured_path, encoding="utf-8-sig", parse_dates=["race_date"])

    if "league" in df.columns:
        df = df[df["league"].str.upper() == "JRA"].reset_index(drop=True)
        logger.info(f"JRAフィルタ後: {len(df)} rows")

    df = df.sort_values("race_date").reset_index(drop=True)
    df = df.dropna(subset=["top3"])

    train = df[df["race_date"] < TRAIN_CUTOFF]
    val = df[(df["race_date"] >= TRAIN_CUTOFF) & (df["race_date"] <= VAL_END)]
    logger.info(f"Train: {len(train)} rows ({train['race_date'].min()} ~ {train['race_date'].max()})")
    logger.info(f"Val:   {len(val)} rows ({val['race_date'].min()} ~ {val['race_date'].max()})")

    if len(train) < 100 or len(val) < 10:
        # 日付範囲が足りない場合は先頭60%/後40%でフォールバック
        logger.warning(
            f"指定期間のデータが不足 (train={len(train)}, val={len(val)})。"
            "先頭60%/後40%の時系列分割にフォールバックします。"
        )
        split_idx = int(len(df) * 0.6)
        train = df.iloc[:split_idx]
        val   = df.iloc[split_idx:]
        logger.info(f"フォールバック Train: {len(train)}, Val: {len(val)}")
        if len(train) < 100 or len(val) < 10:
            logger.error("データが少なすぎます。featured_races.csv を確認してください。")
            sys.exit(1)

    xgb_params = dict(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=2.0, random_state=42,
        use_label_encoder=False, verbosity=0,
        objective="binary:logistic", eval_metric="auc",
    )

    results = {}
    for label, cols in [("old_52", OLD_FEATURE_COLS), ("new_61", NEW_FEATURE_COLS)]:
        avail = [c for c in cols if c in train.columns]
        if not avail:
            logger.warning(f"{label}: 利用可能な特徴量がありません")
            continue

        X_tr = train[avail].astype(float)
        y_tr = train["top3"].astype(int)
        X_va = val[avail].astype(float)
        y_va = val["top3"].astype(int)

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        y_prob = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, y_prob)

        imp = pd.DataFrame({
            "feature": avail,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        results[label] = {"auc": auc, "importance": imp, "n_features": len(avail)}
        logger.info(f"{label}: AUC={auc:.4f} ({len(avail)} features)")

    # Eloランク確認
    elo_rank = None
    if "new_61" in results:
        imp = results["new_61"]["importance"]
        matches = imp[imp["feature"] == "elo_minus_field_avg"]
        if not matches.empty:
            elo_rank = int(matches.index[0]) + 1

    train_range = (train["race_date"].min(), train["race_date"].max())
    val_range   = (val["race_date"].min(),   val["race_date"].max())
    _write_report(results, elo_rank, len(train), len(val), train_range, val_range)
    logger.info(f"レポート保存: {REPORT_PATH}")


def _write_report(
    results: dict,
    elo_rank: int | None,
    n_train: int,
    n_val: int,
    train_range: tuple | None = None,
    val_range: tuple | None = None,
) -> None:
    old = results.get("old_52", {})
    new = results.get("new_61", {})
    old_auc = old.get("auc", float("nan"))
    new_auc = new.get("auc", float("nan"))
    delta = new_auc - old_auc if not (np.isnan(old_auc) or np.isnan(new_auc)) else float("nan")

    def _fmt_range(r):
        if r is None:
            return "不明"
        try:
            return f"{pd.Timestamp(r[0]).date()} 〜 {pd.Timestamp(r[1]).date()}"
        except Exception:
            return "不明"

    lines = [
        "# JRA特徴量追加 効果測定レポート",
        "",
        f"- 学習期間: {_fmt_range(train_range)} ({n_train:,} rows)",
        f"- 検証期間: {_fmt_range(val_range)} ({n_val:,} rows)",
        f"- 対象: JRA のみ",
        "",
        "## AUC比較",
        "",
        "| モデル | 特徴量数 | AUC |",
        "|--------|---------|-----|",
        f"| 旧（ベースライン） | {old.get('n_features', 52)} | {old_auc:.4f} |",
        f"| 新（クラス/Elo/乗り替わり追加） | {new.get('n_features', 61)} | {new_auc:.4f} |",
        f"| **差分** | | **{delta:+.4f}** |",
        "",
    ]

    if "new_61" in results:
        imp = results["new_61"]["importance"]
        lines += [
            f"## XGBoost特徴量重要度 Top {TOP_N_IMPORTANCE}（新モデル）",
            "",
            "| Rank | Feature | Importance |",
            "|------|---------|-----------|",
        ]
        for i, row in imp.head(TOP_N_IMPORTANCE).iterrows():
            lines.append(f"| {i+1} | {row['feature']} | {row['importance']:.4f} |")

        lines += [
            "",
            "## 新特徴量の重要度順位",
            "",
        ]
        new_feats = [
            "race_class_level", "prev_class_level", "class_diff",
            "is_class_up", "is_class_down",
            "horse_elo", "elo_minus_field_avg", "prev_race_opp_elo",
            "is_jockey_change",
        ]
        for feat in new_feats:
            matches = imp[imp["feature"] == feat]
            if not matches.empty:
                rank = int(matches.index[0]) + 1
                val = matches.iloc[0]["importance"]
                lines.append(f"- `{feat}`: **{rank}位** (importance={val:.4f})")
            else:
                lines.append(f"- `{feat}`: N/A (特徴量なし)")

        if elo_rank is not None:
            lines += [
                "",
                f"> `elo_minus_field_avg` の重要度順位: **{elo_rank}位**",
                "> (NARでは4位相当だった実績あり)",
            ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_eval()
