"""
特徴量エンジニアリングモジュール

生成する特徴量:
  - 過去3走・5走の平均タイム（コース別）
  - 騎手の複勝率（直近3ヶ月）
  - 調教師の複勝率（直近3ヶ月）
  - オッズ・人気の数値化
  - 馬場状態エンコード
  - 距離適性（前走との距離差）
  - 馬体重変化量
  - 枠番・馬番
  - 性別・年齢
  - 上がり3ハロン
"""

import logging
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# 特徴量として使う列の定義
FEATURE_COLS = [
    # 基本情報
    "distance",
    "course_type_enc",
    "track_condition_enc",
    "weather_enc",
    "frame_number",
    "horse_number",
    "weight_carried",
    "odds",
    "popularity",
    "sex_enc",
    "age",
    # 馬体重
    "horse_weight",
    "horse_weight_diff",
    # 上がり3ハロン（当日）
    "last_3f",
    # 生成特徴量
    "avg_time_3",          # 過去3走平均タイム（同コース）
    "avg_time_5",          # 過去5走平均タイム（同コース）
    "avg_time_3_any",      # 過去3走平均タイム（全コース）
    "avg_time_5_any",      # 過去5走平均タイム（全コース）
    "jockey_fukusho_rate", # 騎手複勝率（直近3ヶ月）
    "trainer_fukusho_rate",# 調教師複勝率（直近3ヶ月）
    "dist_diff_prev",      # 前走との距離差
    "days_since_last_race",# 前走からの日数
    "prev_finish_pos",     # 前走着順
    "prev_odds",           # 前走オッズ
]


def _rolling_avg_time(
    df: pd.DataFrame,
    key_cols: list[str],
    n: int,
    col_name: str,
) -> pd.Series:
    """
    馬ごと（+ key_cols条件）に直近n走の平均タイムを計算する。
    当該レース自身は含めない（leakage防止）。
    """
    df = df.sort_values("race_date")

    def _horse_group_avg(group: pd.DataFrame) -> pd.Series:
        times = group["time_sec"].shift(1)  # 前走以前のみ使う
        return times.rolling(n, min_periods=1).mean()

    result = df.groupby(["horse_id"] + key_cols, group_keys=False).apply(
        _horse_group_avg
    )
    result.name = col_name
    return result


def add_past_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """過去走の平均タイム特徴量を追加する。"""
    df = df.sort_values(["horse_id", "race_date"]).reset_index(drop=True)

    # コース別（芝/ダート・距離）
    for n, col in [(3, "avg_time_3"), (5, "avg_time_5")]:
        vals = _rolling_avg_time(df, ["course_type_enc", "distance"], n, col)
        df[col] = vals.values

    # コース問わず（全体平均）
    for n, col in [(3, "avg_time_3_any"), (5, "avg_time_5_any")]:
        vals = _rolling_avg_time(df, [], n, col)
        df[col] = vals.values

    return df


def _win_rate_rolling(
    df: pd.DataFrame,
    id_col: str,
    window_days: int = 90,
) -> pd.Series:
    """
    id_col（騎手IDまたは調教師ID）ごとに直近window_days日の複勝率を計算する。
    計算基準日はそのレースの race_date。

    leakageを避けるため、当日レースは含めない。
    """
    df = df.sort_values("race_date").reset_index(drop=True)
    result = pd.Series(index=df.index, dtype=float)

    # IDごとにグループ化してインデックス一覧を取得
    grouped = df.groupby(id_col)

    for agent_id, group in grouped:
        idxs = group.index.tolist()
        dates = group["race_date"].values
        top3s = group["top3"].values

        for i, idx in enumerate(idxs):
            cutoff = dates[i]
            start = cutoff - pd.Timedelta(days=window_days)
            # 当日より前の期間
            mask = (dates[:i] >= start) & (dates[:i] < cutoff)
            recent = top3s[:i][mask]
            if len(recent) == 0:
                result[idx] = np.nan
            else:
                result[idx] = recent.sum() / len(recent)

    return result


def add_win_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """騎手・調教師の複勝率特徴量を追加する。"""
    logger.info("騎手複勝率を計算中...")
    df["jockey_fukusho_rate"] = _win_rate_rolling(df, "jockey_id", window_days=90)
    logger.info("調教師複勝率を計算中...")
    df["trainer_fukusho_rate"] = _win_rate_rolling(df, "trainer_id", window_days=90)
    return df


def add_prev_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """前走情報の特徴量を追加する。"""
    df = df.sort_values(["horse_id", "race_date"]).reset_index(drop=True)

    def _prev(group: pd.DataFrame, col: str) -> pd.Series:
        return group[col].shift(1)

    df["dist_diff_prev"] = df.groupby("horse_id", group_keys=False).apply(
        lambda g: _prev(g, "distance")
    ).values - df["distance"].values

    df["prev_finish_pos"] = df.groupby("horse_id", group_keys=False).apply(
        lambda g: _prev(g, "finish_position")
    ).values

    df["prev_odds"] = df.groupby("horse_id", group_keys=False).apply(
        lambda g: _prev(g, "odds")
    ).values

    # 前走からの経過日数
    def _days_diff(group: pd.DataFrame) -> pd.Series:
        return (group["race_date"] - group["race_date"].shift(1)).dt.days

    df["days_since_last_race"] = df.groupby("horse_id", group_keys=False).apply(
        _days_diff
    ).values

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    クリーニング済みDataFrameにすべての特徴量を追加して返す。
    """
    logger.info("特徴量エンジニアリング開始")

    df = df.copy()
    df = add_past_time_features(df)
    df = add_win_rate_features(df)
    df = add_prev_race_features(df)

    # 存在しない特徴量列を NaN で補完
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    logger.info(f"特徴量エンジニアリング完了: {len(df)} rows, {len(FEATURE_COLS)} features")
    return df


def load_and_build(
    cleaned_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    クリーニング済みCSVを読み込んで特徴量を構築し保存する。
    """
    if cleaned_path is None:
        cleaned_path = DATA_DIR / "cleaned_races.csv"
    if output_path is None:
        output_path = DATA_DIR / "featured_races.csv"

    df = pd.read_csv(cleaned_path, encoding="utf-8-sig", parse_dates=["race_date"])
    df = build_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"保存: {output_path}")
    return df


if __name__ == "__main__":
    df = load_and_build()
    print(df[FEATURE_COLS].describe())
