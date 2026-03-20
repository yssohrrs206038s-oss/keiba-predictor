"""
スクレイピングした生データのクリーニング・型変換モジュール
"""

import re
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def parse_time_to_seconds(time_str: str) -> float | None:
    """
    タイム文字列を秒数に変換する。
    例: "1:23.4" -> 83.4, "1:23" -> 83.0
    """
    if not isinstance(time_str, str) or not time_str.strip():
        return None
    time_str = time_str.strip()
    m = re.match(r"(\d+):(\d+)\.(\d+)", time_str)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 10
    m = re.match(r"(\d+):(\d+)", time_str)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    try:
        return float(time_str)
    except ValueError:
        return None


def parse_odds(odds_str: str) -> float | None:
    """オッズ文字列を浮動小数点に変換する。"""
    if not isinstance(odds_str, str):
        try:
            return float(odds_str)
        except (TypeError, ValueError):
            return None
    odds_str = odds_str.strip().replace(",", "")
    try:
        return float(odds_str)
    except ValueError:
        return None


def parse_finish_position(pos_str: str) -> int | None:
    """
    着順文字列を整数に変換する。
    除外・中止などの場合はNoneを返す。
    """
    if not isinstance(pos_str, str):
        try:
            return int(pos_str)
        except (TypeError, ValueError):
            return None
    pos_str = pos_str.strip()
    try:
        return int(pos_str)
    except ValueError:
        return None  # "除", "中", "失" など


def parse_sex_age(sex_age_str: str) -> tuple[str, int | None]:
    """
    性齢文字列を (性別, 年齢) に分解する。
    例: "牡3" -> ("牡", 3)
    """
    if not isinstance(sex_age_str, str):
        return ("", None)
    m = re.match(r"([牡牝セ騸])(\d+)", sex_age_str.strip())
    if m:
        return (m.group(1), int(m.group(2)))
    return (sex_age_str, None)


TRACK_CONDITION_MAP = {
    "良": 0,
    "稍重": 1,
    "重": 2,
    "不良": 3,
}

WEATHER_MAP = {
    "晴": 0,
    "曇": 1,
    "雨": 2,
    "小雨": 2,
    "雪": 3,
    "小雪": 3,
}

COURSE_TYPE_MAP = {
    "芝": 0,
    "ダート": 1,
    "障害": 2,
}


def clean_raw_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    スクレイピング生データをクリーニングし、型変換・エンコードを行う。

    Returns:
        クリーニング済みDataFrame
    """
    df = raw_df.copy()

    # ── 数値変換 ─────────────────────────────────────────────
    df["finish_position"] = df["finish_position"].apply(parse_finish_position)
    df["time_sec"] = df["time"].apply(parse_time_to_seconds)
    df["odds"] = df["odds"].apply(parse_odds)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").astype("Int64")
    df["weight_carried"] = pd.to_numeric(df["weight_carried"], errors="coerce")
    df["horse_weight"] = pd.to_numeric(df["horse_weight"], errors="coerce").astype("Int64")
    df["horse_weight_diff"] = pd.to_numeric(df["horse_weight_diff"], errors="coerce").astype("Int64")
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce").astype("Int64")
    df["last_3f"] = pd.to_numeric(df["last_3f"], errors="coerce")

    # ── 性別・年齢 ───────────────────────────────────────────
    if "sex_age" in df.columns:
        parsed = df["sex_age"].apply(parse_sex_age)
        df["sex"] = parsed.apply(lambda x: x[0])
        df["age"] = parsed.apply(lambda x: x[1])
        df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int64")
    else:
        df["sex"] = ""
        df["age"] = pd.NA

    # ── カテゴリエンコード ───────────────────────────────────
    df["track_condition_enc"] = df["track_condition"].map(TRACK_CONDITION_MAP)
    df["weather_enc"] = df["weather"].map(WEATHER_MAP)
    df["course_type_enc"] = df["course_type"].map(COURSE_TYPE_MAP)

    # sex のエンコード
    sex_map = {"牡": 0, "牝": 1, "セ": 2, "騸": 2}
    df["sex_enc"] = df["sex"].map(sex_map)

    # ── 日付変換 ─────────────────────────────────────────────
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

    # ── 目的変数の整合 ───────────────────────────────────────
    # finish_positionが確定している場合は再計算して確実にする
    mask = df["finish_position"].notna()
    df.loc[mask, "top3"] = (df.loc[mask, "finish_position"] <= 3).astype(int)

    # ── 不要行除去（中止・除外など着順不明） ─────────────────
    df = df[df["finish_position"].notna()].reset_index(drop=True)

    # ── 枠番・馬番の数値化 ────────────────────────────────────
    df["frame_number"] = pd.to_numeric(df["frame_number"], errors="coerce").astype("Int64")
    df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce").astype("Int64")

    logger.info(f"クリーニング完了: {len(df)} rows")
    return df


def load_and_clean(raw_path: Path | None = None, output_path: Path | None = None) -> pd.DataFrame:
    """
    生データCSVを読み込んでクリーニングし保存する。

    Args:
        raw_path:    生データCSVパス（デフォルト: data/raw_races.csv）
        output_path: 保存先（デフォルト: data/cleaned_races.csv）

    Returns:
        クリーニング済みDataFrame
    """
    if raw_path is None:
        raw_path = DATA_DIR / "raw_races.csv"
    if output_path is None:
        output_path = DATA_DIR / "cleaned_races.csv"

    df_raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    logger.info(f"生データ読み込み: {len(df_raw)} rows")

    df_clean = clean_raw_data(df_raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"保存: {output_path}")
    return df_clean


if __name__ == "__main__":
    df = load_and_clean()
    print(df.dtypes)
    print(df.head(3))
