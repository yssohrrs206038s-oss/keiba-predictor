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


def _ensure_col(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    """
    カラムが存在すればそのSeriesを、なければデフォルト値で埋めたSeriesを返す。
    KeyError を防ぐためのヘルパー。
    """
    if col in df.columns:
        return df[col]
    logger.warning(f"カラム '{col}' が見つからないため空列として処理します")
    return pd.Series([default] * len(df), index=df.index)


def clean_raw_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    スクレイピング生データをクリーニングし、型変換・エンコードを行う。

    スクレイパーがHTMLを正しくパースできなかった場合に一部カラムが
    欠損していても KeyError にならないよう _ensure_col を使う。

    Returns:
        クリーニング済みDataFrame
    """
    df = raw_df.copy()

    # ── 実際のカラム状況をログ出力（デバッグ用） ────────────
    logger.info(f"入力カラム: {list(df.columns)}")

    # ── 数値変換 ─────────────────────────────────────────────
    df["finish_position"] = _ensure_col(df, "finish_position").apply(parse_finish_position)
    df["time_sec"]        = _ensure_col(df, "time").apply(parse_time_to_seconds)
    df["odds"]            = _ensure_col(df, "odds").apply(parse_odds)
    df["popularity"]      = pd.to_numeric(_ensure_col(df, "popularity"), errors="coerce").astype("Int64")
    df["weight_carried"]  = pd.to_numeric(_ensure_col(df, "weight_carried"), errors="coerce")
    df["horse_weight"]    = pd.to_numeric(_ensure_col(df, "horse_weight"), errors="coerce").astype("Int64")
    df["horse_weight_diff"] = pd.to_numeric(_ensure_col(df, "horse_weight_diff"), errors="coerce").astype("Int64")
    df["distance"]        = pd.to_numeric(_ensure_col(df, "distance"), errors="coerce").astype("Int64")
    df["last_3f"]         = pd.to_numeric(_ensure_col(df, "last_3f"), errors="coerce")

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
    df["track_condition_enc"] = _ensure_col(df, "track_condition").map(TRACK_CONDITION_MAP)
    df["weather_enc"]         = _ensure_col(df, "weather").map(WEATHER_MAP)
    df["course_type_enc"]     = _ensure_col(df, "course_type").map(COURSE_TYPE_MAP)

    # sex のエンコード
    sex_map = {"牡": 0, "牝": 1, "セ": 2, "騸": 2}
    df["sex_enc"] = df["sex"].map(sex_map)

    # ── 日付変換 ─────────────────────────────────────────────
    df["race_date"] = pd.to_datetime(_ensure_col(df, "race_date"), errors="coerce")


    # ── 目的変数の整合 ───────────────────────────────────────
    if "top3" not in df.columns:
        df["top3"] = np.nan
    mask = df["finish_position"].notna()
    df.loc[mask, "top3"] = (df.loc[mask, "finish_position"] <= 3).astype(int)

    # ── 不要行除去（中止・除外など着順不明） ─────────────────
    df = df[df["finish_position"].notna()].reset_index(drop=True)

    # ── 枠番・馬番の数値化 ────────────────────────────────────
    df["frame_number"] = pd.to_numeric(_ensure_col(df, "frame_number"), errors="coerce").astype("Int64")
    df["horse_number"] = pd.to_numeric(_ensure_col(df, "horse_number"), errors="coerce").astype("Int64")

    # ── league 列の補完（race_id のフォーマット差で JRA/NAR を判定） ──
    # JRA race_id: YYYY + MM(01-12) + DD + 競馬場(01-10) + レース番号
    #              → race_id[4:6] = 月（01〜12）
    # NAR race_id: YYYY + 競馬場(30-55) + MM + DD + レース番号
    #              → race_id[4:6] = 競馬場コード（30〜55）
    # 判定ルール: race_id[4:6] の数値が 13 以上なら NAR、12 以下なら JRA
    # （月は最大12、NARの競馬場コードは最小30のため重複しない）
    if "race_id" in df.columns:
        if "league" not in df.columns:
            df["league"] = pd.NA
        missing_league = df["league"].isna()
        if missing_league.any():
            pos_4_6 = df.loc[missing_league, "race_id"].astype(str).str[4:6]
            derived = pos_4_6.apply(
                lambda v: "JRA" if v.isdigit() and int(v) <= 12 else
                          "NAR" if v.isdigit() and int(v) >= 13 else pd.NA
            )
            df.loc[missing_league, "league"] = derived
            n_jra = (derived == "JRA").sum()
            n_nar = (derived == "NAR").sum()
            logger.info(f"league補完: JRA={n_jra}行, NAR={n_nar}行 (race_id[4:6]から判定)")

    # ── race_date / venue の補完・修正（league 確定後に実施） ──────────
    # JRA race_id は YYYY+競馬場+開催回+開催日目+R で日付を含まない。
    # 先頭8桁を YYYYMMDD とみなす旧処理は誤った日付を作っていたため廃止し、
    #   JRA: 開催カレンダー由来の対応表（jra_kaisai_dates.csv）
    #   NAR: race_id[6:10]（MMDD）
    # から補完する。JRA の venue は race_id[4:6] から常に上書きする。
    if "race_id" in df.columns and "league" in df.columns:
        from keiba_predictor.scraper.jra_dates import (
            load_kaisai_map, jra_date_from_race_id, nar_date_from_race_id,
            jra_venue_from_race_id,
        )
        rid = df["race_id"].astype(str)
        is_jra = df["league"].astype(str).str.upper().eq("JRA")
        is_nar = df["league"].astype(str).str.upper().eq("NAR")

        need = df["race_date"].isna() | (df["race_date"].dt.year == 1970)
        # JRA: 先頭8桁から作られた疑いのある日付も対応表で検証・修正する
        fake_jra = is_jra & (df["race_date"].dt.strftime("%Y%m%d") == rid.str[:8])
        kaisai_map = load_kaisai_map() if (is_jra & (need | fake_jra)).any() else {}

        jra_fix = is_jra & (need | fake_jra)
        if jra_fix.any():
            mapped = pd.to_datetime(rid[jra_fix].map(lambda r: jra_date_from_race_id(r, kaisai_map)),
                                    errors="coerce")
            df.loc[jra_fix, "race_date"] = mapped
            n_bad = int(mapped.isna().sum())
            logger.info(f"race_date(JRA): {int(jra_fix.sum())}行を開催日対応表で補完・修正"
                        f"（対応表に無く未確定: {n_bad}行）")

        nar_fix = is_nar & need
        if nar_fix.any():
            df.loc[nar_fix, "race_date"] = pd.to_datetime(
                rid[nar_fix].map(nar_date_from_race_id), errors="coerce")
            logger.info(f"race_date(NAR): {int(nar_fix.sum())}行を race_id[6:10] から補完")

        if is_jra.any():
            df.loc[is_jra, "venue"] = rid[is_jra].map(jra_venue_from_race_id)

        n_nat = int(df["race_date"].isna().sum())
        if n_nat:
            logger.warning(f"race_date が確定しない行: {n_nat}行（開催日対応表の更新が必要）")

    # ── league 分布を表示 ─────────────────────────────────────
    if "league" in df.columns:
        counts = df["league"].value_counts(dropna=False)
        logger.info(f"league分布: {counts.to_dict()}")

    # ── 欠損カラムの補足レポート ─────────────────────────────
    missing = [c for c in ["distance", "course_type", "weather", "track_condition", "race_date"]
               if c not in df.columns or df[c].isna().all()]
    if missing:
        logger.warning(
            f"以下のカラムが全行NaNです（スクレイパーのHTML解析を確認してください）: {missing}"
        )

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
    if "league" in df_raw.columns:
        logger.info(f"  [raw] league分布: {df_raw['league'].value_counts(dropna=False).to_dict()}")

    df_clean = clean_raw_data(df_raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"保存: {output_path}")
    return df_clean


if __name__ == "__main__":
    df = load_and_clean()
    print(df.dtypes)
    print(df.head(3))
