"""
リアルタイム特徴量生成モジュール

出馬表データ（scrape_shutuba の返り値）+ 過去成績CSV から
予測用の特徴量 DataFrame を生成する。

過去成績がない馬はデータセット全体の中央値で補完する。
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from keiba_predictor.features.feature_engineering import FEATURE_COLS

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


# ══════════════════════════════════════════════════════════════
# 内部ヘルパー
# ══════════════════════════════════════════════════════════════

def _load_history(cleaned_path: Optional[Path] = None) -> pd.DataFrame:
    if cleaned_path is None:
        cleaned_path = DATA_DIR / "cleaned_races.csv"
    if not cleaned_path.exists():
        logger.warning(f"過去成績CSVが見つかりません: {cleaned_path} → 過去成績なしで予想を実行します")
        return pd.DataFrame()
    try:
        df = pd.read_csv(cleaned_path, encoding="utf-8-sig")
        if "race_date" in df.columns:
            df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
        else:
            logger.warning(f"cleaned_races.csv に race_date 列がありません → 過去成績なしで予想を実行します")
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.warning(f"cleaned_races.csv 読み込み失敗: {e} → 過去成績なしで予想を実行します")
        return pd.DataFrame()


def _column_medians(history: pd.DataFrame) -> dict:
    """数値列の中央値を {col: value} で返す（補完用デフォルト値）。"""
    medians: dict = {}
    for col in FEATURE_COLS:
        if col in history.columns:
            val = pd.to_numeric(history[col], errors="coerce").median()
            medians[col] = float(val) if pd.notna(val) else np.nan
        else:
            medians[col] = np.nan
    return medians


def _horse_hist_features(
    horse_hist: pd.DataFrame,
    race_date: pd.Timestamp,
    distance: int,
    course_type_enc: int,
) -> dict:
    """1頭の過去成績（レース前の全記録）から特徴量を計算する。"""
    if horse_hist.empty or "race_date" not in horse_hist.columns:
        return {}
    past = horse_hist[horse_hist["race_date"] < race_date].sort_values("race_date")
    if past.empty:
        return {}

    last = past.iloc[-1]

    feats: dict = {}

    # 前走情報
    feats["prev_finish_pos"]    = pd.to_numeric(last.get("finish_position"), errors="coerce")
    feats["prev_odds"]          = pd.to_numeric(last.get("odds"),            errors="coerce")
    days_since = (race_date - last["race_date"]).days
    feats["days_since_last_race"] = days_since
    feats["weeks_since_last_race"] = days_since / 7.0
    feats["is_fresh"] = 1.0 if days_since >= 56 else 0.0        # 中8週以上
    feats["is_continuous"] = 1.0 if days_since <= 21 else 0.0    # 中2週以下

    last_dist = pd.to_numeric(last.get("distance"), errors="coerce")
    feats["dist_diff_prev"] = float(last_dist - distance) if pd.notna(last_dist) else np.nan

    # 斤量増減
    last_wc = pd.to_numeric(last.get("weight_carried"), errors="coerce")
    feats["_prev_weight_carried"] = last_wc  # row構築時に使用

    # 前々走（2走前）
    if len(past) >= 2:
        prev2 = past.iloc[-2]
        feats["prev2_finish_pos"] = pd.to_numeric(prev2.get("finish_position"), errors="coerce")
        feats["prev2_odds"]       = pd.to_numeric(prev2.get("odds"), errors="coerce")
        feats["prev2_last_3f"]    = pd.to_numeric(prev2.get("last_3f"), errors="coerce")
    # 前3走（3走前）
    if len(past) >= 3:
        prev3 = past.iloc[-3]
        feats["prev3_finish_pos"] = pd.to_numeric(prev3.get("finish_position"), errors="coerce")
        feats["prev3_last_3f"]    = pd.to_numeric(prev3.get("last_3f"), errors="coerce")

    # 着順トレンド: (前走 - 前3走) / 2
    fp1 = feats.get("prev_finish_pos")
    fp3 = feats.get("prev3_finish_pos")
    if pd.notna(fp1) and pd.notna(fp3):
        feats["finish_pos_trend"] = (float(fp1) - float(fp3)) / 2.0

    # 平均タイム（同コース / 全コース）
    times_all  = pd.to_numeric(past["time_sec"], errors="coerce").dropna()
    same_course = past[pd.to_numeric(past.get("course_type_enc", pd.Series(dtype=float)),
                                     errors="coerce") == course_type_enc]
    times_same  = pd.to_numeric(same_course["time_sec"], errors="coerce").dropna()

    feats["avg_time_3"]     = float(times_same.tail(3).mean()) if len(times_same) >= 1 else np.nan
    feats["avg_time_5"]     = float(times_same.tail(5).mean()) if len(times_same) >= 1 else np.nan
    feats["avg_time_3_any"] = float(times_all.tail(3).mean())  if len(times_all)  >= 1 else np.nan
    feats["avg_time_5_any"] = float(times_all.tail(5).mean())  if len(times_all)  >= 1 else np.nan

    # 同コース複勝率
    top3_same = pd.to_numeric(same_course["top3"], errors="coerce").dropna()
    feats["horse_course_fukusho_rate"] = float(top3_same.mean()) if len(top3_same) >= 1 else np.nan

    # 同距離帯複勝率（400m幅ビン）
    dist_band = (distance // 400) * 400
    if "distance" in past.columns:
        same_dist = past[(pd.to_numeric(past["distance"], errors="coerce") // 400 * 400) == dist_band]
        top3_dist = pd.to_numeric(same_dist["top3"], errors="coerce").dropna()
        feats["horse_dist_fukusho_rate"] = float(top3_dist.mean()) if len(top3_dist) >= 1 else np.nan
    else:
        feats["horse_dist_fukusho_rate"] = np.nan

    return feats


def _jockey_rate(jockey_id: str, history: pd.DataFrame, race_date: pd.Timestamp) -> float:
    """騎手の直近90日複勝率を返す。"""
    if not jockey_id or history.empty or "race_date" not in history.columns:
        return np.nan
    cutoff = race_date - pd.Timedelta(days=90)
    jh = history[
        (history["jockey_id"].astype(str) == jockey_id) &
        (history["race_date"] >= cutoff) &
        (history["race_date"] <  race_date)
    ]
    top3 = pd.to_numeric(jh["top3"], errors="coerce").dropna()
    return float(top3.mean()) if len(top3) > 0 else np.nan


def _trainer_rate(trainer_id: str, history: pd.DataFrame, race_date: pd.Timestamp) -> float:
    """調教師の直近90日複勝率を返す。"""
    if not trainer_id or history.empty or "race_date" not in history.columns:
        return np.nan
    cutoff = race_date - pd.Timedelta(days=90)
    th = history[
        (history["trainer_id"].astype(str) == trainer_id) &
        (history["race_date"] >= cutoff) &
        (history["race_date"] <  race_date)
    ]
    top3 = pd.to_numeric(th["top3"], errors="coerce").dropna()
    return float(top3.mean()) if len(top3) > 0 else np.nan


def _jockey_horse_rate(
    horse_hist: pd.DataFrame,
    jockey_id: str,
    race_date: pd.Timestamp,
    fallback: float,
) -> float:
    """騎手×馬コンビの複勝率（3回未満なら騎手全体で補完）。"""
    if horse_hist.empty or not jockey_id or "race_date" not in horse_hist.columns:
        return fallback
    combo = horse_hist[
        (horse_hist["jockey_id"].astype(str) == jockey_id) &
        (horse_hist["race_date"] < race_date)
    ]
    top3 = pd.to_numeric(combo["top3"], errors="coerce").dropna()
    return float(top3.mean()) if len(top3) >= 3 else fallback


def _jockey_course_rate(
    jockey_id: str,
    history: pd.DataFrame,
    race_date: pd.Timestamp,
    venue: str,
    course_type_enc: int,
) -> float:
    """騎手×コース（venue+course_type）の過去複勝率を返す。"""
    if not jockey_id or history.empty or "race_date" not in history.columns:
        return np.nan
    if "venue" not in history.columns:
        return np.nan
    past = history[
        (history["jockey_id"].astype(str) == jockey_id) &
        (history["race_date"] < race_date) &
        (history["venue"].astype(str) == str(venue)) &
        (pd.to_numeric(history["course_type_enc"], errors="coerce") == course_type_enc)
    ]
    top3 = pd.to_numeric(past["top3"], errors="coerce").dropna()
    return float(top3.mean()) if len(top3) >= 1 else np.nan


def _jockey_dist_rate(
    jockey_id: str,
    history: pd.DataFrame,
    race_date: pd.Timestamp,
    distance: int,
) -> float:
    """騎手×距離帯の過去複勝率を返す。

    距離帯: 短距離(<1400m), マイル(1400-1800m), 中距離(1800-2200m), 長距離(>2200m)
    """
    if not jockey_id or history.empty or "race_date" not in history.columns:
        return np.nan
    if "distance" not in history.columns:
        return np.nan

    hist_dist = pd.to_numeric(history["distance"], errors="coerce")
    if distance < 1400:
        band_mask = hist_dist < 1400
    elif distance <= 1800:
        band_mask = (hist_dist >= 1400) & (hist_dist <= 1800)
    elif distance <= 2200:
        band_mask = (hist_dist > 1800) & (hist_dist <= 2200)
    else:
        band_mask = hist_dist > 2200

    past = history[
        (history["jockey_id"].astype(str) == jockey_id) &
        (history["race_date"] < race_date) &
        band_mask
    ]
    top3 = pd.to_numeric(past["top3"], errors="coerce").dropna()
    return float(top3.mean()) if len(top3) >= 1 else np.nan


def _jockey_trainer_rate(
    jockey_id: str,
    trainer_id: str,
    history: pd.DataFrame,
    race_date: pd.Timestamp,
) -> float:
    """騎手×調教師コンビの過去複勝率（5走以上）を返す。"""
    if not jockey_id or not trainer_id or history.empty or "race_date" not in history.columns:
        return np.nan
    past = history[
        (history["jockey_id"].astype(str) == jockey_id) &
        (history["trainer_id"].astype(str) == trainer_id) &
        (history["race_date"] < race_date)
    ]
    top3 = pd.to_numeric(past["top3"], errors="coerce").dropna()
    return float(top3.mean()) if len(top3) >= 5 else np.nan


def _horse_track_rate(
    horse_hist: pd.DataFrame,
    race_date: pd.Timestamp,
    track_condition_enc: int,
) -> float:
    """この馬の指定馬場状態での複勝率を返す。"""
    if horse_hist.empty or "race_date" not in horse_hist.columns:
        return np.nan
    if "track_condition_enc" not in horse_hist.columns:
        return np.nan
    past = horse_hist[
        (horse_hist["race_date"] < race_date) &
        (pd.to_numeric(horse_hist["track_condition_enc"], errors="coerce") == track_condition_enc)
    ]
    top3 = pd.to_numeric(past["top3"], errors="coerce").dropna()
    return float(top3.mean()) if len(top3) >= 1 else np.nan


# ══════════════════════════════════════════════════════════════
# 公開 API
# ══════════════════════════════════════════════════════════════

def build_live_features(
    shutuba_info: dict,
    cleaned_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    出馬表情報 + 過去成績CSV から予測用 DataFrame を生成する。

    Args:
        shutuba_info : scrape_shutuba() の返り値
        cleaned_path : 過去成績クリーニング済みCSVのパス（省略時はデフォルト）

    Returns:
        FEATURE_COLS を含む DataFrame（1行 = 1頭）
        過去成績がない馬は中央値で補完済み。
    """
    horses_df       = shutuba_info["horses"]
    race_id         = shutuba_info.get("race_id", "")
    race_name       = shutuba_info.get("race_name", "")
    race_date_str   = shutuba_info.get("race_date", "")
    distance            = int(shutuba_info.get("distance", 0))
    course_type_enc     = int(shutuba_info.get("course_type_enc", 1))
    venue               = str(shutuba_info.get("venue", ""))
    race_grade_enc      = int(shutuba_info.get("race_grade_enc", 0))
    track_condition_enc = shutuba_info.get("track_condition_enc")  # None if unknown

    try:
        race_date = pd.Timestamp(race_date_str)
    except Exception:
        race_date = pd.Timestamp(date.today())

    # 過去成績を読み込む
    history  = _load_history(cleaned_path)
    defaults = _column_medians(history) if not history.empty else {c: np.nan for c in FEATURE_COLS}

    if horses_df.empty:
        logger.warning("出馬表が空のため特徴量を生成できません")
        return pd.DataFrame()

    rows = []
    for _, h in horses_df.iterrows():
        horse_id   = str(h.get("horse_id",   ""))
        jockey_id  = str(h.get("jockey_id",  ""))
        trainer_id = str(h.get("trainer_id", ""))

        # この馬の過去成績を絞り込む
        horse_hist = (
            history[history["horse_id"].astype(str) == horse_id]
            if not history.empty and horse_id
            else pd.DataFrame()
        )

        # 過去成績由来の特徴量
        hist_feats = _horse_hist_features(horse_hist, race_date, distance, course_type_enc)

        # 騎手・調教師・コンビ複勝率
        jockey_rate  = _jockey_rate(jockey_id,  history, race_date)
        trainer_rate = _trainer_rate(trainer_id, history, race_date)
        combo_rate   = _jockey_horse_rate(horse_hist, jockey_id, race_date, jockey_rate)
        jockey_course_rate = _jockey_course_rate(jockey_id, history, race_date, venue, course_type_enc)
        jockey_dist_rate   = _jockey_dist_rate(jockey_id, history, race_date, distance)
        jt_rate = _jockey_trainer_rate(jockey_id, trainer_id, history, race_date)

        row: dict = {
            # メタ情報（モデル特徴量ではないが後処理で使用）
            "race_id":    race_id,
            "race_name":  race_name,
            "race_date":  race_date,
            "horse_id":   horse_id,
            "horse_name": h.get("horse_name", ""),
            "jockey_id":  jockey_id,
            "trainer_id": trainer_id,
            # 出馬表から直接取得できる特徴量
            "distance":         distance,
            "course_type_enc":  course_type_enc,
            "race_grade_enc":   race_grade_enc,
            "frame_number":     h.get("frame_number"),
            "horse_number":     h.get("horse_number"),
            "weight_carried":   h.get("weight_carried"),
            "horse_weight":     h.get("horse_weight"),
            "horse_weight_diff":h.get("horse_weight_diff"),
            "sex_enc":          h.get("sex_enc", 0),
            "age":              h.get("age"),
            "odds":             h.get("odds"),
            "popularity":       h.get("popularity"),
            # レース当日情報
            "track_condition_enc": track_condition_enc if track_condition_enc is not None else np.nan,
            "weather_enc":         np.nan,
            "last_3f":             np.nan,
            # 過去成績由来
            **hist_feats,
            "jockey_fukusho_rate":      jockey_rate,
            "trainer_fukusho_rate":     trainer_rate,
            "jockey_horse_fukusho_rate": combo_rate,
            "horse_track_fukusho_rate": (
                _horse_track_rate(horse_hist, race_date, track_condition_enc)
                if track_condition_enc is not None else np.nan
            ),
            "running_style_enc": h.get("running_style_enc"),
            "jockey_course_fukusho_rate": jockey_course_rate,
            "jockey_dist_fukusho_rate":   jockey_dist_rate,
            "jockey_trainer_fukusho_rate": jt_rate,
        }

        # 斤量増減
        cur_wc = pd.to_numeric(h.get("weight_carried"), errors="coerce")
        prev_wc = hist_feats.get("_prev_weight_carried")
        if pd.notna(cur_wc) and pd.notna(prev_wc):
            row["weight_carried_diff"] = float(cur_wc - prev_wc)
            row["is_weight_increase"] = 1.0 if cur_wc > prev_wc else 0.0
        # _prev_weight_carried は内部用なので削除
        row.pop("_prev_weight_carried", None)

        # FEATURE_COLS に含まれる列が NaN なら中央値で補完
        for col in FEATURE_COLS:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                row[col] = defaults.get(col, np.nan)

        rows.append(row)

    result_df = pd.DataFrame(rows)

    # 展開圧力（同レース内の逃げ+先行馬の数）
    if "running_style_enc" in result_df.columns:
        rs = pd.to_numeric(result_df["running_style_enc"], errors="coerce")
        result_df["pace_pressure"] = float((rs <= 1).sum())
    else:
        result_df["pace_pressure"] = np.nan

    # FEATURE_COLS の列を数値型に統一
    for col in FEATURE_COLS:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

    logger.info(f"ライブ特徴量生成完了: {len(result_df)}頭 / race_id={race_id}")
    return result_df
