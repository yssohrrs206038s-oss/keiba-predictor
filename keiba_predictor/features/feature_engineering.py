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
  - [追加] 同コース・同距離帯の複勝率
  - [追加] レース格エンコード
  - [追加] 騎手×馬コンビ複勝率
"""

import logging
import re
from pathlib import Path

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
    "prev2_finish_pos",    # 前々走着順
    "prev2_odds",          # 前々走オッズ
    "prev3_finish_pos",    # 前3走着順
    "prev2_last_3f",       # 前々走上がり3F
    "prev3_last_3f",       # 前3走上がり3F
    "finish_pos_trend",    # 着順トレンド（マイナス=改善傾向）
    # [追加特徴量]
    "horse_course_fukusho_rate",  # 同コース（芝/ダート）過去複勝率
    "horse_dist_fukusho_rate",    # 同距離帯（±200m）過去複勝率
    "race_grade_enc",             # レース格 G1=6 … 未勝利=0
    "jockey_horse_fukusho_rate",  # 騎手×馬コンビ複勝率
    "horse_track_fukusho_rate",  # 馬場状態別複勝率（良/稍重/重/不良）
    "running_style_enc",         # 脚質（逃=0/先=1/差=2/追=3）
    "pace_pressure",             # 展開圧力（同レース内の逃げ+先行馬の数）
    "jockey_course_fukusho_rate",# 騎手×コース（venue+course_type）複勝率
    "jockey_dist_fukusho_rate",  # 騎手×距離帯複勝率
    "weeks_since_last_race",     # 前走からの週数
    "is_fresh",                  # 休み明けフラグ（中8週以上=1）
    "is_continuous",             # 連闘・中2週フラグ（中2週以下=1）
    "jockey_trainer_fukusho_rate",  # 騎手×調教師コンビ複勝率（5走以上）
    "weight_carried_diff",       # 前走からの斤量増減
    "is_weight_increase",        # 斤量増加フラグ（1=増加）
    "same_day_rank",             # 同レース内AI確率順位
    "prob_vs_avg",               # AI確率 - レース平均確率
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

    groupby().transform() を使うことで、常に入力と同じ長さの
    Series が返り、NaNキーの行は自動的に NaN になる。
    """
    df = df.sort_values("race_date")

    result = (
        df.groupby(["horse_id"] + key_cols, group_keys=False)["time_sec"]
        .transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    )
    result.name = col_name
    return result


def add_past_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """過去走の平均タイム特徴量を追加する。"""
    df = df.sort_values(["horse_id", "race_date"]).reset_index(drop=True)

    # コース別（芝/ダート・距離）
    for n, col in [(3, "avg_time_3"), (5, "avg_time_5")]:
        df[col] = _rolling_avg_time(df, ["course_type_enc", "distance"], n, col)

    # コース問わず（全体平均）
    for n, col in [(3, "avg_time_3_any"), (5, "avg_time_5_any")]:
        df[col] = _rolling_avg_time(df, [], n, col)

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

    # 前々走・前3走
    def _prev_n(group: pd.DataFrame, col: str, n: int) -> pd.Series:
        return group[col].shift(n)

    df["prev2_finish_pos"] = df.groupby("horse_id", group_keys=False).apply(
        lambda g: _prev_n(g, "finish_position", 2)
    ).values
    df["prev2_odds"] = df.groupby("horse_id", group_keys=False).apply(
        lambda g: _prev_n(g, "odds", 2)
    ).values
    df["prev3_finish_pos"] = df.groupby("horse_id", group_keys=False).apply(
        lambda g: _prev_n(g, "finish_position", 3)
    ).values
    df["prev2_last_3f"] = df.groupby("horse_id", group_keys=False).apply(
        lambda g: _prev_n(g, "last_3f", 2)
    ).values
    df["prev3_last_3f"] = df.groupby("horse_id", group_keys=False).apply(
        lambda g: _prev_n(g, "last_3f", 3)
    ).values

    # 着順トレンド: (前走 - 前3走) / 2  マイナスなら改善傾向
    fp1 = pd.to_numeric(df["prev_finish_pos"], errors="coerce")
    fp3 = pd.to_numeric(df["prev3_finish_pos"], errors="coerce")
    df["finish_pos_trend"] = (fp1 - fp3) / 2.0

    # 前走からの経過日数
    def _days_diff(group: pd.DataFrame) -> pd.Series:
        return (group["race_date"] - group["race_date"].shift(1)).dt.days

    df["days_since_last_race"] = df.groupby("horse_id", group_keys=False).apply(
        _days_diff
    ).values

    # 斤量増減
    if "weight_carried" in df.columns:
        prev_wc = df.groupby("horse_id", group_keys=False).apply(
            lambda g: _prev(g, "weight_carried")
        ).values
        wc = pd.to_numeric(df["weight_carried"], errors="coerce")
        prev_wc_num = pd.to_numeric(pd.Series(prev_wc), errors="coerce")
        df["weight_carried_diff"] = wc - prev_wc_num
        df["is_weight_increase"] = (df["weight_carried_diff"] > 0).astype(float)
    else:
        df["weight_carried_diff"] = np.nan
        df["is_weight_increase"] = np.nan

    # 前走間隔の派生特徴量
    days = pd.to_numeric(df["days_since_last_race"], errors="coerce")
    df["weeks_since_last_race"] = days / 7.0
    df["is_fresh"] = (days >= 56).astype(float)        # 中8週以上 = 休み明け
    df["is_continuous"] = (days <= 21).astype(float)    # 中2週以下 = 連闘・中2週

    return df


# ── レース格マッピング ────────────────────────────────────────────
# race_name に含まれる文字列でマッチングする順序に注意（上位優先）
_GRADE_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"[（(]G\s*[1Ⅰ][）)]|[（(]GI[）)]", re.I), 6),
    (re.compile(r"[（(]G\s*[2Ⅱ][）)]|[（(]GII[）)]", re.I), 5),
    (re.compile(r"[（(]G\s*[3Ⅲ][）)]|[（(]GIII[）)]", re.I), 4),
    (re.compile(r"[（(]L[）)]|オープン|（OP）|\(OP\)", re.I),  3),
    (re.compile(r"3勝クラス|1600万"),                          2),
    (re.compile(r"2勝クラス|1000万|900万"),                    1),
    # 1勝クラス・未勝利・新馬・500万 → デフォルト 0
]


def _encode_race_grade(name) -> int:
    """race_name 文字列からレース格を数値に変換する。"""
    if not isinstance(name, str):
        return 0
    for pat, val in _GRADE_PATTERNS:
        if pat.search(name):
            return val
    return 0


def add_race_grade_feature(df: pd.DataFrame) -> pd.DataFrame:
    """race_name からレース格を数値エンコードして race_grade_enc 列を追加する。

    G1=6, G2=5, G3=4, L/OP=3, 3勝=2, 2勝=1, 1勝/未勝利=0
    """
    df["race_grade_enc"] = df["race_name"].map(_encode_race_grade).astype(int)
    logger.info(
        "race_grade_enc 分布: "
        + str(df["race_grade_enc"].value_counts().sort_index().to_dict())
    )
    return df


def add_horse_course_dist_features(df: pd.DataFrame) -> pd.DataFrame:
    """馬の同コース・同距離帯（400m幅）での過去複勝率を追加する。

    距離帯は distance // 400 * 400 によるビン（±200m 相当）。
    当該レース自身は含めない（leakage防止）。
    """
    df = df.sort_values(["horse_id", "race_date"]).reset_index(drop=True)

    # 同コース（芝/ダート）複勝率
    logger.info("馬の同コース複勝率を計算中...")
    df["horse_course_fukusho_rate"] = (
        df.groupby(["horse_id", "course_type_enc"], group_keys=False)["top3"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # 同距離帯複勝率（400m幅ビン ≒ ±200m）
    logger.info("馬の同距離帯複勝率を計算中...")
    df["_dist_band"] = (df["distance"] // 400) * 400
    df["horse_dist_fukusho_rate"] = (
        df.groupby(["horse_id", "_dist_band"], group_keys=False)["top3"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df = df.drop(columns=["_dist_band"])

    # 馬場状態別複勝率
    logger.info("馬の馬場状態別複勝率を計算中...")
    if "track_condition_enc" in df.columns:
        df["horse_track_fukusho_rate"] = (
            df.groupby(["horse_id", "track_condition_enc"], group_keys=False)["top3"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
    else:
        df["horse_track_fukusho_rate"] = np.nan

    # 脚質エンコード（通過順位から推定）
    if "running_style_enc" not in df.columns:
        # cleaned_races.csv に running_style_enc がない場合、通過順位から推定
        if "passing" in df.columns:
            logger.info("通過順位から脚質を推定中...")
            def _estimate_style(passing):
                if not isinstance(passing, str):
                    return np.nan
                m = re.match(r"(\d+)", str(passing))
                if not m:
                    return np.nan
                pos = int(m.group(1))
                if pos <= 2: return 0  # 逃
                if pos <= 5: return 1  # 先
                if pos <= 10: return 2  # 差
                return 3  # 追
            df["running_style_enc"] = df["passing"].apply(_estimate_style)
        else:
            df["running_style_enc"] = np.nan

    # 展開圧力（同レース内の逃げ+先行馬の数）
    logger.info("展開圧力を計算中...")
    if "running_style_enc" in df.columns and "race_id" in df.columns:
        rs_numeric = pd.to_numeric(df["running_style_enc"], errors="coerce")
        df["pace_pressure"] = (
            (rs_numeric <= 1).astype(float)
            .groupby(df["race_id"]).transform("sum")
        )
    else:
        df["pace_pressure"] = np.nan

    return df


def add_jockey_horse_features(df: pd.DataFrame) -> pd.DataFrame:
    """騎手×馬コンビの過去複勝率を追加する。

    過去の乗り鞍が 3 回未満の場合は jockey_fukusho_rate で補完する。
    jockey_fukusho_rate が NaN のときはさらに NaN のまま残す。
    """
    logger.info("騎手×馬コンビ複勝率を計算中...")

    df = df.sort_values("race_date").reset_index(drop=True)
    result = pd.Series(np.nan, index=df.index, dtype=float)

    # (horse_id, jockey_id) ペアごとに時系列累積複勝率を計算
    for (_, _), group in df.groupby(["horse_id", "jockey_id"], sort=False):
        group = group.sort_values("race_date")
        idxs = group.index.tolist()
        top3s = pd.to_numeric(group["top3"], errors="coerce").values

        for i, idx in enumerate(idxs):
            past = top3s[:i]
            valid = past[~np.isnan(past)]
            if len(valid) >= 3:
                result[idx] = valid.mean()
            else:
                # サンプル不足 → 騎手全体複勝率で補完
                result[idx] = (
                    df.at[idx, "jockey_fukusho_rate"]
                    if "jockey_fukusho_rate" in df.columns
                    else np.nan
                )

    df["jockey_horse_fukusho_rate"] = result
    return df


def add_jockey_trainer_features(df: pd.DataFrame) -> pd.DataFrame:
    """騎手×調教師コンビの過去複勝率を追加する（最低5走以上）。"""
    logger.info("騎手×調教師コンビ複勝率を計算中...")
    df = df.sort_values("race_date").reset_index(drop=True)
    result = pd.Series(np.nan, index=df.index, dtype=float)

    for (_, _), group in df.groupby(["jockey_id", "trainer_id"], sort=False):
        group = group.sort_values("race_date")
        idxs = group.index.tolist()
        top3s = pd.to_numeric(group["top3"], errors="coerce").values

        for i, idx in enumerate(idxs):
            past = top3s[:i]
            valid = past[~np.isnan(past)]
            if len(valid) >= 5:
                result[idx] = valid.mean()

    df["jockey_trainer_fukusho_rate"] = result
    return df


def _dist_band_label(distance) -> str:
    """距離帯ラベルを返す: 短距離/マイル/中距離/長距離。"""
    d = pd.to_numeric(distance, errors="coerce")
    if pd.isna(d):
        return "unknown"
    if d < 1400:
        return "sprint"
    if d <= 1800:
        return "mile"
    if d <= 2200:
        return "middle"
    return "long"


def add_jockey_course_dist_features(df: pd.DataFrame) -> pd.DataFrame:
    """騎手×コース（venue+course_type）複勝率と騎手×距離帯複勝率を追加する。

    当該レース自身は含めない（leakage防止）。
    """
    df = df.sort_values(["jockey_id", "race_date"]).reset_index(drop=True)

    # 騎手×venue×course_type 複勝率
    logger.info("騎手×コース複勝率を計算中...")
    if "venue" in df.columns and "course_type_enc" in df.columns:
        df["jockey_course_fukusho_rate"] = (
            df.groupby(["jockey_id", "venue", "course_type_enc"], group_keys=False)["top3"]
            .transform(lambda x: x.shift(1).expanding().mean())
        )
    else:
        df["jockey_course_fukusho_rate"] = np.nan

    # 騎手×距離帯 複勝率
    logger.info("騎手×距離帯複勝率を計算中...")
    df["_jockey_dist_band"] = df["distance"].apply(_dist_band_label)
    df["jockey_dist_fukusho_rate"] = (
        df.groupby(["jockey_id", "_jockey_dist_band"], group_keys=False)["top3"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df = df.drop(columns=["_jockey_dist_band"])

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    クリーニング済みDataFrameにすべての特徴量を追加して返す。
    """
    logger.info("特徴量エンジニアリング開始")

    df = df.copy()
    df = add_past_time_features(df)
    df = add_win_rate_features(df)          # jockey_fukusho_rate を先に生成
    df = add_prev_race_features(df)
    df = add_race_grade_feature(df)
    df = add_horse_course_dist_features(df)
    df = add_jockey_horse_features(df)      # jockey_fukusho_rate に依存するため最後
    df = add_jockey_course_dist_features(df)
    df = add_jockey_trainer_features(df)

    # same_day_rank / prob_vs_avg は学習時には使えない（leakage）ため NaN で埋める
    # ライブ予測時のみ predict_race() 後に計算する
    if "same_day_rank" not in df.columns:
        df["same_day_rank"] = np.nan
    if "prob_vs_avg" not in df.columns:
        df["prob_vs_avg"] = np.nan

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
    if "league" in df.columns:
        logger.info(f"  [cleaned] league分布: {df['league'].value_counts(dropna=False).to_dict()}")
    df = build_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"保存: {output_path}")
    return df


if __name__ == "__main__":
    df = load_and_build()
    print(df[FEATURE_COLS].describe())
