"""JRA固有特徴量（クラス昇降・Elo・騎手乗り替わり）のユニットテスト。"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from keiba_predictor.features.feature_engineering import (
    add_class_features,
    add_elo_features,
    add_jockey_change_feature,
    parse_race_class_jra,
)


# ─────────────────────────────────────────────────────────────
# parse_race_class_jra
# ─────────────────────────────────────────────────────────────
class TestParseRaceClassJRA:
    def test_shinba(self):
        assert parse_race_class_jra("3歳新馬") == 0.0

    def test_mishori(self):
        assert parse_race_class_jra("3歳未勝利") == 1.0

    def test_mishori_age_prefix(self):
        assert parse_race_class_jra("4歳以上未勝利") == 1.0

    def test_1sho_class(self):
        assert parse_race_class_jra("4歳以上1勝クラス") == 2.0

    def test_2sho_class(self):
        assert parse_race_class_jra("4歳以上2勝クラス") == 3.0

    def test_3sho_parentheses(self):
        # 是政ステークス(3勝) → 4
        assert parse_race_class_jra("是政ステークス(3勝)") == 4.0

    def test_listed(self):
        # スイートピーS(L) → 5
        assert parse_race_class_jra("スイートピーS(L)") == 5.0

    def test_g3_parentheses(self):
        assert parse_race_class_jra("東京新聞杯（G3）") == 6.0

    def test_g2_parentheses(self):
        assert parse_race_class_jra("宝塚記念（G2）") == 7.0

    def test_g1_bare(self):
        # 単体の"G1"文字列 → 8
        assert parse_race_class_jra("G1") == 8.0

    def test_g1_parentheses(self):
        assert parse_race_class_jra("ジャパンC（G1）") == 8.0

    def test_unknown_returns_nan(self):
        assert np.isnan(parse_race_class_jra("判定不能レース名"))

    def test_none_returns_nan(self):
        assert np.isnan(parse_race_class_jra(None))

    def test_int_returns_nan(self):
        assert np.isnan(parse_race_class_jra(123))

    def test_3sho_class_text(self):
        assert parse_race_class_jra("4歳以上3勝クラス") == 4.0

    def test_op_parentheses(self):
        assert parse_race_class_jra("OP特別（OP）") == 5.0


# ─────────────────────────────────────────────────────────────
# add_class_features
# ─────────────────────────────────────────────────────────────
class TestAddClassFeatures:
    def _make_df(self):
        return pd.DataFrame({
            "horse_id":  ["H001", "H001", "H002"],
            "race_id":   ["R001", "R002", "R001"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-01"]),
            "race_name": ["3歳未勝利", "4歳以上1勝クラス", "3歳未勝利"],
        })

    def test_race_class_level(self):
        df = add_class_features(self._make_df())
        assert df[df["race_id"] == "R001"]["race_class_level"].iloc[0] == 1.0

    def test_prev_class_level_first_race_nan(self):
        df = add_class_features(self._make_df())
        h1 = df[(df["horse_id"] == "H001")].sort_values("race_date")
        assert np.isnan(h1["prev_class_level"].iloc[0])

    def test_class_diff_positive_for_class_up(self):
        df = add_class_features(self._make_df())
        h1 = df[(df["horse_id"] == "H001")].sort_values("race_date")
        # 未勝利(1) → 1勝クラス(2): class_diff = +1
        assert h1["class_diff"].iloc[1] == pytest.approx(1.0)
        assert h1["is_class_up"].iloc[1] == 1.0
        assert h1["is_class_down"].iloc[1] == 0.0

    def test_missing_race_name_column(self):
        df = pd.DataFrame({"horse_id": ["H001"], "race_date": pd.to_datetime(["2024-01-01"])})
        df = add_class_features(df)
        assert "race_class_level" in df.columns
        assert np.isnan(df["race_class_level"].iloc[0])


# ─────────────────────────────────────────────────────────────
# add_elo_features
# ─────────────────────────────────────────────────────────────
class TestAddEloFeatures:
    def _make_df(self):
        return pd.DataFrame({
            "race_id":         ["R001", "R001", "R002", "R002"],
            "race_date":       pd.to_datetime(["2024-01-01"] * 2 + ["2024-01-08"] * 2),
            "horse_id":        ["H001", "H002", "H001", "H002"],
            "finish_position":  [1, 2, 2, 1],
            "course_type_enc": [0, 0, 0, 0],
            "jockey_id":       ["J001", "J002", "J001", "J002"],
        })

    def test_initial_elo_is_1500(self):
        df = add_elo_features(self._make_df())
        r1 = df[df["race_id"] == "R001"]
        assert (r1["horse_elo"] == 1500.0).all()

    def test_no_leakage(self):
        """horse_elo はレース前の値: 勝ち馬のR2 horse_elo > 1500。"""
        df = add_elo_features(self._make_df())
        r2 = df[df["race_id"] == "R002"]
        h1_elo = r2[r2["horse_id"] == "H001"]["horse_elo"].iloc[0]
        h2_elo = r2[r2["horse_id"] == "H002"]["horse_elo"].iloc[0]
        assert h1_elo > 1500.0, "R001で1位のH001はR002前にElo > 1500 のはず"
        assert h2_elo < 1500.0, "R001で2位のH002はR002前にElo < 1500 のはず"

    def test_elo_field_avg_zero_sum_in_race(self):
        """elo_minus_field_avg の合計はゼロ（同レース内）。"""
        df = add_elo_features(self._make_df())
        for rid in ["R001", "R002"]:
            total = df[df["race_id"] == rid]["elo_minus_field_avg"].sum()
            assert abs(total) < 1e-9, f"{rid}: elo_minus_field_avg の合計が 0 でない: {total}"

    def test_prev_race_opp_elo_nan_for_first_race(self):
        """初戦は prev_race_opp_elo が NaN。"""
        df = add_elo_features(self._make_df())
        r1 = df[df["race_id"] == "R001"]
        assert r1["prev_race_opp_elo"].isna().all()

    def test_prev_race_opp_elo_set_for_second_race(self):
        """2戦目は prev_race_opp_elo が NaN でない。"""
        df = add_elo_features(self._make_df())
        r2 = df[df["race_id"] == "R002"]
        assert r2["prev_race_opp_elo"].notna().all()

    def test_handicap_excluded(self):
        """障害レース(course_type_enc==2)の馬はhorse_eloがNaN。"""
        df = pd.DataFrame({
            "race_id":         ["R001", "R001"],
            "race_date":       pd.to_datetime(["2024-01-01"] * 2),
            "horse_id":        ["H001", "H002"],
            "finish_position":  [1, 2],
            "course_type_enc": [2, 2],
            "jockey_id":       ["J001", "J002"],
        })
        df = add_elo_features(df)
        assert df["horse_elo"].isna().all()


# ─────────────────────────────────────────────────────────────
# add_jockey_change_feature
# ─────────────────────────────────────────────────────────────
class TestAddJockeyChangeFeature:
    def test_first_race_is_nan(self):
        df = pd.DataFrame({
            "horse_id":  ["H001"],
            "race_id":   ["R001"],
            "race_date": pd.to_datetime(["2024-01-01"]),
            "jockey_id": ["J001"],
        })
        df = add_jockey_change_feature(df)
        assert np.isnan(df["is_jockey_change"].iloc[0])

    def test_same_jockey_is_zero(self):
        df = pd.DataFrame({
            "horse_id":  ["H001", "H001"],
            "race_id":   ["R001", "R002"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "jockey_id": ["J001", "J001"],
        })
        df = add_jockey_change_feature(df)
        assert df["is_jockey_change"].iloc[1] == 0.0

    def test_different_jockey_is_one(self):
        df = pd.DataFrame({
            "horse_id":  ["H001", "H001"],
            "race_id":   ["R001", "R002"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "jockey_id": ["J001", "J002"],
        })
        df = add_jockey_change_feature(df)
        assert df["is_jockey_change"].iloc[1] == 1.0

    def test_multiple_horses_independent(self):
        df = pd.DataFrame({
            "horse_id":  ["H001", "H001", "H002", "H002"],
            "race_id":   ["R001", "R002", "R001", "R002"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-01-08"] * 2),
            "jockey_id": ["J001", "J002", "J003", "J003"],
        })
        df = add_jockey_change_feature(df)
        h1 = df[df["horse_id"] == "H001"].sort_values("race_date")
        h2 = df[df["horse_id"] == "H002"].sort_values("race_date")
        assert np.isnan(h1["is_jockey_change"].iloc[0])
        assert h1["is_jockey_change"].iloc[1] == 1.0
        assert np.isnan(h2["is_jockey_change"].iloc[0])
        assert h2["is_jockey_change"].iloc[1] == 0.0

    def test_missing_jockey_id_column(self):
        df = pd.DataFrame({
            "horse_id":  ["H001"],
            "race_date": pd.to_datetime(["2024-01-01"]),
        })
        df = add_jockey_change_feature(df)
        assert "is_jockey_change" in df.columns
        assert np.isnan(df["is_jockey_change"].iloc[0])
