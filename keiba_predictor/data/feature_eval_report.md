# JRA特徴量追加 効果測定レポート

- 学習期間: 2026-05-02 〜 2026-05-23 (3,246 rows)
- 検証期間: 2026-05-23 〜 2026-06-07 (2,164 rows)
- 対象: JRA のみ

## AUC比較

| モデル | 特徴量数 | AUC |
|--------|---------|-----|
| 旧（ベースライン） | 52 | 0.8370 |
| 新（クラス/Elo/乗り替わり追加） | 61 | 0.8417 |
| **差分** | | **+0.0047** |

## XGBoost特徴量重要度 Top 20（新モデル）

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | odds | 0.1022 |
| 2 | popularity | 0.0613 |
| 3 | horse_course_fukusho_rate | 0.0368 |
| 4 | course_type_enc | 0.0313 |
| 5 | last_3f | 0.0298 |
| 6 | horse_track_fukusho_rate | 0.0253 |
| 7 | prev_odds | 0.0247 |
| 8 | distance | 0.0240 |
| 9 | weather_enc | 0.0236 |
| 10 | bms_win_rate | 0.0233 |
| 11 | bms_course_win_rate | 0.0231 |
| 12 | prev_finish_pos | 0.0227 |
| 13 | sire_dist_win_rate | 0.0226 |
| 14 | avg_time_5_any | 0.0224 |
| 15 | race_grade_enc | 0.0222 |
| 16 | sire_win_rate | 0.0216 |
| 17 | elo_minus_field_avg | 0.0215 |
| 18 | horse_elo | 0.0214 |
| 19 | jockey_fukusho_rate | 0.0209 |
| 20 | horse_weight_diff | 0.0208 |

## 新特徴量の重要度順位

- `race_class_level`: **34位** (importance=0.0184)
- `prev_class_level`: **40位** (importance=0.0151)
- `class_diff`: **57位** (importance=0.0000)
- `is_class_up`: **59位** (importance=0.0000)
- `is_class_down`: **58位** (importance=0.0000)
- `horse_elo`: **18位** (importance=0.0214)
- `elo_minus_field_avg`: **17位** (importance=0.0215)
- `prev_race_opp_elo`: **60位** (importance=0.0000)
- `is_jockey_change`: **61位** (importance=0.0000)

> `elo_minus_field_avg` の重要度順位: **17位**
> (NARでは4位相当だった実績あり)
