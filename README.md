# 競馬予想システム (keiba-predictor)

netkeiba.com から過去レースデータをスクレイピングし、
XGBoost で「3着以内に入る馬」を予測する機械学習システムです。

## プロジェクト構成

```
keiba_predictor/
├── scraper/
│   ├── netkeiba_scraper.py   # スクレイピング
│   └── data_cleaner.py       # データクリーニング・型変換
├── features/
│   └── feature_engineering.py  # 特徴量生成
├── model/
│   ├── train.py              # モデル学習
│   ├── predict.py            # 予測・出力
│   └── xgb_model.pkl         # 学習済みモデル（学習後生成）
├── data/                     # CSVデータ保存先
│   ├── raw_races.csv         # スクレイピング生データ
│   ├── cleaned_races.csv     # クリーニング済み
│   ├── featured_races.csv    # 特徴量付き
│   └── feature_importance.csv
└── main.py                   # CLIエントリポイント
```

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd keiba-predictor
```

### 2. Python 仮想環境の作成

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

## 使い方

すべてのコマンドはプロジェクトルートから実行します。

### ステップ 1: データ収集

```bash
# 直近1ヶ月
python -m keiba_predictor.main scrape --year 2024 --month 1

# 期間指定（例: 2023年1月〜12月）
python -m keiba_predictor.main scrape --start 2023-01 --end 2023-12
```

> **注意**: netkeiba.com への過度なアクセスはご遠慮ください。
> 本システムはリクエスト間に 1〜2 秒の sleep を入れています。
> まず直近1〜3ヶ月分から試すことをお勧めします。

### ステップ 2: データクリーニング

```bash
python -m keiba_predictor.main clean
```

### ステップ 3: 特徴量エンジニアリング

```bash
python -m keiba_predictor.main features
```

### ステップ 4: モデル学習

```bash
python -m keiba_predictor.main train
```

学習後、以下が出力されます：

- TimeSeriesSplit 交差検証の AUC・複勝的中率
- Feature Importance（上位15件）
- `data/feature_importance.png`（棒グラフ）

### ステップ 5: レース予測

```bash
# race_id は netkeiba の URL から取得 (12桁)
python -m keiba_predictor.main predict --race-id 202305050811
```

**出力例:**

```
=======================================================
【予測結果】東京10R ヴィクトリアマイル
=======================================================

■ 各馬の3着以内確率
 順位  馬番  馬名                確率   人気  オッズ
--------------------------------------------------
   1位    3番  ソングライン        72.3%    2人気    3.50倍
   2位    7番  ナミュール          65.1%    1人気    2.80倍
   ...

■ 予想印
  ◎ 本命: [3番] ソングライン (72.3%)
  ○ 対抗: [7番] ナミュール (65.1%)
  △ 穴馬: [12番] ウインマリリン (41.2%)

■ 推奨買い目
  馬連 (上位3頭のボックス):
    3-7
    3-12
    7-12
  ワイド (上位3頭のボックス):
    3-7
    3-12
    7-12
  三連複:
    3-7-12
=======================================================
```

### 全ステップを一括実行

```bash
python -m keiba_predictor.main all --start 2023-01 --end 2023-12
```

## 特徴量一覧

| 特徴量 | 説明 |
|--------|------|
| `distance` | コース距離 (m) |
| `course_type_enc` | 芝=0, ダート=1, 障害=2 |
| `track_condition_enc` | 良=0, 稍重=1, 重=2, 不良=3 |
| `weather_enc` | 晴=0, 曇=1, 雨=2, 雪=3 |
| `frame_number` | 枠番 |
| `horse_number` | 馬番 |
| `weight_carried` | 斤量 (kg) |
| `odds` | 単勝オッズ |
| `popularity` | 人気順 |
| `sex_enc` | 牡=0, 牝=1, セン=2 |
| `age` | 馬齢 |
| `horse_weight` | 馬体重 (kg) |
| `horse_weight_diff` | 馬体重変化 (kg) |
| `last_3f` | 上がり3ハロン (秒) |
| `avg_time_3` | 過去3走平均タイム（同コース・同距離） |
| `avg_time_5` | 過去5走平均タイム（同コース・同距離） |
| `avg_time_3_any` | 過去3走平均タイム（全コース） |
| `avg_time_5_any` | 過去5走平均タイム（全コース） |
| `jockey_fukusho_rate` | 騎手複勝率（直近3ヶ月） |
| `trainer_fukusho_rate` | 調教師複勝率（直近3ヶ月） |
| `dist_diff_prev` | 前走との距離差 (m) |
| `days_since_last_race` | 前走からの日数 |
| `prev_finish_pos` | 前走着順 |
| `prev_odds` | 前走オッズ |

## データ量の目安

| 期間 | レース数 | 精度 |
|------|---------|------|
| 1年分 | ~3,000 | 基礎レベル |
| 3年分 | ~9,000 | 実用レベル |
| 5年以上 | 15,000+ | 高精度 |

## 精度向上のヒント

- **調教タイムの追加**: 追い切り情報を特徴量に加えることで精度向上が期待できます
- **LightGBM アンサンブル**: XGBoost と LightGBM のアンサンブルが有効です
- **Optuna チューニング**: ハイパーパラメータを自動最適化できます

## 注意事項

- netkeiba.com の利用規約・robots.txt を必ず確認してください
- 過度なアクセスはアカウント停止や法的問題につながる可能性があります
- 本システムは学習・研究目的で作成されています
- 競馬の予測は不確実であり、投資判断は自己責任でお願いします
