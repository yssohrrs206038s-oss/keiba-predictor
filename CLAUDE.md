# KEIBA EDGE - AI競馬予想システム

## プロジェクト概要
- XGBoost × Claude AIによるJRA重賞予想システム
- AUC: 0.8638 / 47特徴量 / 247,003件の学習データ
- リポジトリ: https://github.com/yssohrrs206038s-oss/keiba-predictor
- ブランチ: claude/horse-racing-predictor-TLZas
- ダッシュボード: https://yssohrrs206038s-oss.github.io/keiba-predictor/

## スケジュール
- 水曜 09:00: データ収集（Actions自動）
- 金曜 21:00: 週末重賞予告X投稿（Actions自動）
- 土日 09:00: 平場＋重賞予想生成（Actions自動）
- 土日 13:00: オッズ更新・大口検知（Actions自動）
- 土日 14:00: Discord/X通知（Actions自動）
- 土日 14:30: 最終オッズ再計算（Actions自動）
- 土曜 18:00: 土曜結果照合（Actions自動）
- 日曜 18:00: 全結果照合・週次サマリー（Actions自動）

## 主要ファイル
- `keiba_predictor/model/xgb_model.pkl`: 学習済みモデル
- `keiba_predictor/model/best_params.json`: Optunaベストパラメータ
- `keiba_predictor/data/predictions_cache.json`: 予想キャッシュ
- `keiba_predictor/data/results_history.csv`: 結果履歴
- `keiba_predictor/data/manual_results.json`: 手動結果入力
- `docs/index.html`: ダッシュボード
- `.github/workflows/`: GitHub Actionsワークフロー

## 主要コマンド
```bash
# ローカル再学習（月1回程度）
python -m keiba_predictor.main features
python -m keiba_predictor.main train
git add keiba_predictor/model/
git commit -m "chore: モデル再学習"
git push origin claude/horse-racing-predictor-TLZas

# Optunaチューニング（夜間実行）
python -m keiba_predictor.main tune --n-trials 100

# 手動テスト
python -m keiba_predictor.main predict <race_id>
python -m keiba_predictor.main notify --mode result
```

## モデル情報
- アルゴリズム: XGBoost（距離帯別モデルあり）
- 特徴量数: 52
- 主要特徴量: オッズ・人気・脚質・馬場状態・前走成績・騎手適性・血統（父馬/母父）など
- AUC: 0.8645 / 複勝的中率: 61.1%
- 血統DB: 53,712頭（pedigree_db.csv）
- モデル再学習: ローカルPCで実行（Actionsでは行わない）

## 開発ルール
- モデル学習は必ずローカルPCで実行（Actions環境はデータ不足）
- git pushは必ずgit pull --rebaseしてから
- Discord Webhookはダミーを使わない（本番URLはSecretsに登録済み）
- X投稿はENABLE_X_POST=falseで停止中（アカウント凍結）

## 未実装リスト
- LINE公式アカウント連携
- 天気予報連携
- 競馬新聞本命集計
- WIN5予想
- 独自サイト構築

あなたの作業が完了したら、Codexが出力をレビューします。
