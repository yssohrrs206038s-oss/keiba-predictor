"""
競馬予想システム メインエントリポイント

使い方:
    # 1. データ収集（直近1ヶ月）
    python -m keiba_predictor.main scrape --year 2024 --month 1

    # 2. データ収集（期間指定）
    python -m keiba_predictor.main scrape --start 2023-01 --end 2023-12

    # 3. データクリーニング
    python -m keiba_predictor.main clean

    # 4. 特徴量エンジニアリング
    python -m keiba_predictor.main features

    # 5. モデル学習
    python -m keiba_predictor.main train

    # 6. 特定レースを予測
    python -m keiba_predictor.main predict --race-id 202305050811

    # 6b. 出馬表からリアルタイム予測（CSV不要）
    python -m keiba_predictor.main predict --race-id 202305050811 --live

    # 6c. リアルタイム予測して Discord にも送信
    python -m keiba_predictor.main predict --race-id 202305050811 --live --notify --webhook-url https://discord.com/api/webhooks/...

    # 7. 全ステップを一括実行
    python -m keiba_predictor.main all --start 2023-01 --end 2023-12

    # 8. 週次レポート生成（note 用 Markdown）
    python -m keiba_predictor.main report --week 2026-03-29
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_scrape(args: argparse.Namespace) -> None:
    from keiba_predictor.scraper.netkeiba_scraper import scrape_races, scrape_nar_races

    if args.start and args.end:
        sy, sm = map(int, args.start.split("-"))
        ey, em = map(int, args.end.split("-"))
    else:
        today = datetime.today()
        sy, sm = today.year, today.month
        ey, em = today.year, today.month

    logger.info(f"スクレイピング開始: {sy}-{sm:02d} ~ {ey}-{em:02d}")
    df = scrape_races(sy, sm, ey, em)
    logger.info(f"JRA 取得完了: {len(df)} rows")

    if getattr(args, "nar", False):
        logger.info(f"[NAR] 地方競馬スクレイピング開始: {sy}-{sm:02d} ~ {ey}-{em:02d}")
        nar_df = scrape_nar_races(sy, sm, ey, em)
        logger.info(f"NAR 取得完了: {len(nar_df)} rows")


def cmd_clean(args: argparse.Namespace) -> None:
    from keiba_predictor.scraper.data_cleaner import load_and_clean
    df = load_and_clean()
    logger.info(f"クリーニング完了: {len(df)} rows")


def cmd_features(args: argparse.Namespace) -> None:
    from keiba_predictor.features.feature_engineering import load_and_build
    df = load_and_build()
    logger.info(f"特徴量生成完了: {len(df)} rows")


def cmd_train(args: argparse.Namespace) -> None:
    from keiba_predictor.model.train import train
    train(n_splits=args.cv_splits, league=getattr(args, "league", "jra"))


def cmd_tune(args: argparse.Namespace) -> None:
    import json
    import pandas as pd
    from keiba_predictor.model.train import (
        tune_hyperparameters, BEST_PARAMS_PATH, DATA_DIR,
    )
    from keiba_predictor.features.feature_engineering import FEATURE_COLS

    featured_path = DATA_DIR / "featured_races.csv"
    df = pd.read_csv(featured_path, encoding="utf-8-sig", parse_dates=["race_date"])
    df = df.dropna(subset=["top3"]).reset_index(drop=True)
    df = df.sort_values("race_date").reset_index(drop=True)

    league = getattr(args, "league", "jra")
    if league.lower() != "all" and "league" in df.columns:
        df = df[df["league"].str.upper() == league.upper()].reset_index(drop=True)

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    best_params = tune_hyperparameters(
        df, available_cols,
        n_trials=args.n_trials,
        n_splits=args.cv_splits,
    )

    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    logger.info(f"最良パラメータ保存: {BEST_PARAMS_PATH}")


def cmd_predict(args: argparse.Namespace) -> None:
    webhook = getattr(args, "webhook_url", None)
    notify  = getattr(args, "notify", False)

    if getattr(args, "live", False):
        # ── ライブモード: 出馬表をスクレイピングして予測 ──────
        from keiba_predictor.model.predict import predict_live
        predict_live(
            race_id=args.race_id,
            notify=notify,
            webhook_url=webhook,
        )
    else:
        # ── 通常モード: featured_races.csv から予測 ───────────
        from keiba_predictor.model.predict import predict_from_csv
        predict_from_csv(args.race_id, notify=notify, webhook_url=webhook)


def cmd_all(args: argparse.Namespace) -> None:
    """スクレイピング → クリーニング → 特徴量 → 学習 を一括実行"""
    cmd_scrape(args)
    cmd_clean(args)
    cmd_features(args)
    cmd_train(args)


def cmd_report(args: argparse.Namespace) -> None:
    """週次レポートを Markdown 形式で生成する。"""
    from keiba_predictor.history import build_weekly_report
    from pathlib import Path as _Path

    output = _Path(args.output) if args.output else None
    report_text = build_weekly_report(args.week, output_path=output)
    # 保存先をログに出力（print で標準出力にも）
    print(report_text[:200] + "\n…（レポート生成完了）")


def cmd_snapshot(args: argparse.Namespace) -> None:
    """predictions_cache.json のスナップショットを時間帯別で保存する。

    13時台: predictions_snapshot_13.json（本スナップショット・上書きしない）
    14時台: predictions_snapshot_14.json（最終更新用）
    """
    import json as _json
    from datetime import datetime as _dt
    from pathlib import Path as _Path

    _data_dir = _Path(__file__).parent / "data"
    src = _data_dir / "predictions_cache.json"
    if not src.exists():
        logger.error(f"キャッシュファイルが見つかりません: {src}")
        sys.exit(1)

    from datetime import timezone as _tz, timedelta as _td
    _JST = _tz(_td(hours=9))
    now = _dt.now(tz=_JST)
    hour = now.hour

    if hour < 14:
        tag = "13"
    else:
        tag = "14"

    dst = _data_dir / f"predictions_snapshot_{tag}.json"

    # 13時のスナップショットは上書きしない（後出し防止）
    if tag == "13" and dst.exists():
        logger.info(f"13時スナップショットは既に存在 → スキップ: {dst}")
        return

    # タイムスタンプを付与して保存
    with open(src, encoding="utf-8") as f:
        cache = _json.load(f)
    cache["_snapshot_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
    dst.write_text(_json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"スナップショット保存: {dst} ({cache['_snapshot_time']})")


def _cmd_note_report() -> None:
    """note記事（アナリストレポート）を生成する。"""
    from keiba_predictor.note_analyst import generate_all_reports
    paths = generate_all_reports()
    logger.info(f"note記事生成完了: {len(paths)} 記事")


def cmd_update_featured(args: argparse.Namespace) -> None:
    """翌週末の重賞レースをスクレイピングして featured_races.csv に保存する。"""
    from keiba_predictor.discord_notify import update_featured_races_csv
    count = update_featured_races_csv()
    if count == 0:
        logger.warning("重賞レースが見つかりませんでした。featured_races.csv は更新されませんでした。")
        sys.exit(1)
    logger.info(f"featured_races.csv 更新完了: {count} レース")

    # --save-cache: 今週末のレース情報をキャッシュに保存（ダッシュボード表示用）
    if getattr(args, "save_cache", False):
        from keiba_predictor.discord_notify import _save_upcoming_to_cache
        _save_upcoming_to_cache()
        logger.info("予想キャッシュにレース情報を保存完了")


def cmd_notify(args: argparse.Namespace) -> None:
    import os
    if getattr(args, "debug", False):
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("デバッグモード有効")

    # --webhook-url 未指定時は環境変数にフォールバック（GitHub Actions 対応）
    webhook = getattr(args, "webhook_url", None) or os.environ.get("DISCORD_WEBHOOK_URL")

    if not webhook:
        logger.error(
            "Discord Webhook URL が未設定です。"
            "--webhook-url オプションまたは環境変数 DISCORD_WEBHOOK_URL を設定してください。"
        )
        sys.exit(1)

    logger.info(f"Webhook URL: {'設定済み (***' + webhook[-6:] + ')' if webhook else '未設定'}")

    from keiba_predictor.discord_notify import run_predict_notify, run_result_notify
    if args.mode == "predict":
        run_predict_notify(
            webhook_url=webhook,
            test_race_id=getattr(args, "test_race_id", None),
            use_live=getattr(args, "live", False),
        )
    else:
        run_result_notify(
            webhook_url=webhook,
            race_id=getattr(args, "race_id", None),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m keiba_predictor.main",
        description="netkeiba.com 競馬予想システム",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── scrape ─────────────────────────────────────────────
    p_scrape = sub.add_parser("scrape", help="レースデータをスクレイピング")
    p_scrape.add_argument(
        "--start", metavar="YYYY-MM",
        help="取得開始年月 (例: 2023-01)"
    )
    p_scrape.add_argument(
        "--end", metavar="YYYY-MM",
        help="取得終了年月 (例: 2023-12)"
    )
    p_scrape.add_argument(
        "--year", type=int,
        help="単月取得時の年（--monthと組み合わせる）"
    )
    p_scrape.add_argument(
        "--month", type=int,
        help="単月取得時の月"
    )
    p_scrape.add_argument(
        "--nar", action="store_true",
        help="地方競馬（NAR）のデータも追加取得する"
    )
    p_scrape.set_defaults(func=cmd_scrape)

    # ── clean ──────────────────────────────────────────────
    p_clean = sub.add_parser("clean", help="データクリーニング")
    p_clean.set_defaults(func=cmd_clean)

    # ── features ───────────────────────────────────────────
    p_feat = sub.add_parser("features", help="特徴量エンジニアリング")
    p_feat.set_defaults(func=cmd_features)

    # ── train ──────────────────────────────────────────────
    p_train = sub.add_parser("train", help="モデル学習")
    p_train.add_argument(
        "--cv-splits", type=int, default=5,
        help="TimeSeriesSplit の分割数 (デフォルト: 5)"
    )
    p_train.add_argument(
        "--league", choices=["jra", "nar", "all"], default="jra",
        help="学習対象リーグ: jra=JRAのみ(デフォルト) / nar=NARのみ / all=全データ"
    )
    p_train.set_defaults(func=cmd_train)

    # ── tune ──────────────────────────────────────────────
    p_tune = sub.add_parser("tune", help="Optunaでハイパーパラメータを自動チューニング")
    p_tune.add_argument(
        "--n-trials", type=int, default=100,
        help="Optunaの試行回数 (デフォルト: 100)"
    )
    p_tune.add_argument(
        "--cv-splits", type=int, default=5,
        help="TimeSeriesSplit の分割数 (デフォルト: 5)"
    )
    p_tune.add_argument(
        "--league", choices=["jra", "nar", "all"], default="jra",
        help="対象リーグ (デフォルト: jra)"
    )
    p_tune.set_defaults(func=cmd_tune)

    # ── predict ────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="指定レースの予測")
    p_pred.add_argument(
        "--race-id", required=True,
        help="netkeibaのレースID (例: 202305050811)"
    )
    p_pred.add_argument(
        "--live", action="store_true",
        help="出馬表をリアルタイムでスクレイピングして予測（CSV不要）"
    )
    p_pred.add_argument(
        "--notify", action="store_true",
        help="予測結果を Discord に送信する"
    )
    p_pred.add_argument(
        "--webhook-url", dest="webhook_url",
        help="Discord Webhook URL（未指定時は環境変数 DISCORD_WEBHOOK_URL を使用）"
    )
    p_pred.set_defaults(func=cmd_predict)

    # ── all ────────────────────────────────────────────────
    p_all = sub.add_parser("all", help="全ステップを一括実行")
    p_all.add_argument("--start", metavar="YYYY-MM", help="取得開始年月")
    p_all.add_argument("--end", metavar="YYYY-MM", help="取得終了年月")
    p_all.add_argument("--cv-splits", type=int, default=5)
    p_all.set_defaults(func=cmd_all)

    # ── report ─────────────────────────────────────────────
    p_report = sub.add_parser("report", help="週次レポートを Markdown で生成")
    p_report.add_argument(
        "--week", required=True, metavar="YYYY-MM-DD",
        help="レポート対象週に含まれる日付 (例: 2026-03-29)",
    )
    p_report.add_argument(
        "--output", metavar="PATH",
        help="出力先ファイルパス（省略時: data/reports/report_YYYYMMDD.md）",
    )
    p_report.set_defaults(func=cmd_report)

    # ── snapshot ───────────────────────────────────────────
    p_snap = sub.add_parser("snapshot", help="predictions_cache.json のスナップショットを保存")
    p_snap.set_defaults(func=cmd_snapshot)

    # ── note_report ────────────────────────────────────────
    p_note = sub.add_parser("note_report", help="note記事（アナリストレポート）を生成")
    p_note.set_defaults(func=lambda args: _cmd_note_report())

    # ── update-featured ────────────────────────────────────
    p_uf = sub.add_parser(
        "update-featured",
        help="翌週末の重賞レースを自動取得して featured_races.csv に保存"
    )
    p_uf.add_argument(
        "--save-cache", action="store_true",
        help="レース情報を predictions_cache.json にも保存（ダッシュボード表示用）"
    )
    p_uf.set_defaults(func=cmd_update_featured)

    # ── notify ─────────────────────────────────────────────
    p_notify = sub.add_parser("notify", help="Discord 週末重賞通知")
    p_notify.add_argument(
        "--mode", choices=["predict", "result"], required=True,
        help="predict=金曜予想送信 / result=日曜結果送信"
    )
    p_notify.add_argument(
        "--webhook-url", dest="webhook_url",
        help="Discord Webhook URL（未指定時は環境変数 DISCORD_WEBHOOK_URL を使用）"
    )
    p_notify.add_argument(
        "--debug", action="store_true",
        help="デバッグログを有効化（詳細なスクレイピングログを出力）"
    )
    p_notify.add_argument(
        "--test-race-id", dest="test_race_id", metavar="RACE_ID",
        help="テスト用race_id（指定時は週末重賞検索をスキップして該当レースのみ送信）"
    )
    p_notify.add_argument(
        "--race-id", dest="race_id", metavar="RACE_ID",
        help="指定レースIDのみ結果照合する（--mode result 時に使用）"
    )
    p_notify.add_argument(
        "--live", action="store_true",
        help="出馬表をリアルタイム取得して予測（出馬表未確定時はCSVにフォールバック）"
    )
    p_notify.set_defaults(func=cmd_notify)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --year/--month を --start/--end に変換
    if hasattr(args, "year") and args.year and hasattr(args, "month") and args.month:
        args.start = f"{args.year}-{args.month:02d}"
        args.end = f"{args.year}-{args.month:02d}"

    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("中断されました")
        sys.exit(0)
    except Exception as e:
        logger.error(f"エラー: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
