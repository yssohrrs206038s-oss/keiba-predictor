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
    from keiba_predictor.scraper.netkeiba_scraper import scrape_races

    if args.start and args.end:
        sy, sm = map(int, args.start.split("-"))
        ey, em = map(int, args.end.split("-"))
    else:
        today = datetime.today()
        sy, sm = today.year, today.month
        ey, em = today.year, today.month

    logger.info(f"スクレイピング開始: {sy}-{sm:02d} ~ {ey}-{em:02d}")
    df = scrape_races(sy, sm, ey, em)
    logger.info(f"取得完了: {len(df)} rows")


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
    train(n_splits=args.cv_splits)


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
        from keiba_predictor.model.predict import predict_from_csv, format_prediction, calc_ev_and_flags
        import os
        result = predict_from_csv(args.race_id)
        if notify:
            from keiba_predictor.discord_notify import send_discord
            url = webhook or os.environ.get("DISCORD_WEBHOOK_URL")
            if not url:
                logger.error("--webhook-url または環境変数 DISCORD_WEBHOOK_URL を指定してください")
                return
            result = calc_ev_and_flags(result)
            race_name = result["race_name"].iloc[0] if "race_name" in result.columns else args.race_id
            msg = format_prediction(result, race_name=race_name)
            ok = send_discord(url, msg)
            logger.info("Discord への送信が完了しました" if ok else "Discord への送信に失敗しました")


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
        run_predict_notify(webhook_url=webhook)
    else:
        run_result_notify(webhook_url=webhook)


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
    p_train.set_defaults(func=cmd_train)

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
