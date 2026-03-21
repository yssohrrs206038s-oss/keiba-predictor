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

    # 7. 全ステップを一括実行
    python -m keiba_predictor.main all --start 2023-01 --end 2023-12

    # 8. Twitter/X 連携
    # Xにログイン（初回・セッション切れ時）
    python -m keiba_predictor.main twitter login --username YOUR_USERNAME --password YOUR_PASSWORD

    # TLを監視して伸びてる投稿に引用ポスト
    python -m keiba_predictor.main twitter quote

    # ドライラン（実際には投稿しない）
    python -m keiba_predictor.main twitter quote --dry-run

    # 競馬テンプレートを使用して一定間隔で繰り返す
    python -m keiba_predictor.main twitter quote --keiba --interval 600 --max-posts 3
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
    from keiba_predictor.model.predict import predict_from_csv
    predict_from_csv(args.race_id)


def cmd_all(args: argparse.Namespace) -> None:
    """スクレイピング → クリーニング → 特徴量 → 学習 を一括実行"""
    cmd_scrape(args)
    cmd_clean(args)
    cmd_features(args)
    cmd_train(args)


# ── Twitter/X 連携コマンド ──────────────────────────────────────

def cmd_twitter_login(args: argparse.Namespace) -> None:
    """ChromeでXにログインしてセッションを保存する。"""
    from keiba_predictor.twitter.auth import login

    success = login(
        username=getattr(args, "username", None),
        password=getattr(args, "password", None),
        headless=getattr(args, "headless", False),
    )
    if success:
        logger.info("ログイン成功！次回からセッションが再利用されます。")
    else:
        logger.error("ログインに失敗しました。認証情報を確認してください。")
        sys.exit(1)


def cmd_twitter_quote(args: argparse.Namespace) -> None:
    """
    TLをスキャンして伸びてる/伸びそうな投稿に引用ポストを作成する。

    --interval が指定された場合は繰り返し実行する。
    """
    import time as time_mod
    from keiba_predictor.twitter.auth import open_browser_with_session
    from keiba_predictor.twitter.timeline import fetch_trending_tweets
    from keiba_predictor.twitter.quote_poster import process_trending_tweets

    dry_run = getattr(args, "dry_run", False)
    interval = getattr(args, "interval", 0)
    scroll_count = getattr(args, "scroll", 5)
    max_posts = getattr(args, "max_posts", 5)
    template = getattr(args, "template", None)
    use_keiba = getattr(args, "keiba", False)
    headless = getattr(args, "headless", False)
    min_likes = getattr(args, "min_likes", 0)

    already_quoted: set = set()

    pw, browser, context, page = open_browser_with_session(headless=headless)

    try:
        while True:
            logger.info("=== タイムラインスキャン開始 ===")

            # TLに戻る
            page.goto("https://x.com/home", wait_until="domcontentloaded", timeout=30000)
            time_mod.sleep(3)

            # トレンド投稿を取得
            trending = fetch_trending_tweets(
                page,
                scroll_count=scroll_count,
                min_likes=min_likes,
            )

            if not trending:
                logger.info("トレンド候補が見つかりませんでした。")
            else:
                # 引用投稿を実行
                results = process_trending_tweets(
                    page,
                    trending,
                    template=template,
                    use_keiba_template=use_keiba,
                    max_posts=max_posts,
                    dry_run=dry_run,
                    already_quoted=already_quoted,
                )

                success = sum(1 for r in results if r.success)
                logger.info(f"今回の実行: {success}/{len(results)} 件投稿")

            if interval <= 0:
                break

            logger.info(f"次のスキャンまで {interval} 秒待機...")
            time_mod.sleep(interval)

    finally:
        browser.close()
        pw.stop()


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
    p_pred.set_defaults(func=cmd_predict)

    # ── all ────────────────────────────────────────────────
    p_all = sub.add_parser("all", help="全ステップを一括実行")
    p_all.add_argument("--start", metavar="YYYY-MM", help="取得開始年月")
    p_all.add_argument("--end", metavar="YYYY-MM", help="取得終了年月")
    p_all.add_argument("--cv-splits", type=int, default=5)
    p_all.set_defaults(func=cmd_all)

    # ── twitter ────────────────────────────────────────────
    p_tw = sub.add_parser("twitter", help="Twitter/X 連携機能")
    tw_sub = p_tw.add_subparsers(dest="twitter_command", required=True)

    # twitter login
    p_tw_login = tw_sub.add_parser("login", help="ChromeでXにログインしてセッションを保存")
    p_tw_login.add_argument(
        "--username", "-u",
        help="Xのユーザー名またはメールアドレス (未指定時は環境変数 X_USERNAME を使用)"
    )
    p_tw_login.add_argument(
        "--password", "-p",
        help="Xのパスワード (未指定時は環境変数 X_PASSWORD を使用)"
    )
    p_tw_login.add_argument(
        "--headless", action="store_true",
        help="ヘッドレスモードで起動 (画面なし)"
    )
    p_tw_login.set_defaults(func=cmd_twitter_login)

    # twitter quote
    p_tw_quote = tw_sub.add_parser(
        "quote",
        help="TLをスキャンして伸びてる投稿に引用ポストを作成"
    )
    p_tw_quote.add_argument(
        "--dry-run", action="store_true",
        help="実際には投稿せずログのみ表示"
    )
    p_tw_quote.add_argument(
        "--interval", type=int, default=0, metavar="SECONDS",
        help="繰り返し実行する間隔（秒）。0なら1回だけ実行 (デフォルト: 0)"
    )
    p_tw_quote.add_argument(
        "--scroll", type=int, default=5, metavar="N",
        help="TLをスクロールする回数 (デフォルト: 5)"
    )
    p_tw_quote.add_argument(
        "--max-posts", type=int, default=5, metavar="N",
        help="1回の実行で投稿する最大件数 (デフォルト: 5)"
    )
    p_tw_quote.add_argument(
        "--template", metavar="TEXT",
        help="カスタム引用テンプレート ({hashtags} でハッシュタグを挿入)"
    )
    p_tw_quote.add_argument(
        "--keiba", action="store_true",
        help="競馬向けテンプレートを使用"
    )
    p_tw_quote.add_argument(
        "--min-likes", type=int, default=0, metavar="N",
        help="対象とする最低いいね数 (デフォルト: 0)"
    )
    p_tw_quote.add_argument(
        "--headless", action="store_true",
        help="ヘッドレスモードで起動 (画面なし)"
    )
    p_tw_quote.set_defaults(func=cmd_twitter_quote)

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
