"""
引用投稿作成モジュール

トレンド投稿に対して引用ポストを作成・投稿する。

引用文のテンプレートは競馬予想システムのコンテキストに合わせているが、
カスタムテンプレートも設定可能。
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import List, Optional

from keiba_predictor.twitter.timeline import TweetData

logger = logging.getLogger(__name__)

# デフォルト引用テンプレート (競馬関連 + 汎用)
DEFAULT_QUOTE_TEMPLATES = [
    "これは注目！{hashtags}",
    "興味深い視点ですね。{hashtags}",
    "参考になります👀 {hashtags}",
    "確かに！{hashtags}",
    "要チェックです📌 {hashtags}",
    "なるほど🤔 {hashtags}",
    "シェアします🔁 {hashtags}",
    "注目の投稿です⚡ {hashtags}",
]

KEIBA_QUOTE_TEMPLATES = [
    "競馬ファン必見！🏇 {hashtags}",
    "レース情報として参考にします🏇 {hashtags}",
    "馬券検討の参考に📊 {hashtags}",
    "競馬予想に役立つ情報です🎯 {hashtags}",
]


@dataclass
class QuoteResult:
    """引用投稿の結果"""
    tweet: TweetData
    quote_text: str
    success: bool
    error: Optional[str] = None
    quote_url: Optional[str] = None


def generate_quote_text(
    tweet: TweetData,
    template: Optional[str] = None,
    use_keiba_template: bool = False,
) -> str:
    """
    引用投稿のテキストを生成する。

    Args:
        tweet: 引用対象の投稿データ
        template: カスタムテンプレート ({hashtags} を含めると自動挿入)
        use_keiba_template: 競馬向けテンプレートを使用するか

    Returns:
        引用投稿のテキスト
    """
    if template:
        templates = [template]
    elif use_keiba_template:
        templates = KEIBA_QUOTE_TEMPLATES
    else:
        templates = DEFAULT_QUOTE_TEMPLATES

    chosen = random.choice(templates)

    # ハッシュタグを元ツイートから抽出
    import re
    hashtags_in_tweet = re.findall(r"#\S+", tweet.text)
    hashtags = " ".join(hashtags_in_tweet[:3]) if hashtags_in_tweet else ""

    return chosen.format(hashtags=hashtags).strip()


def post_quote_tweet(
    page,
    tweet: TweetData,
    quote_text: str,
    dry_run: bool = False,
) -> QuoteResult:
    """
    指定した投稿への引用ポストを作成する。

    Args:
        page: Playwrightのページオブジェクト
        tweet: 引用対象の投稿
        quote_text: 引用テキスト
        dry_run: Trueの場合、実際には投稿せずログのみ

    Returns:
        QuoteResult
    """
    if dry_run:
        logger.info(
            f"[DRY RUN] 引用投稿をスキップ\n"
            f"  対象: {tweet.tweet_url}\n"
            f"  テキスト: {quote_text}"
        )
        return QuoteResult(
            tweet=tweet,
            quote_text=quote_text,
            success=True,
            error=None,
            quote_url="[dry_run]",
        )

    try:
        logger.info(f"引用投稿を作成中: {tweet.tweet_url}")

        # 対象ツイートのページを開く
        page.goto(tweet.tweet_url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)

        # 引用ポストボタンをクリック (リポストアイコン→引用ポスト)
        retweet_btn = page.wait_for_selector(
            f'[data-testid="retweet"]',
            timeout=10000,
        )
        retweet_btn.click()
        time.sleep(1)

        # 「引用する」メニュー項目を選択
        quote_option = page.wait_for_selector(
            '[data-testid="quoteTweet"]',
            timeout=5000,
        )
        quote_option.click()
        time.sleep(2)

        # 引用投稿のテキストエリアに入力
        text_area = page.wait_for_selector(
            '[data-testid="tweetTextarea_0"]',
            timeout=10000,
        )
        text_area.click()
        text_area.fill(quote_text)
        time.sleep(1)

        # 投稿ボタンをクリック
        post_btn = page.wait_for_selector(
            '[data-testid="tweetButtonInline"]',
            timeout=5000,
        )
        post_btn.click()
        time.sleep(3)

        logger.info(f"引用投稿を送信しました: '{quote_text[:50]}...'")
        return QuoteResult(
            tweet=tweet,
            quote_text=quote_text,
            success=True,
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"引用投稿に失敗: {error_msg}")

        # デバッグ用スクリーンショット
        try:
            from keiba_predictor.twitter.browser import DATA_DIR
            ss_path = DATA_DIR / f"quote_error_{tweet.tweet_id}.png"
            page.screenshot(path=str(ss_path))
            logger.info(f"エラー時のスクリーンショット: {ss_path}")
        except Exception:
            pass

        return QuoteResult(
            tweet=tweet,
            quote_text=quote_text,
            success=False,
            error=error_msg,
        )


def process_trending_tweets(
    page,
    tweets: List[TweetData],
    template: Optional[str] = None,
    use_keiba_template: bool = False,
    max_posts: int = 5,
    post_interval_sec: float = 30.0,
    dry_run: bool = False,
    already_quoted: Optional[set] = None,
) -> List[QuoteResult]:
    """
    トレンド投稿リストに対して順次引用ポストを作成する。

    Args:
        page: Playwrightのページオブジェクト
        tweets: トレンド投稿リスト (fetch_trending_tweets の結果)
        template: カスタム引用テンプレート
        use_keiba_template: 競馬向けテンプレートを使用するか
        max_posts: 1回の実行で投稿する最大件数
        post_interval_sec: 投稿間の待機秒数 (レート制限対策)
        dry_run: Trueの場合、実際には投稿しない
        already_quoted: 既に引用済みのツイートIDセット (重複防止)

    Returns:
        QuoteResult のリスト
    """
    if already_quoted is None:
        already_quoted = set()

    results: List[QuoteResult] = []
    posted_count = 0

    # 伸びてる順にソート (いいね + リポスト)
    sorted_tweets = sorted(
        tweets,
        key=lambda t: (t.is_trending, t.likes + t.reposts * 3, t.engagement_velocity),
        reverse=True,
    )

    for tweet in sorted_tweets:
        if posted_count >= max_posts:
            logger.info(f"最大投稿数 ({max_posts} 件) に達しました。終了します。")
            break

        if tweet.tweet_id in already_quoted:
            logger.debug(f"既に引用済み: {tweet.tweet_id}")
            continue

        quote_text = generate_quote_text(
            tweet,
            template=template,
            use_keiba_template=use_keiba_template,
        )

        result = post_quote_tweet(page, tweet, quote_text, dry_run=dry_run)
        results.append(result)

        if result.success:
            already_quoted.add(tweet.tweet_id)
            posted_count += 1

            if posted_count < max_posts and not dry_run:
                logger.info(f"{post_interval_sec}秒待機中...")
                time.sleep(post_interval_sec)
        else:
            logger.warning(f"投稿失敗のためスキップ: {tweet.tweet_id}")

    success_count = sum(1 for r in results if r.success)
    logger.info(
        f"引用投稿完了: {success_count}/{len(results)} 件成功"
        + (" [DRY RUN]" if dry_run else "")
    )
    return results
