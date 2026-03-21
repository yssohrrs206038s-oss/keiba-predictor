"""
タイムライン監視・トレンド分析モジュール

XのTLをスキャンして「伸びてる or 伸びそうな」投稿を検出する。

判定ロジック:
  - 既に伸びてる: いいね数 or リポスト数が閾値を超えている
  - 伸びそう (急上昇): 投稿後の経過時間が短いのにエンゲージメントが多い
                       = エンゲージメント速度 (件/分) が高い
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── トレンド判定の閾値 ───────────────────────────────────────────
# 既に伸びてる投稿
TRENDING_LIKES_THRESHOLD = 500        # いいね 500以上
TRENDING_REPOST_THRESHOLD = 100       # リポスト 100以上
TRENDING_VIEWS_THRESHOLD = 10_000     # 表示回数 1万以上

# 伸びそう (エンゲージメント速度)
RISING_VELOCITY_THRESHOLD = 10.0     # 10件/分 以上のエンゲージメント増加
RISING_MAX_AGE_MINUTES = 60          # 投稿後60分以内の投稿のみ対象


@dataclass
class TweetData:
    """スクレイピングした投稿データ"""
    tweet_id: str
    tweet_url: str
    author_name: str
    author_handle: str
    text: str
    posted_at: Optional[datetime]
    likes: int = 0
    reposts: int = 0
    replies: int = 0
    views: int = 0
    bookmarks: int = 0
    is_trending: bool = False
    is_rising: bool = False
    engagement_velocity: float = 0.0  # エンゲージメント/分


def _parse_count(text: str) -> int:
    """
    'K' '万' などの単位が含まれた数値文字列を整数に変換する。

    例: '1.2K' -> 1200, '2.5万' -> 25000, '1,234' -> 1234
    """
    if not text:
        return 0
    text = text.strip().replace(",", "").replace(" ", "")
    try:
        if text.endswith("K") or text.endswith("k"):
            return int(float(text[:-1]) * 1_000)
        if text.endswith("M") or text.endswith("m"):
            return int(float(text[:-1]) * 1_000_000)
        if text.endswith("万"):
            return int(float(text[:-1]) * 10_000)
        if text.endswith("億"):
            return int(float(text[:-1]) * 100_000_000)
        return int(float(text))
    except (ValueError, TypeError):
        return 0


def _parse_posted_at(datetime_str: str) -> Optional[datetime]:
    """
    投稿日時文字列をdatetimeに変換する。

    例: '2024-01-15T10:30:00.000Z'
    """
    if not datetime_str:
        return None
    try:
        return datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _scrape_tweet_from_article(article) -> Optional[TweetData]:
    """
    Playwrightのarticle要素から投稿データを抽出する。
    """
    try:
        # ツイートURL・IDの取得
        time_el = article.query_selector("time")
        if not time_el:
            return None

        posted_at = _parse_posted_at(time_el.get_attribute("datetime") or "")

        link_el = time_el.query_selector("xpath=..")  # timeの親aタグ
        if not link_el:
            # timeの親要素を探す
            link_el = article.query_selector('a[href*="/status/"]')

        tweet_url = ""
        tweet_id = ""
        if link_el:
            href = link_el.get_attribute("href") or ""
            if "/status/" in href:
                tweet_url = f"https://x.com{href}" if href.startswith("/") else href
                tweet_id = href.split("/status/")[-1].split("?")[0]

        if not tweet_id:
            return None

        # テキスト取得
        text_el = article.query_selector('[data-testid="tweetText"]')
        text = text_el.inner_text() if text_el else ""

        # 著者情報
        user_el = article.query_selector('[data-testid="User-Name"]')
        author_name = ""
        author_handle = ""
        if user_el:
            spans = user_el.query_selector_all("span")
            texts = [s.inner_text().strip() for s in spans if s.inner_text().strip()]
            # 通常は [表示名, @handle] の順
            for t in texts:
                if t.startswith("@") and not author_handle:
                    author_handle = t
                elif not author_name and not t.startswith("@"):
                    author_name = t

        # エンゲージメント数値
        def get_count(testid: str) -> int:
            el = article.query_selector(f'[data-testid="{testid}"]')
            if not el:
                return 0
            span = el.query_selector('span[data-testid="app-text-transition-container"]')
            if span:
                return _parse_count(span.inner_text())
            return _parse_count(el.inner_text())

        likes = get_count("like")
        reposts = get_count("retweet")
        replies = get_count("reply")
        bookmarks = get_count("bookmark")

        # 表示回数 (Views)
        views = 0
        analytics_el = article.query_selector('[data-testid="analytics"]')
        if analytics_el:
            views = _parse_count(analytics_el.inner_text())
        else:
            # aria-labelで取得を試みる
            view_el = article.query_selector('a[href*="analytics"]')
            if view_el:
                views = _parse_count(view_el.inner_text())

        return TweetData(
            tweet_id=tweet_id,
            tweet_url=tweet_url,
            author_name=author_name,
            author_handle=author_handle,
            text=text,
            posted_at=posted_at,
            likes=likes,
            reposts=reposts,
            replies=replies,
            views=views,
            bookmarks=bookmarks,
        )

    except Exception as e:
        logger.debug(f"ツイート取得中にエラー: {e}")
        return None


def analyze_trend(tweet: TweetData) -> TweetData:
    """
    投稿のトレンド状態を分析し、is_trending / is_rising / engagement_velocity を設定する。
    """
    now = datetime.now(timezone.utc)

    # エンゲージメント速度を計算
    if tweet.posted_at:
        age_minutes = max((now - tweet.posted_at).total_seconds() / 60, 1)
        total_engagement = tweet.likes + tweet.reposts + tweet.replies
        tweet.engagement_velocity = total_engagement / age_minutes
    else:
        age_minutes = None
        tweet.engagement_velocity = 0.0

    # 既に伸びてる判定
    tweet.is_trending = (
        tweet.likes >= TRENDING_LIKES_THRESHOLD
        or tweet.reposts >= TRENDING_REPOST_THRESHOLD
        or tweet.views >= TRENDING_VIEWS_THRESHOLD
    )

    # 急上昇 (伸びそう) 判定: 投稿後60分以内 かつ エンゲージメント速度が高い
    if age_minutes is not None:
        tweet.is_rising = (
            age_minutes <= RISING_MAX_AGE_MINUTES
            and tweet.engagement_velocity >= RISING_VELOCITY_THRESHOLD
            and not tweet.is_trending  # 既にトレンドなら重複しない
        )

    return tweet


def fetch_trending_tweets(
    page,
    scroll_count: int = 5,
    min_likes: int = 0,
) -> List[TweetData]:
    """
    TLをスクロールしながら「伸びてる or 伸びそうな」投稿を収集する。

    Args:
        page: Playwrightのページオブジェクト (XのTLが開いている状態)
        scroll_count: スクロール回数 (多いほど多くの投稿をチェック)
        min_likes: 最低いいね数フィルタ (0なら全投稿対象)

    Returns:
        トレンド判定された TweetData のリスト
    """
    seen_ids: set[str] = set()
    results: List[TweetData] = []

    logger.info("タイムラインをスキャン中...")

    for scroll_idx in range(scroll_count + 1):
        # 現在表示中のツイート一覧を取得
        articles = page.query_selector_all('article[data-testid="tweet"]')
        logger.debug(f"スクロール {scroll_idx}: {len(articles)} 件の投稿を検出")

        for article in articles:
            tweet = _scrape_tweet_from_article(article)
            if not tweet:
                continue
            if tweet.tweet_id in seen_ids:
                continue
            if tweet.likes < min_likes:
                continue

            seen_ids.add(tweet.tweet_id)
            tweet = analyze_trend(tweet)

            if tweet.is_trending or tweet.is_rising:
                results.append(tweet)
                label = "🔥伸びてる" if tweet.is_trending else "🚀伸びそう"
                logger.info(
                    f"{label} | {tweet.author_handle} | "
                    f"❤️{tweet.likes} 🔁{tweet.reposts} "
                    f"⚡{tweet.engagement_velocity:.1f}/分 | "
                    f"{tweet.text[:40]}..."
                )

        # 次のページへスクロール
        if scroll_idx < scroll_count:
            page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
            time.sleep(2.5)

    logger.info(
        f"スキャン完了: {len(seen_ids)} 件中 {len(results)} 件がトレンド候補"
    )
    return results
