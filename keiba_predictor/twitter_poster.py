"""
Twitter（X）投稿モジュール

tweepy を使用して Twitter に投稿する。

【環境変数】
    TWITTER_API_KEY              : Consumer Key（API Key）
    TWITTER_API_SECRET           : Consumer Secret（API Secret）
    TWITTER_ACCESS_TOKEN         : Access Token
    TWITTER_ACCESS_TOKEN_SECRET  : Access Token Secret

環境変数が未設定の場合は投稿をスキップし、エラーにはなりません。
"""

import logging
import os

logger = logging.getLogger(__name__)


def _build_client():
    """tweepy.Client を構築する。認証情報が未設定なら None を返す。"""
    api_key = os.environ.get("TWITTER_API_KEY", "")
    api_secret = os.environ.get("TWITTER_API_SECRET", "")
    access_token = os.environ.get("TWITTER_ACCESS_TOKEN", "")
    access_token_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET", "")

    if not all([api_key, api_secret, access_token, access_token_secret]):
        logger.warning("Twitter API 認証情報が未設定です → 投稿スキップ")
        return None

    try:
        import tweepy
    except ImportError:
        logger.warning("tweepy が未インストールです → pip install tweepy")
        return None

    return tweepy.Client(
        consumer_key=api_key,
        consumer_secret=api_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
    )


def post_tweet(text: str) -> bool:
    """
    Twitter にテキストを投稿する。

    Args:
        text: 投稿するテキスト（280文字以内推奨）

    Returns:
        True: 投稿成功、False: 投稿失敗またはスキップ
    """
    client = _build_client()
    if client is None:
        return False

    try:
        response = client.create_tweet(text=text)
        tweet_id = response.data["id"]
        logger.info(f"Twitter 投稿成功: id={tweet_id}")
        print(f"[Twitter] 投稿成功: id={tweet_id}", flush=True)
        return True
    except Exception as e:
        logger.error(f"Twitter 投稿失敗: {e}")
        print(f"[Twitter] 投稿失敗: {e}", flush=True)
        return False


def main():
    """テスト用: 「KEIBA EDGE 自動投稿テスト」をツイートする。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ok = post_tweet("KEIBA EDGE 自動投稿テスト\U0001f3c7")
    if ok:
        print("投稿完了", flush=True)
    else:
        print("投稿失敗またはスキップ", flush=True)


if __name__ == "__main__":
    main()
