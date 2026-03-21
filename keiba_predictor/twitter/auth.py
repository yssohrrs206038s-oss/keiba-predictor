"""
X (Twitter) ログイン・セッション管理モジュール

Playwrightを使ってChromeでXにログインし、セッションを保存・再利用する。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from keiba_predictor.twitter.browser import (
    get_browser_and_context,
    save_session,
    is_session_saved,
)

logger = logging.getLogger(__name__)

X_HOME_URL = "https://x.com/home"
X_LOGIN_URL = "https://x.com/i/flow/login"


def is_logged_in(page) -> bool:
    """現在のページがログイン済み状態かを確認する。"""
    try:
        page.goto(X_HOME_URL, wait_until="domcontentloaded", timeout=30000)
        time.sleep(2)
        current_url = page.url
        # ログインページにリダイレクトされていなければログイン済み
        if "login" in current_url or "flow" in current_url:
            return False
        # TLが表示されているか確認
        timeline = page.query_selector('[data-testid="primaryColumn"]')
        return timeline is not None
    except Exception as e:
        logger.warning(f"ログイン状態確認中にエラー: {e}")
        return False


def login(
    username: Optional[str] = None,
    password: Optional[str] = None,
    headless: bool = False,
) -> bool:
    """
    ChromeでXにログインし、セッションを保存する。

    認証情報は引数または環境変数 X_USERNAME / X_PASSWORD から取得する。

    Args:
        username: Xのユーザー名またはメールアドレス
        password: Xのパスワード
        headless: ヘッドレスモードで起動するか

    Returns:
        ログイン成功したか
    """
    username = username or os.environ.get("X_USERNAME", "")
    password = password or os.environ.get("X_PASSWORD", "")

    if not username or not password:
        raise ValueError(
            "認証情報が見つかりません。\n"
            "環境変数 X_USERNAME と X_PASSWORD を設定するか、\n"
            "--username / --password オプションで指定してください。"
        )

    pw, browser, context = get_browser_and_context(
        headless=headless,
        use_saved_session=False,  # ログイン時は既存セッションを使わない
    )

    try:
        page = context.new_page()
        logger.info("Xのログインページを開きます...")
        page.goto(X_LOGIN_URL, wait_until="networkidle", timeout=30000)
        time.sleep(2)

        # ユーザー名入力
        logger.info("ユーザー名を入力中...")
        username_input = page.wait_for_selector(
            'input[autocomplete="username"]',
            timeout=15000,
        )
        username_input.fill(username)
        time.sleep(0.5)

        # 「次へ」ボタン
        next_btn = page.query_selector('[data-testid="LoginForm_Forward_Button"]')
        if not next_btn:
            # テキストで検索
            next_btn = page.get_by_role("button", name="次へ").first
        next_btn.click()
        time.sleep(2)

        # 電話番号/ユーザー名の追加確認が出る場合がある
        unusual_activity = page.query_selector(
            'input[data-testid="ocfEnterTextTextInput"]'
        )
        if unusual_activity:
            logger.info("追加確認が必要です。ユーザー名を再入力中...")
            unusual_activity.fill(username)
            next_btn2 = page.query_selector('[data-testid="ocfEnterTextNextButton"]')
            if next_btn2:
                next_btn2.click()
            time.sleep(2)

        # パスワード入力
        logger.info("パスワードを入力中...")
        password_input = page.wait_for_selector(
            'input[name="password"]',
            timeout=15000,
        )
        password_input.fill(password)
        time.sleep(0.5)

        # ログインボタン
        login_btn = page.query_selector('[data-testid="LoginForm_Login_Button"]')
        if not login_btn:
            login_btn = page.get_by_role("button", name="ログイン").first
        login_btn.click()

        # ログイン完了待ち
        logger.info("ログイン処理中...")
        page.wait_for_url("**/home", timeout=30000)
        time.sleep(3)

        # セッションを保存
        save_session(context)
        logger.info("ログイン成功！セッションを保存しました。")
        return True

    except Exception as e:
        logger.error(f"ログイン中にエラーが発生しました: {e}")
        # デバッグ用スクリーンショット
        try:
            from keiba_predictor.twitter.browser import DATA_DIR
            ss_path = DATA_DIR / "login_error.png"
            page.screenshot(path=str(ss_path))
            logger.info(f"エラー時のスクリーンショット: {ss_path}")
        except Exception:
            pass
        return False

    finally:
        browser.close()
        pw.stop()


def open_browser_with_session(headless: bool = False):
    """
    保存済みセッションでChromeを起動し、XのTLを開く。

    Returns:
        (playwright, browser, context, page) のタプル
    """
    if not is_session_saved():
        raise RuntimeError(
            "保存済みセッションがありません。先に `twitter login` コマンドを実行してください。"
        )

    pw, browser, context = get_browser_and_context(
        headless=headless,
        use_saved_session=True,
    )
    page = context.new_page()

    logger.info("Xのタイムラインを開きます...")
    page.goto(X_HOME_URL, wait_until="domcontentloaded", timeout=30000)
    time.sleep(3)

    if not is_logged_in(page):
        browser.close()
        pw.stop()
        raise RuntimeError(
            "セッションが期限切れです。`twitter login` コマンドで再ログインしてください。"
        )

    logger.info("ログイン済みセッションでXを開きました。")
    return pw, browser, context, page
