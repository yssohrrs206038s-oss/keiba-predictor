"""
Anthropic API クレジット残量監視

残量が閾値以下になったら Discord に警告通知する。

使い方:
    python -m keiba_predictor.credit_monitor
"""

import logging
import os
import sys

import requests

logger = logging.getLogger(__name__)

THRESHOLD = 1.0  # USD


def check_credit() -> None:
    """Anthropic API のクレジット残量を確認し、閾値以下なら Discord に警告する。"""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")

    if not api_key:
        logger.warning("ANTHROPIC_API_KEY が未設定です")
        return

    # Anthropic Balance API
    try:
        resp = requests.get(
            "https://api.anthropic.com/v1/organizations/balance",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"クレジット残量取得失敗: {e}")
        # APIがない場合はUsage APIを試す
        try:
            resp = requests.get(
                "https://api.anthropic.com/v1/usage",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e2:
            logger.warning(f"Usage API も失敗: {e2}")
            return

    # 残量を取得（APIレスポンス構造に応じて調整）
    balance = None
    if "balance" in data:
        balance = float(data["balance"])
    elif "credit_balance" in data:
        balance = float(data["credit_balance"])
    elif "available" in data:
        balance = float(data["available"])

    if balance is None:
        logger.info(f"クレジット残量を解析できませんでした。レスポンス: {data}")
        return

    logger.info(f"Anthropic クレジット残量: ${balance:.2f}")
    print(f"[credit] 残量: ${balance:.2f} (閾値: ${THRESHOLD:.2f})", flush=True)

    if balance <= THRESHOLD:
        msg = (
            f"⚠️ Anthropic APIクレジット残量が${THRESHOLD:.2f}を下回りました"
            f"（現在: ${balance:.2f}）。チャージを検討してください。"
        )
        logger.warning(msg)
        if webhook_url:
            try:
                requests.post(webhook_url, json={"content": msg}, timeout=15)
                logger.info("Discord 警告送信完了")
            except Exception as e:
                logger.warning(f"Discord 送信失敗: {e}")
        else:
            print(msg, flush=True)
    else:
        print(f"[credit] OK: ${balance:.2f} > ${THRESHOLD:.2f}", flush=True)


def check_x_api_credit() -> None:
    """X API のクレジット残量を確認し、閾値以下なら Discord に警告する。"""
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN", "")
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")

    if not bearer_token:
        logger.info("TWITTER_BEARER_TOKEN が未設定 → X APIクレジットチェックをスキップ")
        return

    try:
        resp = requests.get(
            "https://api.twitter.com/2/usage/tweets",
            headers={"Authorization": f"Bearer {bearer_token}"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"X API クレジット残量取得失敗: {e}")
        return

    # レスポンスから残量を取得
    balance = None
    if "data" in data:
        d = data["data"]
        if "cap_remaining" in d:
            balance = float(d["cap_remaining"])
        elif "daily_project_usage" in d:
            # 使用量からの推定
            usage = d["daily_project_usage"]
            if isinstance(usage, list) and usage:
                used = sum(u.get("usage", [{}])[0].get("tweets", 0) for u in usage if u.get("usage"))
                logger.info(f"X API 今日の使用量: {used} tweets")
    elif "cap_remaining" in data:
        balance = float(data["cap_remaining"])

    if balance is not None:
        logger.info(f"X API クレジット残量: ${balance:.2f}")
        print(f"[credit] X API 残量: ${balance:.2f} (閾値: ${THRESHOLD:.2f})", flush=True)

        if balance <= THRESHOLD:
            msg = (
                f"⚠️ X APIクレジット残量が${THRESHOLD:.2f}を下回りました"
                f"（現在: ${balance:.2f}）。チャージを検討してください。"
            )
            logger.warning(msg)
            if webhook_url:
                try:
                    requests.post(webhook_url, json={"content": msg}, timeout=15)
                    logger.info("Discord X API警告送信完了")
                except Exception as e:
                    logger.warning(f"Discord 送信失敗: {e}")
            else:
                print(msg, flush=True)
        else:
            print(f"[credit] X API OK: ${balance:.2f} > ${THRESHOLD:.2f}", flush=True)
    else:
        logger.info(f"X API クレジット残量を解析できませんでした。レスポンス: {data}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    check_credit()
    check_x_api_credit()
