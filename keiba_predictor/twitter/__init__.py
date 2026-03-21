"""
Twitter/X 自動引用投稿モジュール

使い方:
    # Chromeを起動してXにログイン
    python -m keiba_predictor.main twitter login

    # TLを監視して伸びてる投稿に引用投稿
    python -m keiba_predictor.main twitter quote

    # 一定間隔で繰り返し実行
    python -m keiba_predictor.main twitter quote --interval 600
"""
