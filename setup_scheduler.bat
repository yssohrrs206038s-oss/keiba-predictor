@echo off
:: ============================================================
::  週末重賞 Discord 通知 ── Windowsタスクスケジューラ登録スクリプト
::
::  【使い方】
::    1. このファイルを管理者権限で実行してください
::       （右クリック → 管理者として実行）
::    2. .env ファイルに DISCORD_WEBHOOK_URL を設定済みであること
::       例) DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
::
::  【登録されるタスク】
::    ・競馬予想_金曜予想 : 毎週金曜 09:00 → predict モード
::    ・競馬予想_日曜結果 : 毎週日曜 17:00 → result  モード
::
::  【削除するには】
::    schtasks /delete /tn "競馬予想_金曜予想" /f
::    schtasks /delete /tn "競馬予想_日曜結果" /f
:: ============================================================

setlocal

:: プロジェクトルート（このバッチファイルのディレクトリ）
set "PROJECT_DIR=%~dp0"
:: 末尾の \ を除去
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

:: Python 実行ファイルを確認
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python が見つかりません。Python をインストールしてください。
    pause
    exit /b 1
)

:: ── 金曜 09:00 予想タスク ──────────────────────────────────
echo [INFO] タスク「競馬予想_金曜予想」を登録中...
schtasks /create ^
  /tn "競馬予想_金曜予想" ^
  /tr "cmd /c \"cd /d \"%PROJECT_DIR%\" && python -m keiba_predictor.main notify --mode predict >> \"%PROJECT_DIR%\\logs\\predict.log\" 2>&1\"" ^
  /sc WEEKLY ^
  /d FRI ^
  /st 09:00 ^
  /f

if errorlevel 1 (
    echo [ERROR] 金曜予想タスクの登録に失敗しました。
    echo         管理者権限で実行しているか確認してください。
    pause
    exit /b 1
)
echo [OK] 競馬予想_金曜予想 登録完了

:: ── 日曜 17:00 結果タスク ──────────────────────────────────
echo [INFO] タスク「競馬予想_日曜結果」を登録中...
schtasks /create ^
  /tn "競馬予想_日曜結果" ^
  /tr "cmd /c \"cd /d \"%PROJECT_DIR%\" && python -m keiba_predictor.main notify --mode result >> \"%PROJECT_DIR%\\logs\\result.log\" 2>&1\"" ^
  /sc WEEKLY ^
  /d SUN ^
  /st 17:00 ^
  /f

if errorlevel 1 (
    echo [ERROR] 日曜結果タスクの登録に失敗しました。
    echo         管理者権限で実行しているか確認してください。
    pause
    exit /b 1
)
echo [OK] 競馬予想_日曜結果 登録完了

:: ── 確認 ───────────────────────────────────────────────────
echo.
echo ============================================================
echo  登録完了！
echo ============================================================
echo  金曜 09:00: python -m keiba_predictor.main notify --mode predict
echo  日曜 17:00: python -m keiba_predictor.main notify --mode result
echo.
echo  手動実行:
echo    python -m keiba_predictor.main notify --mode predict
echo    python -m keiba_predictor.main notify --mode result
echo.
echo  ログ:
echo    %PROJECT_DIR%\logs\predict.log
echo    %PROJECT_DIR%\logs\result.log
echo.
echo  環境変数 DISCORD_WEBHOOK_URL を .env に設定してください。
echo ============================================================

:: ログディレクトリを作成
if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs"

endlocal
pause
