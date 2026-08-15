@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "APP_URL=http://localhost:8000/demo.html"

where docker >nul 2>&1 || (
  echo Docker Desktop with Docker Compose is required.
  pause
  exit /b 1
)

if not exist "outputs\checkpoints\best.pt" (
  echo Production checkpoint not found: outputs\checkpoints\best.pt
  pause
  exit /b 1
)

echo Starting RL-ROI-Net with outputs\checkpoints\best.pt ...
docker compose up --build -d api || (
  echo.
  echo The API did not start. Check Docker Desktop, NVIDIA Container Toolkit, and GPU access.
  docker compose logs api
  pause
  exit /b 1
)

for /L %%I in (1,1,40) do (
  powershell -NoProfile -Command "try { $r = Invoke-RestMethod -TimeoutSec 3 http://localhost:8000/health; if ($r.ready) { exit 0 }; exit 1 } catch { exit 1 }" >nul 2>&1
  if not errorlevel 1 goto :ready
  timeout /t 3 /nobreak >nul
)

echo.
echo The API started but is not ready. Check the checkpoint, CUDA runtime, and logs:
docker compose logs api
pause
exit /b 1

:ready
start "RL-ROI-Net" "%APP_URL%"
echo RL-ROI-Net is ready at %APP_URL%
echo To stop it later, run: docker compose stop api
endlocal
