# AstraFX (Render deployment)

This package is an exact copy of your local setup adapted for Render:
- Uses your **server.py** strategies/engine unchanged.
- Frontend files from your local root are in **public/** (including **AstraFX_live.html** and **index.html**).
- The UI now points its API base to **window.location.origin** (same origin), so no CORS or hardcoded localhost.

## Endpoints
- `/` → serves **public/index.html** (redirects to AstraFX_live.html, like your local launcher)
- `/AstraFX_live.html` → your full UI
- `/api/ohlc` and `/api/scan` → unchanged JSON API
- `/scan` and `/ohlc` → aliases redirecting to the API
- `/health` → as in your server

## Environment variables
- `OANDA_TOKEN` (required)
- `OANDA_ENV` = `practice` (default) or `live`
- Optional: `DATA_PROVIDER` = `OANDA` | `TWELVEDATA` | `ALPHAVANTAGE`

## Render deploy steps
1. Push these files to GitHub (replace existing files).
2. In Render: New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: leave blank (Render reads Procfile) or set `gunicorn -k uvicorn.workers.UvicornWorker server:app`
5. Add env vars above, then Deploy.
6. Open `/` and click **Scan**. If you run the same scan twice quickly, the API should respond normally.

## Local run (optional)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OANDA_TOKEN=... ; export OANDA_ENV=practice ; export PORT=10000
python server.py
# open http://localhost:10000/AstraFX_live.html
```
