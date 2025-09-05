# AstraFX Backend (Vercel Port)

## Setup
1. Replace `server.py` with your actual backend `server.py` from your zip export.
2. Push this folder to your GitHub repo.
3. Connect repo to Vercel. Vercel will deploy the FastAPI app.

## Env Vars
- `OANDA_TOKEN` (required)
- `OANDA_ENV` (optional: practice/live)
- `DATA_PROVIDER` (optional: OANDA/TWELVEDATA/ALPHAVANTAGE)

## Test URLs
- `/health`
- `/scan?pairs=EURUSD,GBPUSD&tf=H1&min_score=70`
- `/ohlc?pair=EURUSD&tf=H1&limit=200`
