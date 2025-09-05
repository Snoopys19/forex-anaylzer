# AstraFX + Vercel (Frontend + Backend together)

This contains your AstraFX UI and a serverless API for OANDA.

## Deploy
1) Push to a GitHub repo.
2) On **Vercel → New Project**, import your repo and **Deploy**.
3) In Project **Settings → Environment Variables**, add:
   - `OANDA_TOKEN` = your OANDA Practice API token
4) Visit your URL and click **Analyze**.

If you later host an external API, you can set:
```js
localStorage.setItem('astra_api','https://my-api.example.com');
```
