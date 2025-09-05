// /api/scan.js — compatibility shim
export default async function handler(req, res) {
  try {
    const token = process.env.OANDA_TOKEN;
    if (!token) return res.status(500).json({ error: 'Server missing OANDA_TOKEN' });

    let { symbol = 'EUR_USD', granularity = 'M15', count = '200', price = 'M' } = req.query;
    if (/^[A-Z]{6}$/.test(symbol)) symbol = symbol.slice(0,3) + '_' + symbol.slice(3);

    const base = 'https://api-fxpractice.oanda.com';
    const url  = `${base}/v3/instruments/${encodeURIComponent(symbol)}/candles?granularity=${granularity}&count=${count}&price=${price}`;

    const r = await fetch(url, { headers: { Authorization: `Bearer ${token}`, Accept: 'application/json' } });
    const data = await r.json();
    res.status(r.ok ? 200 : r.status).json(data);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
}
