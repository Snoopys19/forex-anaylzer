// /api/ohlc.js — returns a simple OHLC array (o,h,l,c,v,t) for frontends expecting flat data
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
    if (!r.ok) return res.status(r.status).json(data);

    const candles = (data.candles || []).map(c => ({
      t: c.time,
      o: +((c.mid?.o) ?? c.bid?.o ?? c.ask?.o ?? 0),
      h: +((c.mid?.h) ?? c.bid?.h ?? c.ask?.h ?? 0),
      l: +((c.mid?.l) ?? c.bid?.l ?? c.ask?.l ?? 0),
      c: +((c.mid?.c) ?? c.bid?.c ?? c.ask?.c ?? 0),
      v: c.volume ?? 0
    }));

    res.json({ symbol: data.instrument || symbol, granularity, candles });
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
}
