export default async function handler(req, res) {
  try {
    const token = process.env.OANDA_TOKEN;
    if (!token) return res.status(500).json({ error: 'Missing OANDA_TOKEN' });

    const { pair = 'EURUSD', tf = 'M15', limit = '220' } = req.query;

    // Convert to OANDA format: EURUSD → EUR_USD
    let symbol = pair.toUpperCase();
    if (/^[A-Z]{6}$/.test(symbol)) symbol = symbol.slice(0,3) + '_' + symbol.slice(3);

    const base = 'https://api-fxpractice.oanda.com';
    const url  = `${base}/v3/instruments/${symbol}/candles?granularity=${tf}&count=${limit}&price=M`;

    const r = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
    const data = await r.json();

    if (!r.ok) return res.status(r.status).json(data);

    // Adapt to your frontend’s expected format
    const ohlc = (data.candles || []).map(c => ({
      open:  parseFloat(c.mid.o),
      high:  parseFloat(c.mid.h),
      low:   parseFloat(c.mid.l),
      close: parseFloat(c.mid.c),
      time:  c.time
    }));

    res.json({ ohlc });
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
}
