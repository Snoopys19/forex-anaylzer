// AstraFX minimal Supabase Auth integration (no UI changes to header).
// Adds Alt+A to open an auth modal. Exposes window.AstraAuth helpers.
(function(){
  const SUPABASE_URL = window.ASTRA_SUPABASE_URL || "";
  const SUPABASE_ANON = window.ASTRA_SUPABASE_ANON || "";
  let supa = null;

  function ensureClient(){
    if (!window.supabase) { alert("Supabase SDK not loaded."); return null; }
    if (!SUPABASE_URL || !SUPABASE_ANON){
      console.warn("Set window.ASTRA_SUPABASE_URL and window.ASTRA_SUPABASE_ANON in index.html");
    }
    if (!supa) supa = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON);
    return supa;
  }

  const css = `
  .astra-auth-modal{ position:fixed; inset:0; display:none; align-items:center; justify-content:center; z-index:2147483000; }
  .astra-auth-modal.show{ display:flex; }
  .astra-auth-backdrop{ position:absolute; inset:0; background:rgba(0,0,0,.5); }
  .astra-auth-card{ position:relative; width: 320px; max-width:90vw; background:#0b0d19; color:#fff; border-radius:16px; padding:18px; box-shadow:0 10px 30px rgba(0,0,0,.6); border:1px solid rgba(255,255,255,.08); font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  .astra-auth-card h3{ margin:0 0 8px; font-size:18px; }
  .astra-auth-card label{ display:block; font-size:12px; opacity:.8; margin-top:10px; }
  .astra-auth-card input{ width:100%; padding:10px 12px; border-radius:10px; border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.06); color:#fff; outline:none; }
  .astra-auth-actions{ display:flex; gap:8px; margin-top:14px; }
  .astra-auth-btn{ flex:1; padding:10px 12px; border-radius:10px; border:1px solid rgba(255,255,255,.15); background:linear-gradient(180deg,#1f2a44,#172134); color:#fff; cursor:pointer; font-weight:600; }
  .astra-auth-close{ position:absolute; right:10px; top:10px; background:transparent; border:none; color:#fff; font-size:18px; cursor:pointer; }
  .astra-auth-hint{ font-size:12px; opacity:.7; margin-top:6px; }
  `;
  const style = document.createElement('style'); style.textContent = css; document.head.appendChild(style);

  const modal = document.createElement('div'); modal.className='astra-auth-modal';
  modal.innerHTML = `
    <div class="astra-auth-backdrop"></div>
    <div class="astra-auth-card">
      <button class="astra-auth-close" title="Close">✕</button>
      <h3>AstraFX Account</h3>
      <label>Email</label>
      <input id="astra-auth-email" type="email" placeholder="you@example.com" spellcheck="false" autocomplete="email" />
      <label>Password</label>
      <input id="astra-auth-pass" type="password" placeholder="••••••••" autocomplete="current-password" />
      <div class="astra-auth-actions">
        <button class="astra-auth-btn" id="astra-auth-login">Sign In</button>
        <button class="astra-auth-btn" id="astra-auth-signup">Sign Up</button>
      </div>
      <div class="astra-auth-actions">
        <button class="astra-auth-btn" id="astra-auth-logout">Sign Out</button>
      </div>
      <div class="astra-auth-hint">Press <b>Alt+A</b> to open this panel.</div>
    </div>`;
  document.addEventListener('DOMContentLoaded', function(){ document.body.appendChild(modal); });

  function open(){ modal.classList.add('show'); }
  function close(){ modal.classList.remove('show'); }
  document.addEventListener('click', (e)=>{
    if (e.target.closest('.astra-auth-close') || e.target.classList.contains('astra-auth-backdrop')) close();
  });

  async function signUp(email, password){
    const sb = ensureClient(); if(!sb) return;
    const { data, error } = await sb.auth.signUp({ email, password });
    if (error) { alert(error.message); return null; }
    return data;
  }
  async function signIn(email, password){
    const sb = ensureClient(); if(!sb) return;
    const { data, error } = await sb.auth.signInWithPassword({ email, password });
    if (error) { alert(error.message); return null; }
    return data;
  }
  async function signOut(){
    const sb = ensureClient(); if(!sb) return;
    await sb.auth.signOut();
    window.dispatchEvent(new CustomEvent('astra:user-changed', { detail: null }));
  }
  async function getUser(){
    const sb = ensureClient(); if(!sb) return null;
    const { data: { user } } = await sb.auth.getUser();
    return user;
  }

  document.addEventListener('keydown', (e)=>{
    if (e.altKey && (e.key==='a' || e.key==='A')) { e.preventDefault(); open(); }
  });

  // Wire buttons
  document.addEventListener('click', async (e)=>{
    if (e.target.id === 'astra-auth-login'){
      const email = document.getElementById('astra-auth-email').value.trim();
      const pass = document.getElementById('astra-auth-pass').value;
      const res = await signIn(email, pass);
      if (res) { close(); window.dispatchEvent(new CustomEvent('astra:user-changed', { detail: res.user })); }
    }
    if (e.target.id === 'astra-auth-signup'){
      const email = document.getElementById('astra-auth-email').value.trim();
      const pass = document.getElementById('astra-auth-pass').value;
      const res = await signUp(email, pass);
      if (res) { alert('Check your email for confirmation (if enabled).'); }
    }
    if (e.target.id === 'astra-auth-logout'){
      await signOut(); close();
    }
  });

  // Expose API
  window.AstraAuth = { open, close, signUp, signIn, signOut, getUser };

  (async () => {
    const user = await getUser();
    window.dispatchEvent(new CustomEvent('astra:user-changed', { detail: user }));
  })();
})();
