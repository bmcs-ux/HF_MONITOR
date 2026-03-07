function fmt(obj){return JSON.stringify(obj ?? {}, null, 2)}

async function post(url){await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await updateUI();}

document.getElementById('btn-monitor').onclick=()=>post('/api/control/simulate_monitor');
document.getElementById('btn-trade').onclick=()=>post('/api/control/simulate_trade');
document.getElementById('btn-reset').onclick=()=>post('/api/control/reset');

function renderPriceTable(actuals, forecasts){
  const tbody=document.getElementById('price-table');
  const keys=[...new Set([...Object.keys(actuals||{}),...Object.keys(forecasts||{})])];
  tbody.innerHTML=keys.map(k=>{
    const a=Number(actuals?.[k]??0);const f=Number(forecasts?.[k]??0);
    const d=a?((f-a)/a*100):0;
    return `<tr><td>${k}</td><td>${a||'-'}</td><td>${f||'-'}</td><td>${d.toFixed(4)}%</td></tr>`;
  }).join('');
}

function formatCountdown(seconds){
  if(seconds===null || seconds===undefined || Number.isNaN(Number(seconds))) return '-';
  const total=Math.max(0,Math.floor(Number(seconds)));
  const hrs=Math.floor(total/3600);
  const mins=Math.floor((total%3600)/60);
  const secs=total%60;
  return `${String(hrs).padStart(2,'0')}:${String(mins).padStart(2,'0')}:${String(secs).padStart(2,'0')}`;
}

function renderNewsStatus(newsStatus){
  const nextTitle=newsStatus?.next_event?.title || '-';
  const nextCountry=newsStatus?.next_event?.country || '-';
  const isRestricted=Boolean(newsStatus?.is_restricted);
  const statusLabel=isRestricted ? 'ACTIVE (trading paused by NEWS gate)' : 'INACTIVE';
  const payload={
    status: statusLabel,
    countdown_to_next_high_impact: formatCountdown(newsStatus?.seconds_to_next_event),
    next_event_title: nextTitle,
    next_event_country: nextCountry,
    next_event_time_utc: newsStatus?.next_event?.event_time_utc || '-',
    active_event: newsStatus?.active_event || null,
  };
  document.getElementById('news-status').textContent=fmt(payload);
}

async function updateUI(){
  const res=await fetch('/api/get_data');
  const data=await res.json();
  document.getElementById('summary').textContent=fmt(data.summary);
  document.getElementById('global-metrics').textContent=fmt(data.global_metrics);
  document.getElementById('financials').textContent=fmt(data.financials);
  document.getElementById('rls-health').textContent=fmt(data.rls_health);
  document.getElementById('dcc-metrics').textContent=fmt(data.dcc_metrics);
  document.getElementById('kalman-metrics').textContent=fmt(data.kalman_metrics);
  document.getElementById('consensus-metrics').textContent=fmt(data.consensus_metrics);
  document.getElementById('mean-reversion').textContent=fmt(data.mean_reversion_candidates);
  document.getElementById('news-status').textContent=fmt(data.news_status);
  renderNewsStatus(data.news_status);
  document.getElementById('trade-signals').textContent=fmt(data.trade_signals);
  document.getElementById('open-trades').textContent=fmt(data.open_trades_summary);
  document.getElementById('logs').textContent=fmt((data.logs||[]).slice(0,8));
  renderPriceTable(data.latest_actual_prices, data.rls_forecast);
}
setInterval(updateUI, 3000);
updateUI();
