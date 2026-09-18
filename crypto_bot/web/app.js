const $=id=>document.getElementById(id);
const apiOptions={headers:{Authorization:`Bearer ${window.INTRA_SESSION_TOKEN}`}};
let selectedHorizon=1;
let latestSignals=[];
let loadInFlight=false;
let backtestInFlight=false;
let sortMode='profit';
let actionFilter='ALL';
let trackedTrades=JSON.parse(localStorage.getItem('intra-tracked-trades')||'[]');
const moveDeadlines=new Map();
const expiredSignals=new Set();
const lastProbabilities=new Map();
let selectedSignal=null;
const defaultSettings={
  exchange:'auto',timeout:7,workers:8,news:true,newsLimit:8,mcpName:'',mcpUrl:'',mcpEnabled:false,aiProvider:'',aiModel:'',
  horizon1:true,horizon5:true,horizon10:true,probability:.55,fee:4,spread:2,slippage:1,watchlist:'',
  layout:'default',performance:true,newsModule:true,trades:true,corePosition:'center',poll:30,reducedMotion:false,googleClient:'',googleAuth:'',
  consoleWidth:1200,moveWidth:260,telemetryColumns:3
};
let intraSettings={...defaultSettings,...JSON.parse(localStorage.getItem('intra-settings')||'{}')};
let pollTimer=null;
const settingIds={horizon1:'setting-horizon-1',horizon5:'setting-horizon-5',horizon10:'setting-horizon-10',newsLimit:'setting-news-limit',mcpName:'setting-mcp-name',mcpUrl:'setting-mcp-url',mcpEnabled:'setting-mcp-enabled',aiProvider:'setting-ai-provider',aiModel:'setting-ai-model',corePosition:'setting-corePosition',consoleWidth:'setting-consoleWidth',moveWidth:'setting-moveWidth',telemetryColumns:'setting-telemetryColumns'};
function settingElement(key){return $(settingIds[key]||`setting-${key}`);}
function setHeroStatus(value){$('hero-value').textContent=value;$('opinion-state').textContent=value;}
function applySettings(){
  document.body.classList.remove('layout-compact','layout-wide','layout-sidebar-right','layout-hide-news','layout-hide-performance');
  if(intraSettings.layout!=='default')document.body.classList.add(`layout-${intraSettings.layout}`);
  if(intraSettings.corePosition!=='center')document.body.classList.add(`core-${intraSettings.corePosition}`);
  if(!intraSettings.performance)document.body.classList.add('layout-hide-performance');
  if(!intraSettings.newsModule)document.body.classList.add('layout-hide-news');
  $('trade-list').closest('.trade-board').style.display=intraSettings.trades?'':'none';
  document.documentElement.style.setProperty('--console-width',`${Number(intraSettings.consoleWidth)||1200}px`);
  document.documentElement.style.setProperty('--move-width',`${Number(intraSettings.moveWidth)||260}px`);
  document.documentElement.style.setProperty('--telemetry-columns',String(Math.max(1,Math.min(5,Number(intraSettings.telemetryColumns)||3))));
  $('symbol-detail').style.setProperty('--core-position',intraSettings.corePosition);
  document.documentElement.style.setProperty('--core-position',intraSettings.corePosition);
  document.body.classList.toggle('reduced-motion',!!intraSettings.reducedMotion);
}
function schedulePoll(){clearInterval(pollTimer);pollTimer=setInterval(load,Math.max(15,Number(intraSettings.poll)||30)*1000);}
function fillSettings(){
  Object.entries(intraSettings).forEach(([key,value])=>{const element=settingElement(key);if(!element)return;if(element.type==='checkbox')element.checked=Boolean(value);else element.value=value;});
  $('google-status').textContent=localStorage.getItem('intra-google-status')||'Not connected';
  ['consoleWidth','moveWidth','telemetryColumns'].forEach(key=>{const element=settingElement(key);const output=$(`${element.id}-value`);if(output)output.textContent=element.value+(key==='telemetryColumns'?' cols':' px');});
}
function readSettings(){
  const next={...intraSettings};
  Object.keys(defaultSettings).forEach(key=>{const element=settingElement(key);if(!element)return;next[key]=element.type==='checkbox'?element.checked:element.type==='number'?Number(element.value):element.value;});
  return next;
}
function openSettings(){fillSettings();$('settings-dialog').showModal();}
function saveSettings(){
  intraSettings=readSettings();localStorage.setItem('intra-settings',JSON.stringify(intraSettings));applySettings();schedulePoll();
  $('settings-saved').textContent='SETTINGS SAVED';setTimeout(()=>$('settings-saved').textContent='',1800);
}
applySettings();
function signalKey(signal){return `${signal.symbol}:${signal.horizon_minutes}:${signal.action}`;}
function signalBaseKey(signal){return `${signal.symbol}:${signal.horizon_minutes}`;}
function displaySymbol(symbol){
  let value=String(symbol||'').toUpperCase();
  if(value.startsWith('XX')&&value.endsWith('ZUSD'))value=value.slice(1,-4)+'USD';
  for(const quote of ['USDT','USDC','USD'])if(value.endsWith(quote))return value.slice(0,-quote.length);
  return value;
}
function saveTrackedTrades(){localStorage.setItem('intra-tracked-trades',JSON.stringify(trackedTrades));}
function renderTrackedTrades(){
  $('trade-list').innerHTML=trackedTrades.length?trackedTrades.map((trade,index)=>{
    const signal=latestSignals.find(item=>item.symbol===trade.symbol);
    const price=signal?numeric(signal.price):null;
    const hit=price!==null&&((trade.action==='LONG'&&price<=trade.stop)||(trade.action==='SHORT'&&price>=trade.stop));
    const target=price!==null&&((trade.action==='LONG'&&price>=trade.take)||(trade.action==='SHORT'&&price<=trade.take));
    const status=hit?'STOP ALERT':target?'TAKE PROFIT ALERT':trade.status;
    return `<div class="trade-row ${hit||target?'trade-alert':'trade-safe'}"><b>${trade.symbol}</b> ${trade.action} • ${status}<br>ENTRY ${formatPrice(trade.entry)} • STOP ${formatPrice(trade.stop)} • TAKE ${formatPrice(trade.take)}<br><button class="remove-trade" data-trade-index="${index}">REMOVE</button></div>`;
  }).join(''):'No tracked trades.';
  document.querySelectorAll('.remove-trade').forEach(button=>button.onclick=()=>{trackedTrades.splice(Number(button.dataset.tradeIndex),1);saveTrackedTrades();renderTrackedTrades();});
}
const hologram=$('hologram');
const holoState=$('holo-state');
let holoHidden=localStorage.getItem('intra-hologram-hidden')==='true';
if(holoHidden) hologram.classList.add('hidden');
$('holo-hide').onclick=()=>{holoHidden=!holoHidden; hologram.classList.toggle('hidden',holoHidden); localStorage.setItem('intra-hologram-hidden',holoHidden);};
function setHologramState(state){
  const labels={scanning:'ASSEMBLING... 89%',listening:'LISTENING // LIVE',warning:'ALERT // PROVIDER DEGRADED',thinking:'PROCESSING // SCANNING'};
  holoState.textContent=labels[state]||state.toUpperCase();
  $('holo-system-text').textContent=state==='warning'?'CONTOUR MATRIX // DEGRADED':'CONTOUR MATRIX // ONLINE';
  hologram.dataset.state=state;
}
document.addEventListener('mousemove',event=>{
  if(holoHidden||window.matchMedia('(prefers-reduced-motion: reduce)').matches)return;
  const x=Math.max(-12,Math.min(12,(event.clientX/window.innerWidth-.5)*18));
  const y=Math.max(-8,Math.min(8,(event.clientY/window.innerHeight-.5)*10));
  hologram.style.setProperty('--gaze-x',`${x}deg`);hologram.style.setProperty('--gaze-y',`${y}deg`);
});
const defaultInfo='NO CREDENTIALS • NO LIVE ORDERS • PAPER MODE';
const savedInfo=localStorage.getItem('crypto-bot-safety-info')||defaultInfo;
$('mode-info').textContent=savedInfo;
$('edit-info').onclick=()=>{ $('info-input').value=savedInfo; $('info-dialog').showModal(); };
$('save-info').onclick=()=>{ const value=$('info-input').value.trim()||defaultInfo; localStorage.setItem('crypto-bot-safety-info',value); $('mode-info').textContent=value; };
let hiddenSymbols=new Set((localStorage.getItem('crypto-bot-hidden-symbols')||'').split(',').map(value=>value.trim().toUpperCase()).filter(Boolean));
$('hide-symbols-btn').onclick=()=>{ $('hidden-symbols-input').value=[...hiddenSymbols].join(', '); $('symbols-dialog').showModal(); };
$('save-hidden-symbols').onclick=()=>{
  hiddenSymbols=new Set($('hidden-symbols-input').value.split(',').map(value=>value.trim().toUpperCase()).filter(Boolean));
  localStorage.setItem('crypto-bot-hidden-symbols',[...hiddenSymbols].join(','));
  render(latestSignals);
};
function clock(){ $('clock').textContent=new Date().toISOString().slice(11,19)+' UTC'; }
setInterval(clock,1000);clock();
function render(signals){
  latestSignals=signals;
  renderTrackedTrades();
  const evaluated=signals.map(signal=>expiredSignals.has(signalBaseKey(signal))&&signal.action!=='NO_TRADE'
    ? {...signal, action:'NO_TRADE', expected_pnl:0, reason:'time window expired; awaiting fresh reevaluation'}
    : signal);
  const visible=evaluated.filter(s=>s.horizon_minutes===selectedHorizon&&!hiddenSymbols.has(String(s.symbol).toUpperCase())&&(actionFilter==='ALL'||(actionFilter==='NO_TRADE'?s.action!=='NO_TRADE':s.action===actionFilter))).sort((a,b)=>{
    if(sortMode==='symbol')return String(a.symbol).localeCompare(String(b.symbol));
    if(sortMode==='probability')return numeric(b.probability)-numeric(a.probability);
    return numeric(b.expected_pnl??b.expected_return)-numeric(a.expected_pnl??a.expected_return);
  });
  const counts={LONG:0,SHORT:0,NO_TRADE:0}; visible.forEach(s=>counts[s.action]++);
  $('long-count').textContent=counts.LONG;$('short-count').textContent=counts.SHORT;$('neutral-count').textContent=counts.NO_TRADE;
  const currentKeys=new Set(signals.filter(s=>!hiddenSymbols.has(String(s.symbol).toUpperCase())).map(signalKey));
  for(const key of moveDeadlines.keys())if(!currentKeys.has(key))moveDeadlines.delete(key);
  signals.forEach(s=>{
    const key=signalKey(s);
    if(!moveDeadlines.has(key)||moveDeadlines.get(key)<=Date.now()){
      moveDeadlines.set(key,Date.now()+s.horizon_minutes*60*1000);
    }
  });
  const ranked=evaluated.filter(s=>s.action!=='NO_TRADE'&&!hiddenSymbols.has(String(s.symbol).toUpperCase())&&(actionFilter==='ALL'||actionFilter===s.action)).sort((a,b)=>{
    const aScore=numeric(a.probability)*Math.abs(numeric(a.expected_return)||0);
    const bScore=numeric(b.probability)*Math.abs(numeric(b.expected_return)||0);
    return bScore-aScore;
  }).slice(0,5);
  $('move-list').innerHTML=ranked.map((s,index)=>{
    const probability=numeric(s.probability), expected=numeric(s.expected_pnl??s.expected_return);
    const key=signalKey(s);
    const action=s.action.replace('_',' ');
    return `<div class="move-item move-${s.action.toLowerCase()}" data-move-key="${key}"><div class="move-rank">0${index+1}</div><div><div class="move-symbol">${displaySymbol(s.symbol)} <span class="provider">${action}</span></div><div class="move-detail">${probability===null?'--':(probability*100).toFixed(2)+'% likely'} • ${expected===null?'--':(expected*100).toFixed(2)+'% expected'} • ${s.horizon_minutes}M frame</div></div><div class="move-eta" data-deadline="${moveDeadlines.get(key)}">${formatCountdown(moveDeadlines.get(key)-Date.now())}<br><small>EST. WINDOW</small></div></div>`;
  }).join('')||'<div class="mini-log">No actionable calls in the current scan.</div>';
  $('move-updated').textContent=new Date().toLocaleTimeString();
  $('signals').innerHTML=visible.map((s,index)=>{const pct=numeric(s.percent_change), delta=numeric(s.price_change), probability=numeric(s.probability), expected=numeric(s.expected_pnl??s.expected_return), cost=numeric(s.estimated_cost), trigger=numeric(s.trigger_price), stop=numeric(s.stop_loss), target=numeric(s.take_profit); const direction=pct===null?'flat':pct>=0?'up':'down'; const key=signalKey(s); const probabilityChanged=lastProbabilities.has(key)&&lastProbabilities.get(key)!==probability; lastProbabilities.set(key,probability); const copyText=`${displaySymbol(s.symbol)} ${s.action} | TRIGGER ${formatPrice(trigger)} | STOP LOSS ${formatPrice(stop)} | TAKE PROFIT ${formatPrice(target)}`; return `<article class="signal ${s.action.toLowerCase().replace('_','-')}" tabindex="0" data-signal-index="${index}"><div class="signal-top"><div class="sym">${displaySymbol(s.symbol)}</div><div class="provider">${s.provider||'market'}</div></div><div class="price">$${formatPrice(s.price)}</div><div class="change ${direction}">${pct===null?'LIVE DATA PENDING':`${pct>=0?'+':''}${(pct*100).toFixed(2)}%`} <span>${delta===null?'--':`${delta>=0?'+':''}${formatPrice(delta)}`}</span></div><div class="stats"><span>VOL<br><b>${formatVolume(s.volume)}</b></span><span>LIKELY<br><b class="${probabilityChanged?'probability-live':''}" title="Live model probability">${probability===null?'--':(probability*100).toFixed(2)+'%'}</b></span><span>HORIZON<br><b>${s.horizon_minutes}M</b></span><span>TIME TO MOVE<br><b class="telemetry-countdown" data-signal-key="${key}" data-deadline="${moveDeadlines.get(key)}">${formatCountdown(moveDeadlines.get(key)-Date.now())}</b></span></div><div class="action">${s.action.replace('_',' ')}</div><div class="order-levels"><span>TRIGGER<b>${formatPrice(trigger)}</b><button class="copy-one" data-copy="${encodeURIComponent(formatPrice(trigger))}">COPY</button></span><span>STOP LOSS<b>${formatPrice(stop)}</b><button class="copy-one" data-copy="${encodeURIComponent(formatPrice(stop))}">COPY</button></span><span>TAKE PROFIT<b>${formatPrice(target)}</b><button class="copy-one" data-copy="${encodeURIComponent(formatPrice(target))}">COPY</button></span><button class="copy-levels" data-copy="${encodeURIComponent(copyText)}">COPY LEVELS</button></div><div class="meta">EXPECTED PNL ${expected===null?'--':`${expected>=0?'+':''}${(expected*100).toFixed(2)+'%'}`} • COST ${cost===null?'--':(cost*10000).toFixed(1)+' BPS'}</div></article>`;}).join('')||'<div class="mini-log">No market signals available. Check exchange connectivity.</div>';
  document.querySelectorAll('.signal[data-signal-index]').forEach(card=>{
    card.onclick=event=>{if(event.target.closest('.copy-levels'))return;showDetail(visible[Number(card.dataset.signalIndex)]);};
    card.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();showDetail(visible[Number(card.dataset.signalIndex)]);}};
  });
  document.querySelectorAll('.copy-levels').forEach(button=>button.onclick=async()=>{await navigator.clipboard.writeText(decodeURIComponent(button.dataset.copy));button.textContent='COPIED';setTimeout(()=>button.textContent='COPY LEVELS',1200);});
  document.querySelectorAll('.copy-one').forEach(button=>button.onclick=async event=>{
    event.stopPropagation();
    await navigator.clipboard.writeText(decodeURIComponent(button.dataset.copy));
    button.textContent='DONE';
    setTimeout(()=>button.textContent='COPY',1200);
  });
}
function showDetail(signal){
  selectedSignal=signal;
  const expected=numeric(signal.expected_pnl??signal.expected_return);
  $('detail-symbol').textContent=signal.symbol;
  $('detail-action').textContent=signal.action.replace('_',' ');
  $('detail-action').className=`symbol-detail-action ${signal.action.toLowerCase()}`;
  $('detail-horizon').textContent=`${signal.horizon_minutes}M`;
  $('detail-probability').textContent=`${(numeric(signal.probability)*100).toFixed(2)}%`;
  $('detail-trigger').textContent=formatPrice(signal.trigger_price);
  $('detail-stop').textContent=formatPrice(signal.stop_loss);
  $('detail-target').textContent=formatPrice(signal.take_profit);
  $('detail-pnl').textContent=expected===null?'--':`${expected>=0?'+':''}${(expected*100).toFixed(2)}%`;
  $('detail-indicators').textContent=`ADX ${formatPrice(signal.adx)} • RSI ${formatPrice(signal.rsi)} • MACD ${formatPrice(signal.macd)} • ATR ${formatPrice(signal.atr_ratio)} • VWAP ${formatPrice(signal.vwap_distance)} • PROVIDER ${(signal.provider||'market').toUpperCase()}`;
  $('symbol-detail').classList.add('open');
}
$('track-trade').onclick=()=>{
  if(!selectedSignal||selectedSignal.action==='NO_TRADE')return;
  $('trade-status').value='OPEN ORDER';
  $('trade-entry').value=selectedSignal.trigger_price;
  $('trade-stop').value=selectedSignal.stop_loss;
  $('trade-take').value=selectedSignal.take_profit;
  $('trade-dialog').showModal();
};
$('save-trade').onclick=event=>{
  event.preventDefault();
  const entry=numeric($('trade-entry').value), stop=numeric($('trade-stop').value), take=numeric($('trade-take').value);
  if(!selectedSignal||entry===null||stop===null||take===null||entry<=0||stop<=0||take<=0){
    $('last-event').textContent='TRADE TRACKING BLOCKED // COMPLETE ALL PRICES';return;
  }
  trackedTrades=trackedTrades.filter(trade=>trade.symbol!==selectedSignal.symbol||trade.horizon_minutes!==selectedSignal.horizon_minutes);
  trackedTrades.push({symbol:selectedSignal.symbol,horizon_minutes:selectedSignal.horizon_minutes,action:selectedSignal.action,status:$('trade-status').value,entry,stop,take});
  saveTrackedTrades();renderTrackedTrades();$('trade-dialog').close();
  $('last-event').textContent=`TRACKING ${selectedSignal.symbol} // ${$('trade-status').value}`;
};
function breakoutTicket(signal){
  const quantity=numeric($('breakout-quantity-input')?.value)||1;
  return [
    `SYMBOL: ${signal.symbol}`,
    `SIDE: ${signal.action}`,
    `TIMEFRAME: ${signal.horizon_minutes}M`,
    `TRIGGER: ${formatPrice(signal.trigger_price)}`,
    `QUANTITY: ${quantity}`,
    `STOP LOSS: ${formatPrice(signal.stop_loss)}`,
    `TAKE PROFIT: ${formatPrice(signal.take_profit)}`,
    `REFERENCE PRICE: ${formatPrice(signal.price)}`,
    'MODE: MANUAL CONFIRMATION REQUIRED',
  ].join('\n');
}
$('breakout-ticket').onclick=()=>{
  if(!selectedSignal||selectedSignal.action==='NO_TRADE')return;
  $('breakout-ticket-text').textContent=breakoutTicket(selectedSignal);
  $('breakout-quantity-input').value=localStorage.getItem('intra-breakout-quantity')||'1';
  $('breakout-url-input').value=localStorage.getItem('intra-breakout-url')||'';
  $('breakout-dialog').showModal();
};
$('breakout-copy').onclick=async event=>{
  event.preventDefault();
  if(!selectedSignal)return;
  const quantity=numeric($('breakout-quantity-input').value);
  if(quantity===null||quantity<=0){$('last-event').textContent='BREAKOUT HANDOFF BLOCKED // ENTER POSITIVE QUANTITY';return;}
  localStorage.setItem('intra-breakout-quantity',String(quantity));
  await navigator.clipboard.writeText(breakoutTicket(selectedSignal));
  $('last-event').textContent='BREAKOUT FORM VALUES COPIED // PASTE INTO ORDER TICKET';
};
$('breakout-open').onclick=async event=>{
  event.preventDefault();
  const url=$('breakout-url-input').value.trim();
  if(!/^https:\/\//i.test(url)){ $('last-event').textContent='BREAKOUT HANDOFF BLOCKED // HTTPS URL REQUIRED'; return; }
  const quantity=numeric($('breakout-quantity-input').value);
  if(quantity===null||quantity<=0){$('last-event').textContent='BREAKOUT HANDOFF BLOCKED // ENTER POSITIVE QUANTITY';return;}
  localStorage.setItem('intra-breakout-quantity',String(quantity));
  localStorage.setItem('intra-breakout-url',url);
  await navigator.clipboard.writeText(breakoutTicket(selectedSignal));
  window.open(url,'_blank','noopener,noreferrer');
  $('breakout-dialog').close();
  $('last-event').textContent='BREAKOUT TICKET COPIED // REVIEW AND SUBMIT MANUALLY';
};
$('detail-close').onclick=()=>{$('symbol-detail').classList.remove('open');selectedSignal=null;};
function formatCountdown(milliseconds){
  const seconds=Math.max(0,Math.ceil(milliseconds/1000));
  const minutes=Math.floor(seconds/60);
  return `T−${String(minutes).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;
}
function updateMoveCountdowns(){
  let expired=false;
  document.querySelectorAll('.move-eta[data-deadline]').forEach(element=>{
    const remaining=Number(element.dataset.deadline)-Date.now();
    element.firstChild.textContent=formatCountdown(remaining);
    if(remaining<=0){
      const key=element.closest('.move-item')?.dataset.moveKey;
      const signal=latestSignals.find(item=>signalKey(item)===key);
      if(signal&&!expiredSignals.has(signalBaseKey(signal))){expiredSignals.add(signalBaseKey(signal));expired=true;}
    }
  });
  document.querySelectorAll('.telemetry-countdown[data-deadline]').forEach(element=>{
    const remaining=Number(element.dataset.deadline)-Date.now();
    element.textContent=formatCountdown(remaining);
    if(remaining<=0){
      const signal=latestSignals.find(item=>signalKey(item)===element.dataset.signalKey);
      if(signal&&!expiredSignals.has(signalBaseKey(signal))){expiredSignals.add(signalBaseKey(signal));expired=true;}
    }
  });
  if(expired){
    render(latestSignals);
    $('last-event').textContent='TIME WINDOW EXPIRED // REEVALUATING';
    load();
  }
}
function numeric(value){const parsed=Number(value);return Number.isFinite(parsed)?parsed:null;}
function formatPrice(value){const parsed=numeric(value);if(parsed===null)return'--';return parsed.toLocaleString(undefined,{minimumFractionDigits:0,maximumFractionDigits:4});}
function formatVolume(value){const parsed=numeric(value);if(parsed===null||parsed<=0)return'--';if(parsed>=1e9)return(parsed/1e9).toFixed(2)+'B';if(parsed>=1e6)return(parsed/1e6).toFixed(2)+'M';if(parsed>=1e3)return(parsed/1e3).toFixed(2)+'K';return parsed.toFixed(2);}
async function load(){
  if(loadInFlight||backtestInFlight)return;
  loadInFlight=true;
  setHologramState('scanning'); setHeroStatus('SCANNING'); $('last-event').textContent='Requesting public market data...';
  try{
    const [status,response,newsResponse]=await Promise.all([fetch('/api/status',apiOptions),fetch('/api/scan',apiOptions),fetch('/api/news',apiOptions)]);
    const meta=await status.json(), data=await response.json(), newsData=await newsResponse.json();
    $('exchange').textContent=meta.exchange.toUpperCase()+' ROUTE';$('cost').textContent=meta.cost_bps+' BPS ROUND TRIP';
    const outcomes=meta.prediction_outcomes||{}; let right=0,wrong=0;
    const horizonText=[1,5,10].map(h=>{const item=outcomes[String(h)]||{right:0,wrong:0}; right+=item.right;wrong+=item.wrong; return `${h}M ${item.right}✓/${item.wrong}✕`;}).join(' • ');
    $('right-count').textContent=right;$('right-total').textContent=right;
    $('wrong-count').textContent=wrong;$('wrong-total').textContent=wrong;$('review-horizons').textContent=horizonText;
    const performance=meta.paper_performance||{};
    $('profit-grid').innerHTML=[1,5,10].map(h=>{
      const item=performance[String(h)]||{net_return:0,closed_calls:0};
      const value=numeric(item.net_return);
      const cls=value>0?'profit-positive':value<0?'profit-negative':'';
      return `<div class="profit-cell"><small>${h}M • ${item.closed_calls||0} CLOSED</small><b class="${cls}">${value===null?'--':`${value>=0?'+':''}${(value*100).toFixed(2)}%`}</b></div>`;
    }).join('');
    const news=newsData.news||{};
    $('news-direction').textContent=`NEWS SCOUT // ${news.direction||'UNAVAILABLE'} // ${news.headline_count||0} HEADLINES`;
    $('news-list').textContent=(news.headlines||[]).slice(0,3).map(item=>`${item.bias>0?'▲':item.bias<0?'▼':'•'} ${item.title}`).join(' | ')||'No public headlines available.';
    const lead=signals=>signals.filter(s=>s.action!=='NO_TRADE').sort((a,b)=>Number(b.probability)-Number(a.probability))[0];
    const top=lead(data.signals||[]);
    $('opinion-text').textContent=top
      ? `The strongest paper setup is ${top.action} ${top.symbol} over ${top.horizon_minutes}M at ${(Number(top.probability)*100).toFixed(2)}% model likelihood. ${news.disclaimer||'Public headline context is supplemental.'}`
      : 'No actionable paper setup currently clears the cost threshold. Public headlines are context only.';
    if(data.error) throw new Error(data.error);
    data.signals.forEach(signal=>expiredSignals.delete(signalBaseKey(signal)));
    render(data.signals);setHologramState('listening');setHeroStatus('READY');$('last-event').textContent='LIVE '+new Date().toLocaleTimeString();
    $('updated').textContent='UPDATED '+new Date().toLocaleTimeString();$('mini-log').textContent='Market telemetry synchronized.';
  }catch(error){setHologramState('warning');setHeroStatus('DEGRADED');$('last-event').textContent='PUBLIC PROVIDER UNAVAILABLE';$('mini-log').textContent='The app is running. Exchange access is blocked or unavailable; TradingView webhooks remain independent.';}
  finally{loadInFlight=false;}
}
$('scan-btn').onclick=load;$('refresh-btn').onclick=load;load();
$('archive-btn').onclick=async()=>{
  if(!window.confirm('Archive all current paper predictions and reset wins/profits for a fresh scan?'))return;
  $('archive-btn').disabled=true;$('last-event').textContent='ARCHIVING + ANALYZING...';
  try{
    const response=await fetch('/api/archive-reset',{method:'POST',...apiOptions,body:'{}'});
    const data=await response.json();
    if(!response.ok||data.error)throw new Error(data.error||'archive failed');
    render(data.signals||[]);
    $('last-event').textContent=`FRESH START // ARCHIVED ${data.archive.prediction_count} CALLS`;
    await load();
  }catch(error){$('last-event').textContent='ARCHIVE FAILED // '+error.message;}
  finally{$('archive-btn').disabled=false;}
};
$('backtest-btn').onclick=async()=>{
  if(backtestInFlight)return;
  backtestInFlight=true;$('scan-btn').disabled=true;$('refresh-btn').disabled=true;$('backtest-btn').disabled=true;$('last-event').textContent='BACKTESTING ALL LISTED PAIRS // MARKET SCAN PAUSED...';
  try{
    const response=await fetch('/api/backtest',{method:'POST',...apiOptions,body:'{}'}); const data=await response.json();
    if(!response.ok||data.error)throw new Error(data.error||'backtest failed');
    const rows=[1,5,10].map(h=>{const r=data.results[String(h)]||{};return `${h}M: ${r.calls||0} calls, ${(Number(r.accuracy||0)*100).toFixed(1)}% right, ${(Number(r.net_return||0)*100).toFixed(2)}% net`;});
    const pairCount=Object.keys(data.results.per_symbol||{}).length;
    $('learning-note').textContent=`Reviewed ${pairCount} pairs. ${rows.join(' • ')}. Per-pair calibration was updated from paper outcomes; no live orders were placed.`;
    $('learning-dialog').showModal();$('last-event').textContent='BACKTEST COMPLETE // STRATEGY UPDATED';
  }catch(error){$('last-event').textContent='BACKTEST FAILED // '+error.message;}
  finally{backtestInFlight=false;$('scan-btn').disabled=false;$('refresh-btn').disabled=false;$('backtest-btn').disabled=false;}
};
document.querySelectorAll('.timeframe').forEach(button=>button.onclick=()=>{
  selectedHorizon=Number(button.dataset.horizon);
  document.querySelectorAll('.timeframe').forEach(item=>item.classList.toggle('active',item===button));
  render(latestSignals);
});
function setActionFilter(filter){
  actionFilter=filter;
  document.querySelectorAll('.telemetry-filter').forEach(button=>button.classList.toggle('active',button.dataset.filter===filter));
  render(latestSignals);
}
document.querySelectorAll('.telemetry-filter').forEach(button=>button.onclick=()=>setActionFilter(button.dataset.filter));
$('sort-signals').onchange=event=>{sortMode=event.target.value;render(latestSignals);};
schedulePoll();
setInterval(updateMoveCountdowns,1000);

$('settings-launch').onclick=openSettings;
$('settings-close').onclick=()=>$('settings-dialog').close();
$('settings-save').onclick=saveSettings;
$('settings-reset-all').onclick=()=>{intraSettings={...defaultSettings};fillSettings();};
$('apply-layout').onclick=()=>{intraSettings=readSettings();localStorage.setItem('intra-settings',JSON.stringify(intraSettings));applySettings();$('settings-saved').textContent='LAYOUT APPLIED';};
$('settings-reset').onclick=()=>{
  ['intra-settings','intra-google-status','intra-tracked-trades','intra-hologram-hidden','intra-breakout-quantity','intra-breakout-url'].forEach(key=>localStorage.removeItem(key));
  location.reload();
};
$('google-login').onclick=()=>{
  const url=String($('setting-googleAuth').value||'').trim();
  if(!/^https:\/\//i.test(url)){
    localStorage.setItem('intra-google-status','OAuth setup required');
    $('google-status').textContent='Configure an HTTPS OAuth start URL first';
    return;
  }
  localStorage.setItem('intra-google-status','OAuth handoff opened');
  $('google-status').textContent='OAuth handoff opened';
  window.open(url,'_blank','noopener,noreferrer');
};
const settingsTitles={connections:'Connections & providers',signals:'Signals & risk',dashboard:'Dashboard layout',account:'Account & safety'};
document.querySelectorAll('.settings-tab').forEach(tab=>tab.onclick=()=>{
  const section=tab.dataset.settingsTab;
  document.querySelectorAll('.settings-tab').forEach(item=>item.classList.toggle('active',item===tab));
  document.querySelectorAll('.settings-page').forEach(page=>page.hidden=page.dataset.settingsPage!==section);
  $('settings-title').textContent=settingsTitles[section];
});
['consoleWidth','moveWidth','telemetryColumns'].forEach(key=>{const element=settingElement(key);element.oninput=()=>{const output=$(`${element.id}-value`);if(output)output.textContent=element.value+(key==='telemetryColumns'?' cols':' px');};});
