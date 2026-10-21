/* OpenSAE Research Lab — interactive research workspaces.
   Read-only browser: no model execution and no invented model measurements.
   All visible results come from validated, explicitly sourced evidence packs.
*/
(() => {
  'use strict';
  const TYPES = ['sites', 'metrics', 'effects', 'contexts', 'rss', 'coalitions'];
  const MAX_UPLOAD = 25 * 1024 * 1024;
  const state = { pack: null, route: 'effects', effectId: '', effectTarget: 'ablate', edit: 'ablate', removed: false, prediction: '', revealed: false,
    contextMode: 'grid', contextId: '', mask: 0, prevMask: null, contextBit: 0, rssId: '', rssSetting: 'local_format', variantId: '', coalitionId: '', coalitionMask: 0,
    modelProtocol: '', modelX: '', modelY: '', modelFilter: '', modelSelected: '' };
  const SETTING_COPY = {
    local_format: 'Local format — the source item is rewrapped in a different local format.',
    instruction_wrapper: 'Instruction wrapper — an instruction wrapper is placed around the source item.',
    context_wrapper: 'Context wrapper — a context wrapper is placed around the source item.'
  };
  const $ = id => document.getElementById(id);
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  const fmt = (n, digits=4) => Number.isFinite(Number(n)) && n !== null && n !== '' ? new Intl.NumberFormat('en-US',{maximumFractionDigits:digits}).format(Number(n)) : '—';
  const empty = (what, text) => `<div class="empty"><strong>${esc(what)}</strong>${esc(text)}</div>`;
  const safeURL = v => typeof v === 'string' && /^https:\/\/(github\.com|huggingface\.co|arxiv\.org)\//i.test(v) ? v : '';
  const link = (text, url) => safeURL(url) ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(text)} ↗</a>` : '';
  const cases = t => state.pack?.[t] || [];
  const find = (t, id) => cases(t).find(c => c.id === id);
  const metricValue = m => (typeof m.value === 'number' && Number.isFinite(m.value) ? m.value : null);
  const src = c => c?.evidence_url || c?.source_url || state.pack?.meta?.source || '';
  const provenance = c => c ? `<div><strong>${esc(c.model || c.site_id || c.id)}</strong>${c.sae ? ' · '+esc(c.sae):''}${c.layer !== undefined?' · L'+esc(c.layer):''}</div><div>${esc(c.protocol_id || 'Protocol unspecified')}${c.sample_count?' · n='+esc(c.sample_count):''}</div><div>${link('Source record', src(c))}</div>` : '';
  function status(message, error=false) { const node=$('flash'); node.textContent=message; node.classList.toggle('error',error); node.hidden=false; }
  function options(sel, list, selected, placeholder='Select an experiment') {
    if (!sel) return '';
    sel.replaceChildren();
    if(!list.length){const o=new Option('No measured records available','');sel.append(o);sel.disabled=true;return '';}
    sel.disabled=false;
    const ids = list.map(x=>String(x.value));
    const value=ids.includes(String(selected))?String(selected):String(list[0].value);
    for(const item of list){const opt=new Option(item.label,String(item.value));sel.append(opt);}
    sel.value=value;return value;
  }
  function metadata(c) { return provenance(c); }
  const standardSite = (c) => c?.model ? c.model + (c.sae ? ' / '+c.sae : '') + (c.layer!=null ? ' / L'+c.layer : '') : c?.site_id || 'Measured site';
  function clearPrediction(){state.prediction='';state.revealed=false;}
  function validate(p) {
    if(!p || p.schema !== 'opensae.lab.evidence.v1')throw new Error('Unsupported evidence schema. Expected opensae.lab.evidence.v1');
    if(!p.meta || typeof p.meta !== 'object')throw new Error('Pack is missing metadata');
    for(const t of TYPES){if(!Array.isArray(p[t]))throw new Error(`Evidence pack requires ${t} array`);const seen=new Set();
      for(const r of p[t]){if(typeof r !== 'object'||r===null)throw new Error(`Invalid ${t} record`);
        if(t!=='sites'&&t!=='metrics'&&(!r.id||typeof r.id!=='string'))throw new Error(`Each ${t} record requires an ID`);
        if(r.id){if(seen.has(r.id))throw new Error(`Duplicate ${t} ID: ${r.id}`);seen.add(r.id);}
        if(t!=='sites' && t!=='metrics' && (!r.protocol_id||!safeURL(r.evidence_url)))throw new Error(`${t} ${r.id} requires protocol_id and an approved HTTPS evidence_url`);
      }
    }
    for(const c of p.effects){if(!c.site_id||!Number.isInteger(c.feature_id)||!c.condition)throw new Error(`Effect ${c.id} needs site_id, feature_id and condition`);
      for(const field of ['original_top_tokens','edited_top_tokens']){if(c[field]!==undefined){if(!Array.isArray(c[field]))throw new Error(`Effect ${c.id}: ${field} must be an array`);
        for(const t of c[field]){if(typeof t.token!=='string'||typeof t.probability!=='number'||!Number.isFinite(t.probability)||t.probability<0||t.probability>1)throw new Error(`Invalid token probability in ${c.id}`);}}}
      if(c.measured_delta_l2!==undefined && !(typeof c.measured_delta_l2==='number' && Number.isFinite(c.measured_delta_l2) && c.measured_delta_l2>=0))throw new Error(`Invalid L2 in ${c.id}`);
    }
    for(const c of p.contexts){if(!Array.isArray(c.factors)||c.factors.length!==6||!Array.isArray(c.states))throw new Error(`Context ${c.id} needs six factors and 64 states`);
      const masks=new Set(c.states.map(x=>x.mask));if(c.states.length!==64 || masks.size!==64 || [...masks].some(m=>!Number.isInteger(m)||m<0||m>63))throw new Error(`Context ${c.id} is not a complete 64-state grid`);
      for(const r of c.states){if(r.activation!==null && (typeof r.activation!=='number'||!Number.isFinite(r.activation)))throw new Error(`Invalid activation in ${c.id}`);}
    }
    for(const c of p.coalitions){if(!Array.isArray(c.features)||c.features.length<1||c.features.length>16||!Array.isArray(c.responses))throw new Error(`Coalition ${c.id} invalid features or responses`);
      const masks=new Set();for(const r of c.responses){if(!Number.isInteger(r.mask)||r.mask<0||r.mask>=2**c.features.length||typeof r.value!=='number'||!Number.isFinite(r.value))throw new Error(`Coalition ${c.id} has invalid response`);if(masks.has(r.mask))throw new Error(`Duplicate coalition mask in ${c.id}`);masks.add(r.mask);}
    }
    for(const c of p.rss){if(!Array.isArray(c.variants))throw new Error(`RSS ${c.id} needs variants`);}
    for(const m of p.metrics){if(!m.site_id||!m.metric||typeof m.value!=='number'||!Number.isFinite(m.value)||!m.protocol_id||!safeURL(m.evidence_url))throw new Error('Each model metric needs a real source, protocol, site, metric and finite value');}
    return p;
  }
  function setPack(p){state.pack=validate(p);const total=cases('effects').length+cases('contexts').length+cases('rss').length+cases('coalitions').length;
    $('evidence-status').textContent=`${cases('metrics').length} metric records · ${total} interactive cases`;
    state.effectId=cases('effects')[0]?.id||'';state.contextId=cases('contexts')[0]?.id||'';state.rssId=cases('rss')[0]?.id||'';state.coalitionId=cases('coalitions')[0]?.id||'';
    state.rssSetting=(state.rssId.split('|')[2])||'local_format';state.coalitionMask=0;state.prevMask=null;clearPrediction();applyURL();
    const chosen=find('contexts',state.contextId);state.mask=bestMask(chosen);render();
  }
  function applyURL(){const q=new URLSearchParams(location.search);for(const [key,t] of [['effect','effects'],['context','contexts'],['rss','rss'],['coalition','coalitions']]){
    if(q.has(key)&&find(t,q.get(key)))state[key==='effect'?'effectId':key==='context'?'contextId':key==='rss'?'rssId':'coalitionId']=q.get(key);
  }}
  const CRUMBS = {effects:'Feature Effects', context:'Context Lab', models:'Compare models', interactions:'Research'};
  function route(){const r=location.hash.slice(1).split('?')[0];state.route=['effects','context','models','interactions'].includes(r)?r:'effects';
    document.querySelectorAll('.page').forEach(p=>p.hidden=p.id!==`page-${state.route}`);
    document.querySelectorAll('[data-route]').forEach(a=>{a.classList.toggle('active',a.dataset.route===state.route);a.setAttribute('aria-current',a.dataset.route===state.route?'page':'false');});
    if($('crumb')) $('crumb').textContent=CRUMBS[state.route];
    const activeTab=document.querySelector('.rail .tab.active');
    if(activeTab && activeTab.parentElement){activeTab.parentElement.scrollTo({left:Math.max(0,activeTab.offsetLeft-12)});}
    render();}
  function render(){if(!state.pack)return;document.querySelectorAll('.page').forEach(p=>p.hidden=p.id!==`page-${state.route}`);
    if(state.route==='effects')renderEffects();if(state.route==='context')renderContext();if(state.route==='models')renderModels();if(state.route==='interactions')renderCoalitions();}
  function effectTokens(c){const base=new Map((c?.original_top_tokens||[]).map(t=>[t.token,t.probability]));const edited=new Map((c?.edited_top_tokens||[]).map(t=>[t.token,t.probability]));
    const all=[...new Set([...base.keys(),...edited.keys()])];return all.map(token=>({token,base:base.has(token)?base.get(token):null,edited:edited.has(token)?edited.get(token):null}));}
  const CURATED = [
    {id:'authored:gemma1b:gemma1b_scope_resid:L13:capital', name:'A factual question'},
    {id:'authored:gemma1b:gemma1b_scope_resid:L13:sky', name:'A color question'},
    {id:'authored:gemma1b:gemma1b_scope_resid:L13:sum', name:'An arithmetic question'},
    {id:'authored:gemma1b:gemma1b_scope_resid:L13:animal', name:'Name the animal'},
    {id:'authored:pythia410m:pythia410m_mlp_topk:L12:story', name:'A story opening'}
  ];
  function publicEffects(){return cases('effects').filter(c=>c.prompt && Array.isArray(c.original_top_tokens) && c.original_top_tokens.length);}
  function curatedEffects(){const byId=new Map(publicEffects().map(c=>[c.id,c]));return CURATED.filter(item=>byId.has(item.id)).map(item=>({...item, case:byId.get(item.id)}));}
  function piece(row){const raw=String(row?.token??'');if(raw===' ')return 'space';if(raw==='\n')return 'new line';if(raw.trim()==='')return row?.label||'blank';return raw;}
  function quoted(row){return `“${piece(row)}”`;}
  function percent(value){return `${(Number(value)*100).toFixed(1)}%`;}
  function alignedRows(before, after){const afterMap=new Map(after.map(row=>[row.token,row]));const seen=new Set();const rows=[];
    before.forEach(row=>{seen.add(row.token);rows.push({token:row.token, base:row, edited:afterMap.get(row.token)||null});});
    after.forEach(row=>{if(!seen.has(row.token))rows.push({token:row.token, base:null, edited:row});});
    return rows;}
  function renderEffects(){const shown=curatedEffects();if(!shown.length){$('effects-chart').textContent='';return;}
    state.effectId=options($('effect-case'),shown.map(item=>({value:item.id,label:item.name})),state.effectId);
    const item=shown.find(row=>row.id===state.effectId)||shown[0];
    const c=item.case;state.effectId=c.id;
    const operators=(c.operators||[]).filter(op=>Array.isArray(op.edited_top_tokens)&&op.edited_top_tokens.length);
    const selected=operators.find(op=>op.operation===state.edit)||operators.find(op=>op.operation==='ablate')||operators[0];
    const before=c.original_top_tokens;
    const edited=selected.edited_top_tokens;
    const after=state.removed?edited:before;
    const rows=alignedRows(before, state.removed?edited:before);
    const probs=rows.flatMap(row=>[row.base?.probability, state.removed?row.edited?.probability:null]).filter(value=>typeof value==='number');
    const peak=Math.max(...probs, 0.001);
    const marks=new Map();
    const shared=state.removed?rows.filter(row=>row.base&&row.edited).map(row=>({token:row.token, delta:row.edited.probability-row.base.probability})):[];
    if(shared.length){const rise=shared.reduce((best,row)=>row.delta>best.delta?row:best,shared[0]);const fall=shared.reduce((best,row)=>row.delta<best.delta?row:best,shared[0]);if(rise.delta>=0.02)marks.set(rise.token,'rise');if(fall.delta<=-0.02)marks.set(fall.token,'fall');}
    const cell=(row,side)=>{const mark=marks.get(row.token)||'';const value=side==='base'?row.base:row.edited;if(!value)return `<div class="word-row"><span class="token-label">${esc(piece(row.base||row.edited))}</span><span class="bar-track"></span><span class="bar-number">not in the recorded top 5</span></div>`;
      return `<div class="word-row ${mark}"><span class="token-label">${esc(piece(value))}</span><span class="bar-track"><span class="bar-fill ${side==='edited'&&mark==='rise'?'edit-fill':'base-fill'}" style="width:${Math.max(value.probability/peak*100, value.probability>0?2:0)}%"></span></span><span class="bar-number">${percent(value.probability)}</span></div>`;};
    const topBefore=before[0];const topEdited=edited[0];
    $('effect-prompt').textContent=c.prompt;
    $('effect-meta').textContent=c.model;
    $('effect-feature').textContent=`Feature #${c.feature_id}`;
    $('effect-active').textContent='This feature was the strongest nonzero coordinate measured on this input.';
    const onColumn=`<div class="predict-col"><h3>Feature on (normal)</h3><p class="top-line">Top prediction: ${esc(quoted(topBefore))} (${percent(topBefore.probability)})</p>${rows.map(row=>cell(row,'base')).join('')}</div>`;
    const offColumn=`<div class="predict-col"><h3>Feature #${esc(c.feature_id)} off</h3><p class="top-line">Top prediction: ${esc(quoted(topEdited))} (${percent(topEdited.probability)})</p>${rows.map(row=>cell(row,'edited')).join('')}</div>`;
    $('effects-chart').innerHTML=state.removed?`<div class="before-after">${onColumn}${offColumn}</div>`:onColumn;
    const shownBefore=Math.round(topBefore.probability*1000)/10, shownAfter=Math.round(topEdited.probability*1000)/10;
    const dpp=Math.round((shownAfter-shownBefore)*10)/10;
    const same=topBefore.token===topEdited.token;
    const limit=' This shows a change for one input, not what the feature means in general.';
    const verb=dpp<0?'fell':'rose';
    const finding=!same
      ? `Turning off feature #${c.feature_id} changed the model's top next-token prediction from ${quoted(topBefore)} to ${quoted(topEdited)} on this input.${limit}`
      : Math.abs(dpp)<1
        ? `Turning off this feature had little effect on the recorded next-token predictions for this input.${limit}`
        : `The top prediction stayed ${quoted(topBefore)}, but its probability ${verb} from ${percent(topBefore.probability)} to ${percent(topEdited.probability)}.${limit}`;
    $('effect-learned-title').hidden=!state.removed;
    $('effect-footnote').hidden=!state.removed;
    $('effect-footnote').textContent=state.removed?finding:'';
    $('effect-on').setAttribute('aria-pressed', state.removed?'false':'true');
    $('effect-off').setAttribute('aria-pressed', state.removed?'true':'false');
    const legacyLabels={ablate:'Set activity to zero',scale:'Multiply activity by 0.5',add:'Add 1 to its activity',clamp:'Set its activity to 1',coalition:'Turn off the second active feature'};
    const legacy=cases('effects').filter(row=>!row.prompt);
    const othersPublic=publicEffects().filter(row=>!CURATED.some(item=>item.id===row.id));
    const otherLines=othersPublic.map(row=>{const op=(row.operators||[]).find(item=>item.operation==='ablate');const next=op?.edited_top_tokens?.[0];const first=row.original_top_tokens[0];return `<li>${esc(row.model)} · ${esc(row.prompt)} · ${esc(piece(first))} ${percent(first.probability)} → ${next?esc(piece(next))+' '+percent(next.probability):'recorded'}</li>`;}).join('');
    const legacyTable=legacy.length?`<h3>Older score-only records</h3><p>These six saved studies record how far the next-word scores moved. Their prompt wording was not stored, so they are not shown as examples. Scale used a factor of one half, add used plus one, and clamp set the activity to 1.</p><table><tbody>${legacy.map(row=>`<tr><td>${esc(row.model||row.site_id)} layer ${esc(row.layer)}</td><td>${(row.operators||[]).map(op=>`${esc(legacyLabels[op.operation]||op.operation)} ${fmt(op.logit_l2,2)}`).join(' · ')}</td></tr>`).join('')}</tbody></table>`:'';
    $('effect-evidence').innerHTML=`<p>${esc(c.protocol_id)} · run ${esc(c.run_id)} · site ${esc(c.site_id)} · feature ${esc(c.feature_id)}</p><p>Score shift for this edit: ${fmt(selected.logit_l2,4)}. Chances are a full-vocabulary softmax at the last token. Readout position: final token.</p><p>Model fingerprint ${esc(c.model_sha256||'')} · SAE fingerprint ${esc(c.sae_sha256||'')}</p><p>${link('Measurement script', src(c))}</p>${otherLines?`<h3>Other measured examples</h3><ul>${otherLines}</ul>`:''}${legacyTable}`;
    $('effect-measures').textContent='';}
  function revealEffect(){const c=find('effects',state.effectId),t=effectTokens(c).find(t=>t.token===state.effectTarget&&t.base!==null&&t.edited!==null);
    if(!t){$('effect-feedback').textContent='Select an observed token with measured probabilities.';return;}
    const diff=t.edited-t.base;const direction=diff>0?'up':diff<0?'down':'unchanged';
    const correct=(state.prediction===direction);
    $('effect-feedback').textContent=`Measured: ${t.token} went ${direction} (${diff>=0?'+':''}${fmt(diff*100,3)} percentage points). ${state.prediction?correct?'Your prediction matches the observation.':'Your prediction differs from the observation.':'This is the measured outcome.'}`;
  }
  function controlGridCase(){return find('contexts',state.contextId)}
  function renderContext(){const rss=state.contextMode==='rss';$('context-grid-view').hidden=rss;$('context-rss-view').hidden=!rss;
    $('context-grid-tab').classList.toggle('active',!rss);$('context-rss-tab').classList.toggle('active',rss);
    $('context-grid-tab').setAttribute('aria-selected',String(!rss));$('context-rss-tab').setAttribute('aria-selected',String(rss));
    if(rss)renderRSS();else renderGrid();}
  function bestMask(c){let best=0,value=-Infinity;for(const row of c?.states||[]){if(typeof row.activation==='number'&&row.activation>value){value=row.activation;best=row.mask;}}return Number.isFinite(value)?best:0;}
  function pairedMean(rows, bit){const diffs=[];for(const row of rows){if(row.mask&(1<<bit)){const other=rows.find(item=>item.mask===(row.mask^(1<<bit)));if(other&&typeof row.activation==='number'&&typeof other.activation==='number')diffs.push(row.activation-other.activation);}}return diffs;}
  function renderGrid(){state.contextId=options($('context-case'),cases('contexts').map(c=>({value:c.id,label:`Feature ${c.feature_id} · ${standardSite(c)} · var ${fmt(c.variance,4)}`})),state.contextId);
    const c=controlGridCase(),factors=c?.factors||[];
    $('context-factor-grid').innerHTML=factors.map((f,i)=>{const on=!!(state.mask&(1<<i));const levels=f.levels||['off','on'];const changed=state.prevMask!==null&&((state.mask^state.prevMask)&(1<<i));
      return `<button class="factor ${on?'on':''} ${changed?'changed':''}" type="button" data-factor="${i}" aria-pressed="${on}"><strong>${esc(f.label||f.id)}</strong><span class="toggle">${esc(levels[on?1:0])}</span></button>`;}).join('');
    document.querySelectorAll('[data-factor]').forEach(btn=>btn.addEventListener('click',()=>{const bit=Number(btn.dataset.factor);state.prevMask=state.mask;state.contextBit=bit;state.mask^=(1<<bit);renderGrid();}));
    $('mask-id').textContent=String(state.mask).padStart(2,'0')+' / 63';
    const rows=new Map((c?.states||[]).map(r=>[r.mask,r]));const selected=rows.get(state.mask);$('context-text').textContent=selected?.prompt||selected?.sample_id||'';
    const valid=(c?.states||[]).filter(r=>typeof r.activation==='number');const values=valid.map(r=>r.activation);const min=Math.min(...values),max=Math.max(...values),span=Math.max(max-min,1e-9);
    $('grid-count').textContent=c?`${valid.length} / 64 recorded activations`:'';
    $('state-legend').textContent=c?'Darker cells are higher recorded activation on this feature.':'';
    $('context-heatmap').innerHTML=Array.from({length:64},(_,i)=>{const r=rows.get(i),ok=typeof r?.activation==='number';const color=ok?`rgba(44,116,184,${.14+.78*(r.activation-min)/span})`:'#ebf0f5';
      return `<button type="button" class="heat-cell ${i===state.mask?'selected':''}" style="background:${color}" data-mask="${i}" title="Configuration ${i}: ${ok?fmt(r.activation):''}" aria-label="Configuration ${i}${ok?', activation '+fmt(r.activation):''}">${i===state.mask?'●':''}</button>`;}).join('');
    document.querySelectorAll('[data-mask]').forEach(el=>el.addEventListener('click',()=>{state.prevMask=state.mask;state.mask=Number(el.dataset.mask);renderGrid();}));
    const factor=factors[state.contextBit]||factors[0];
    const pairMask=state.mask^(1<<(state.contextBit||0));
    const paired=rows.get(pairMask);
    const levels=factor?.levels||['off','on'];
    const on=!!(state.mask&(1<<(state.contextBit||0)));
    $('context-current').textContent=c&&selected?`Feature ${c.feature_id} activation ${fmt(selected.activation)}. ${factor?.label||'Factor'} is “${levels[on?1:0]}”. The paired state with “${levels[on?0:1]}” is configuration ${pairMask}, activation ${paired?fmt(paired.activation):'—'}.`:'';
    if(!c||valid.length!==64){$('context-marginals').innerHTML='';$('context-pairs').innerHTML='';$('context-evidence').innerHTML='';return;}
    const selectedDiffs=pairedMean(valid, state.contextBit||0);
    const peakDiff=Math.max(...selectedDiffs.map(value=>Math.abs(value)),1e-9);
    $('context-pairs').innerHTML=selectedDiffs.map(value=>`<span class="pair-tick ${value<0?'neg':''}" style="height:${Math.max(8,Math.abs(value)/peakDiff*100)}%" title="${fmt(value,3)}"></span>`).join('')+`<p class="footnote">${selectedDiffs.length} paired on-minus-off differences for ${esc(factor?.label||'this factor')}. Mean ${fmt(selectedDiffs.reduce((sum,value)=>sum+value,0)/selectedDiffs.length,3)}.</p>`;
    const diff=factors.map((f,i)=>{const valuesForBit=pairedMean(valid,i);return {label:f.label||f.id,d:valuesForBit.reduce((sum,value)=>sum+value,0)/valuesForBit.length};});
    const biggest=Math.max(...diff.map(x=>Math.abs(x.d)),1e-9);
    $('context-marginals').innerHTML=diff.map((x,i)=>`<button type="button" class="marginal ${i===state.contextBit?'row-focus':''}" data-marginal="${i}"><span>${esc(x.label)}</span><span class="marginal-track"><span class="marginal-center"></span><span class="marginal-fill" style="left:${x.d>=0?50:50-Math.abs(x.d)/biggest*50}%;width:${Math.abs(x.d)/biggest*50}%"></span></span><span class="marginal-value">${x.d>0?'+':''}${fmt(x.d,3)}</span></button>`).join('');
    document.querySelectorAll('[data-marginal]').forEach(btn=>btn.addEventListener('click',()=>{state.contextBit=Number(btn.dataset.marginal);renderGrid();}));
    $('context-evidence').innerHTML=`<div>${esc(c.scope||'')} ${esc(c.protocol_id)}. Site ${esc(c.site_id)}.</div><div>Model ${esc(c.model_sha256||'')} · SAE ${esc(c.sae_sha256||'')}</div><div>${link('Open source evidence',src(c))}</div>`;}
  function rssGroups(){const groups=new Map();for(const row of cases('rss')){const key=row.site_id+'|'+row.dataset_id;if(!groups.has(key))groups.set(key,{value:key,label:`${row.model} L${row.layer} · ${row.dataset_id}`,site_id:row.site_id,dataset_id:row.dataset_id});}return [...groups.values()];}
  function renderRSS(){const groups=rssGroups();const current=find('rss',state.rssId);const fallback=current?`${current.site_id}|${current.dataset_id}`:groups[0]?.value;
    const groupValue=options($('rss-case'),groups,fallback);
    if(!groupValue){$('rss-shifts').innerHTML='';$('rss-evidence').innerHTML='';$('rss-clean').textContent='';$('rss-changed').textContent='';return;}
    const [siteId,datasetId]=groupValue.split('|');
    const siblings=cases('rss').filter(row=>row.site_id===siteId&&row.dataset_id===datasetId);
    const settings=[...new Set(siblings.map(row=>row.setting))];
    if(!settings.includes(state.rssSetting))state.rssSetting=settings[0]||'local_format';
    $('rss-settings').innerHTML=['local_format','instruction_wrapper','context_wrapper'].map(setting=>{const present=settings.includes(setting);return `<button type="button" class="seg-btn ${state.rssSetting===setting?'active':''}" data-setting="${setting}" ${present?'':'disabled'}>${esc(setting.replaceAll('_',' '))}</button>`;}).join('');
    document.querySelectorAll('[data-setting]').forEach(btn=>btn.addEventListener('click',()=>{if(btn.disabled)return;state.rssSetting=btn.dataset.setting;state.variantId='';renderRSS();}));
    const c=siblings.find(row=>row.setting===state.rssSetting)||siblings[0];
    state.rssId=c?.id||'';
    state.variantId=options($('rss-variant'),(c?.variants||[]).map(v=>({value:v.id,label:v.label||v.id})),state.variantId);
    const v=(c?.variants||[]).find(x=>x.id===state.variantId);$('rss-meta').innerHTML=metadata(c);
    $('rss-chip').textContent='Proposed transformation';
    $('rss-clean').textContent=c?`Source category ${c.dataset_id}. Prompt text is not published.`:'';
    $('rss-changed').textContent=c?`${SETTING_COPY[c.setting]||c.setting} Meaning preservation was not verified.`:'';
    if(!c){$('rss-shifts').innerHTML='';$('rss-evidence').innerHTML='';return;}
    const settingPeak=Math.max(...siblings.map(row=>Math.abs(row.mean_rss)),1e-9);
    const settingBars=siblings.map(row=>`<div class="bar-row ${row.setting===c.setting?'row-focus':''}"><span class="token-label">${esc(row.setting.replaceAll('_',' '))}</span><div class="bar-track"><div class="bar-fill edit-fill" style="width:${Math.abs(row.mean_rss)/settingPeak*100}%"></div></div><span class="bar-number">${fmt(row.mean_rss,3)}</span></div>`).join('');
    const sameSetting=cases('rss').filter(row=>row.site_id===c.site_id&&row.setting===c.setting);
    const dataPeak=Math.max(...sameSetting.map(row=>Math.abs(row.mean_rss)),1e-9);
    const datasetBars=sameSetting.map(row=>`<div class="bar-row ${row.dataset_id===c.dataset_id?'row-focus':''}"><span class="token-label">${esc(row.dataset_id)}</span><div class="bar-track"><div class="bar-fill base-fill" style="width:${Math.abs(row.mean_rss)/dataPeak*100}%"></div></div><span class="bar-number">${fmt(row.mean_rss,3)}</span></div>`).join('');
    const support=(v?.support||[]).slice(0,24).map(id=>`<span class="support-id">${esc(id)}</span>`).join(' ');
    const deltas=v?.feature_deltas||[];
    const deltaChart=deltas.length?deltas.slice().sort((a,b)=>Math.abs(b.delta)-Math.abs(a.delta)).slice(0,20).map(x=>{
      const denom=Math.max(...deltas.map(y=>Math.abs(y.delta)),1e-9),pct=Math.abs(x.delta)/denom*50;
      return `<div class="rss-shift"><span>Feature ${esc(x.feature_id)}</span><span class="rss-track"><span class="rss-fill ${x.delta<0?'neg':''}" style="left:${x.delta<0?50-pct:50}%;width:${pct}%"></span></span><span>${x.delta>0?'+':''}${fmt(x.delta,3)}</span></div>`;}).join(''):'';
    $('rss-shifts').innerHTML=`<div class="measure-list"><div class="measure"><span>Mean RSS</span><strong>${fmt(c.mean_rss,4)}</strong></div><div class="measure"><span>Held-out rate</span><strong>${fmt(c.held_out_rate_mean,4)}</strong></div><div class="measure"><span>Defined / undefined</span><strong>${fmt(c.n_defined,0)} / ${fmt(c.n_undefined,0)}</strong></div></div><div class="section-label space-top">This dataset, three transformations</div>${settingBars}<div class="section-label space-top">This transformation across datasets</div>${datasetBars}${deltaChart}<div class="section-label space-top">Support feature IDs for ${esc(v?.id||'the selected parent')}</div><div class="support-list">${support||''}</div><p class="footnote">Support count ${fmt(v?.support_count,0)}. Held-out variants ${fmt(v?.held_out_variant_count,0)}. Selection variants ${fmt(v?.selection_variant_count,0)}. Means use defined rows only.</p>`;
    $('rss-evidence').innerHTML=`<div>${esc(c.protocol_id)} · ${esc(c.run_id||'')} · semantic review ${c.semantic_review===true?'true':'false'}</div><div>Parents ${fmt(c.n_parent,0)}. Spread of parent Mean RSS ${fmt(c.parent_mean_rss_std,4)}.</div><div>${link('Open source evidence',src(c))}</div>`;}
  function metricSets(){const rows=cases('metrics').filter(m=>m.protocol_id===state.modelProtocol&&(!state.modelFilter||m.model===state.modelFilter));const records=new Map();
    for(const m of rows){const key=m.site_id;if(!records.has(key))records.set(key,{site_id:key,model:m.model||'Unknown',sae:m.sae||'',layer:m.layer,source_url:m.evidence_url,metrics:{}});records.get(key).metrics[m.metric]=m.value;}
    return [...records.values()];}
  function renderModels(){const metrics=cases('metrics'),protocols=[...new Set(metrics.map(x=>x.protocol_id))].sort();state.modelProtocol=options($('model-protocol'),protocols.map(x=>({value:x,label:x})),state.modelProtocol);
    const scoped=metrics.filter(m=>m.protocol_id===state.modelProtocol);const metricNames=[...new Set(scoped.map(m=>m.metric))].sort();
    state.modelX=options($('model-x'),metricNames.map(m=>({value:m,label:m})),state.modelX|| (metricNames.includes('core.l0')?'core.l0':''));
    state.modelY=options($('model-y'),metricNames.map(m=>({value:m,label:m})),state.modelY|| (metricNames.includes('core.explained_variance')?'core.explained_variance':''));
    const modelNames=[...new Set(scoped.map(m=>m.model||'Unknown'))].sort();const filterList=[{value:'all',label:'All models'},...modelNames.map(m=>({value:m,label:m}))];
    state.modelFilter=options($('model-name'),filterList,state.modelFilter||'all');if(state.modelFilter==='all')state.modelFilter='';
    let records=metricSets().filter(r=>typeof r.metrics[state.modelX]==='number'&&typeof r.metrics[state.modelY]==='number');
    $('model-count').textContent=`${records.length} matched sites`;const x=state.modelX,y=state.modelY;
    $('compare-title').textContent= x&&y?`Measured ${y} versus ${x}`:'Quality tradeoffs';
    if(!records.length){$('compare-chart').innerHTML=empty('No matching measured points','Load real model/SAE metric records for both selected axes and a matching protocol. Existing SAEBase results can be exported automatically.');$('compare-selected').innerHTML='';$('compare-table').innerHTML='';return;}
    const vx=records.map(r=>r.metrics[x]),vy=records.map(r=>r.metrics[y]);let x0=Math.min(...vx),x1=Math.max(...vx),y0=Math.min(...vy),y1=Math.max(...vy);const padx=Math.max((x1-x0)*.1,Math.abs(x0)*.03,.001),pady=Math.max((y1-y0)*.1,Math.abs(y0)*.03,.001);x0-=padx;x1+=padx;y0-=pady;y1+=pady;
    const W=780,H=340,L=75,R=25,T=20,B=55,pX=v=>L+(v-x0)/(x1-x0)*(W-L-R),pY=v=>H-B-(v-y0)/(y1-y0)*(H-T-B);
    const ticks=[0,.25,.5,.75,1].map(f=>{let gx=L+f*(W-L-R),gy=H-B-f*(H-T-B);return `<line x1="${gx}" y1="${T}" x2="${gx}" y2="${H-B}"/><line x1="${L}" y1="${gy}" x2="${W-R}" y2="${gy}"/><text x="${gx}" y="${H-32}" text-anchor="middle">${fmt(x0+f*(x1-x0),2)}</text><text x="${L-9}" y="${gy+4}" text-anchor="end">${fmt(y0+f*(y1-y0),2)}</text>`}).join('');
    const circles=records.map(r=>`<circle class="${state.modelSelected===r.site_id?'selected':''}" cx="${pX(r.metrics[x]).toFixed(2)}" cy="${pY(r.metrics[y]).toFixed(2)}" r="6" data-site="${esc(r.site_id)}" tabindex="0"><title>${esc(r.model+' · '+r.sae+' · L'+r.layer)}: ${esc(x)}=${fmt(r.metrics[x])}, ${esc(y)}=${fmt(r.metrics[y])}</title></circle>`).join('');
    $('compare-chart').innerHTML=`<svg class="scatter-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Measured SAE model site scatter plot"><g>${ticks}</g><text x="${(L+W-R)/2}" y="${H-9}" text-anchor="middle">${esc(x)}</text><text transform="translate(15 ${(T+H-B)/2}) rotate(-90)" text-anchor="middle">${esc(y)}</text>${circles}</svg>`;
    $('compare-chart').querySelectorAll('[data-site]').forEach(el=>{const pick=()=>{state.modelSelected=el.dataset.site;renderModels();};el.addEventListener('click',pick);el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick();}})});
    if(!state.modelSelected||!records.some(r=>r.site_id===state.modelSelected))state.modelSelected=records[0].site_id;const selected=records.find(r=>r.site_id===state.modelSelected);
    $('compare-selected').innerHTML=`<strong>Selected: ${esc(standardSite(selected))}</strong> · ${esc(selected.site_id)} · ${link('Source',selected.source_url)}`;
    $('compare-table').innerHTML=`<table><thead><tr><th>Model</th><th>SAE</th><th>Layer</th><th>${esc(x)}</th><th>${esc(y)}</th><th></th></tr></thead><tbody>${records.map(r=>`<tr class="${r.site_id===selected.site_id?'row-focus':''}"><td>${esc(r.model)}</td><td>${esc(r.sae)}</td><td>${esc(r.layer)}</td><td>${fmt(r.metrics[x])}</td><td>${fmt(r.metrics[y])}</td><td><button type="button" data-select-site="${esc(r.site_id)}">Select</button></td></tr>`).join('')}</tbody></table>`;
    $('compare-table').querySelectorAll('[data-select-site]').forEach(b=>b.addEventListener('click',()=>{state.modelSelected=b.dataset.selectSite;renderModels();}));}
  function coalitionCase(){return find('coalitions',state.coalitionId)}
  function response(c,mask){return c?.responses?.find(r=>r.mask===mask)}
  function additive(c,mask){const r0=response(c,0);if(!r0)return null;let pred=r0.value;for(let i=0;i<c.features.length;i++){if(mask&(1<<i)){const single=response(c,1<<i);if(!single)return null;pred+=single.value-r0.value;}}return pred;}
  function renderCoalitions(){state.coalitionId=options($('coalition-case'),cases('coalitions').map(c=>({value:c.id,label:c.title||`${c.features?.length||0} features · ${standardSite(c)}`})),state.coalitionId);
    const c=coalitionCase();$('coalition-meta').innerHTML=metadata(c);$('coalition-features').innerHTML=c?.features.map((f,i)=>`<button type="button" class="coalition-item ${state.coalitionMask&(1<<i)?'active':''}" data-coal-feature="${i}" aria-pressed="${!!(state.coalitionMask&(1<<i))}"><span class="circle"></span>Feature ${esc(f)}</button>`).join('') || `<p class="record-meta">No measured feature coalition groups in this pack.</p>`;
    document.querySelectorAll('[data-coal-feature]').forEach(btn=>btn.addEventListener('click',()=>{state.coalitionMask^=1<<Number(btn.dataset.coalFeature);renderCoalitions();}));
    const n=c?.features.length||0;if(n>0)state.coalitionMask=state.coalitionMask&((1<<n)-1);else state.coalitionMask=0;
    const selected=c?.features.filter((_,i)=>state.coalitionMask&(1<<i))||[];$('coalition-mask').textContent=selected.length?selected.join(', '):'∅';
    const measured=response(c,state.coalitionMask),predicted=additive(c,state.coalitionMask),gap=measured&&predicted!==null?measured.value-predicted:null;
    $('coalition-predicted').textContent=fmt(predicted);$('coalition-observed').textContent=fmt(measured?.value);$('coalition-gap').textContent=fmt(gap);
    $('coalition-chip').textContent=c?`${c.responses.length} recorded subsets`:'No public subsets';
    if(predicted===null||!measured){$('coalition-comparison').innerHTML=empty('Cannot compare this coalition',!c?'Import a measured coalition evidence pack.':'The baseline, required single-feature responses, or selected coalition have not all been measured.');}
    else{const scale=Math.max(Math.abs(predicted),Math.abs(measured.value),1e-9);
      $('coalition-comparison').innerHTML=`<div class="comparison"><label>Predicted effect · ${fmt(predicted)}</label><div class="comparison-track"><div class="comparison-fill" style="width:${Math.abs(predicted)/scale*100}%"></div></div><label>Measured effect · ${fmt(measured.value)}</label><div class="comparison-track"><div class="comparison-fill measured" style="width:${Math.abs(measured.value)/scale*100}%"></div></div></div><div class="footnote">Bar lengths represent absolute magnitude; signed measurements are shown in the labels. Interaction residual = measured − additive prediction.</div>`;}
    $('coalition-table').innerHTML=c?`<table><thead><tr><th>Subset</th><th>Measured response</th><th>Additive prediction</th><th>Residual</th></tr></thead><tbody>${c.responses.slice(0,80).sort((a,b)=>a.mask-b.mask).map(r=>{const p=additive(c,r.mask);return `<tr class="${r.mask===state.coalitionMask?'row-focus':''}"><td><button type="button" data-pick-mask="${r.mask}">${r.mask===0?'Baseline':c.features.filter((_,i)=>r.mask&(1<<i)).join(' + ')}</button></td><td>${fmt(r.value)}</td><td>${fmt(p)}</td><td>${p===null?'—':fmt(r.value-p)}</td></tr>`;}).join('')}</tbody></table>`:'';
    $('coalition-table').querySelectorAll('[data-pick-mask]').forEach(el=>el.addEventListener('click',()=>{state.coalitionMask=Number(el.dataset.pickMask);renderCoalitions();}));
    $('coalition-footnote').innerHTML=c?`Additive prediction is computed only when the baseline and each selected singleton were measured. ${esc(c.protocol_id)}. Different cohort sizes or intervention protocols cannot be silently pooled. ${link('Evidence',src(c))}`:'';}
  function recipe(kind){const t={effects:'effects',context:'contexts',interactions:'coalitions'}[kind];const obj=kind==='effects'?find(t,state.effectId):kind==='context'?controlGridCase():coalitionCase();
    const params=kind==='effects'?{target_token:state.effectTarget}:kind==='context'?{input_mask:state.mask}:{coalition_mask:state.coalitionMask};
    const payload={kind,case_id:obj?.id||null,site_id:obj?.site_id||null,protocol_id:obj?.protocol_id||null,evidence_url:obj?.evidence_url||null,measured_case:!!obj,selection:params,request_only:true,note:'Recipe export is not a new model run or measured result.'};
    const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}),href=URL.createObjectURL(blob),a=document.createElement('a');a.href=href;a.download=`opensae-${kind}-recipe.json`;a.click();setTimeout(()=>URL.revokeObjectURL(href),2000);}
  function share(){const q=new URLSearchParams();if(state.effectId)q.set('effect',state.effectId);if(state.contextId)q.set('context',state.contextId);if(state.rssId)q.set('rss',state.rssId);if(state.coalitionId)q.set('coalition',state.coalitionId);
    const url=location.origin+location.pathname+'?'+q.toString()+'#'+state.route;const done=()=>status('View URL copied. Imported local evidence must also be shared separately to reproduce measurements.');
    if(navigator.clipboard?.writeText)navigator.clipboard.writeText(url).then(done,()=>status(url));else status(url);}
  async function init(){for(const [id,fn] of [
      ['effect-case',()=>{state.effectId=$('effect-case').value;state.edit='ablate';state.removed=false;clearPrediction();renderEffects();}],
      ['context-case',()=>{state.contextId=$('context-case').value;state.prevMask=null;state.mask=bestMask(find('contexts',state.contextId));renderGrid();}],
      ['rss-case',()=>{const [site,dataset]=$('rss-case').value.split('|');const match=cases('rss').find(row=>row.site_id===site&&row.dataset_id===dataset&&row.setting===state.rssSetting)||cases('rss').find(row=>row.site_id===site&&row.dataset_id===dataset);state.rssId=match?.id||'';state.variantId='';renderRSS();}],
      ['rss-variant',()=>{state.variantId=$('rss-variant').value;renderRSS();}],
      ['coalition-case',()=>{state.coalitionId=$('coalition-case').value;state.coalitionMask=0;renderCoalitions();}],
      ['model-protocol',()=>{state.modelProtocol=$('model-protocol').value;state.modelX='';state.modelY='';state.modelFilter='';renderModels();}],
      ['model-x',()=>{state.modelX=$('model-x').value;renderModels();}],
      ['model-y',()=>{state.modelY=$('model-y').value;renderModels();}],
      ['model-name',()=>{state.modelFilter=$('model-name').value==='all'?'':$('model-name').value;renderModels();}]
    ])$(id).addEventListener('change',fn);
    $('effect-next').addEventListener('click',()=>{const list=curatedEffects();if(!list.length)return;const index=Math.max(0,list.findIndex(item=>item.id===state.effectId));state.effectId=list[(index+1)%list.length].id;state.edit='ablate';state.removed=false;clearPrediction();renderEffects();});
    $('effect-on').addEventListener('click',()=>{state.edit='ablate';state.removed=false;renderEffects();});
    $('effect-off').addEventListener('click',()=>{state.edit='ablate';state.removed=true;renderEffects();});
    $('context-surprise').addEventListener('click',()=>{const list=cases('contexts');if(!list.length)return;const pick=list[Math.floor(Math.random()*list.length)];state.contextId=pick.id;state.prevMask=null;state.mask=Math.floor(Math.random()*64);renderGrid();});
    $('context-recipe').addEventListener('click',()=>recipe('context'));
    $('coalition-recipe').addEventListener('click',()=>recipe('interactions'));
    $('context-grid-tab').addEventListener('click',()=>{state.contextMode='grid';renderContext();});$('context-rss-tab').addEventListener('click',()=>{state.contextMode='rss';renderContext();});
    $('share').addEventListener('click',share);
    $('import-json').addEventListener('change',async e=>{const file=e.target.files?.[0];if(!file)return;if(file.size>MAX_UPLOAD){status('JSON file exceeds 25 MB; split into smaller evidence packs.',true);return;}
      try{const parsed=JSON.parse(await file.text());setPack(parsed);status(`Loaded verified-schema evidence pack: ${file.name}. Local import runs entirely in your browser; no data were uploaded.`);}
      catch(error){status('Evidence import rejected: '+error.message,true);}finally{e.target.value='';}});
    window.addEventListener('hashchange',route);
    try{const res=await fetch('data/evidence.json',{cache:'no-store'});if(!res.ok)throw new Error('HTTP '+res.status);setPack(await res.json());route();}
    catch(err){status('Cannot read bundled evidence: '+err.message+'. Serve this directory over HTTP, not file://.',true);$('evidence-status').textContent='Evidence file was not read';}
  }
  document.addEventListener('DOMContentLoaded',init);
})();
