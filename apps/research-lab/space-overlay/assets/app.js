/* SAEBase Research Lab — inspect observed evidence, never synthesize a model response. */
(() => {
  'use strict';
  const D = window.OPENSAE_DATA;
  const $ = id => document.getElementById(id);
  const links = {
    dataset: 'https://huggingface.co/datasets/OpenSAE/SAEBase-Evaluation',
    report: 'https://github.com/OpenSAE/SAEBase/blob/v0.4.0/docs/research-lab/results-20261020.md'
  };
  const friendly = {
    'qwen3_1_7b_base':'Qwen3 1.7B','qwen3_1_7b':'Qwen3 1.7B (AK)',
    'gemma1b':'Gemma 1B','gemma270m':'Gemma 270M',
    'gemma3_4b_pt':'Gemma 3 4B','gemma3_4b_it':'Gemma 3 4B IT',
    'llama31_8b':'Llama 3.1 8B','mistral7b_instruct':'Mistral 7B',
    'pythia410m':'Pythia 410M','qwen35_2b':'Qwen3.5 2B',
    'qwen35_4b':'Qwen3.5 4B','qwen35_9b':'Qwen3.5 9B',
    'qwen3_8b':'Qwen3 8B','gemma2_2b_base':'Gemma 2 2B',
    'gemma2_2b_it':'Gemma 2 2B IT','gemma2_9b_pt':'Gemma 2 9B',
    'r1_distill_qwen15b':'R1 Qwen 1.5B','r1_distill_llama8b':'R1 Llama 8B'
  };
  const el = (tag, cls, text) => { const v = document.createElement(tag); if(cls) v.className=cls; if(text!==undefined) v.textContent=text; return v; };
  const fmt = (num,decimals=1) => Number(num).toLocaleString('en-US',{maximumFractionDigits:decimals,minimumFractionDigits:decimals});
  const short = n => Number(n).toLocaleString('en-US',{maximumFractionDigits:1});
  const nameOf = s => friendly[s.model] || s.model.replaceAll('_',' ');
  const toast = message => { const n=$('toast'); n.textContent=message;n.classList.add('show');clearTimeout(toast.t);toast.t=setTimeout(()=>n.classList.remove('show'),1700); };
  const PROBE_PRESETS = ['qwen3_1_7b_base','gemma3_4b_pt','llama31_8b','mistral7b_instruct'];
  let metrics=[], probes={}, condition='clean', selectedCase=0, probeSite='', selectedMetric='ev',compareModel='qwen3_1_7b_base', selectedLayer=null, factorMask=0;
  const labels={features:'Feature Atlas',effects:'Feature Effects',factors:'Context Lab',models:'Compare models',research:'Research'};
  function navigate(tab){
    const experiments={effects:'effects',factors:'context',models:'models',research:'interactions'};
    const head=String(tab||'').split('/')[0];
    if(experiments[head]){location.replace('lab-experiments/index.html#'+experiments[head]);return;}
    // #play was the old ablation-only tab. Deep links now open the complete feature-first Atlas.
    if(tab==='play' || tab==='atlas' || tab.startsWith('features/')) tab='features';
    if(!labels[tab]) tab='features';
    document.querySelectorAll('.view').forEach(v=>v.classList.toggle('active',v.id==='view-'+tab));
    document.querySelectorAll('.tab').forEach(v=>{let a=v.dataset.tab===tab;v.classList.toggle('active',a);v.setAttribute('aria-current',a?'page':'false');});
    $('crumb').textContent=labels[tab];
    if(!(tab==='features' && location.hash.startsWith('#features/')) && location.hash.slice(1)!==tab) history.replaceState(null,'','#'+tab);
    window.scrollTo({top:0,behavior:'instant'});
  }
  document.querySelectorAll('[data-tab]').forEach(v=>v.addEventListener('click',()=>navigate(v.dataset.tab)));
  document.querySelectorAll('[data-go]').forEach(v=>v.addEventListener('click',()=>navigate(v.dataset.go)));
  window.addEventListener('hashchange',()=>navigate(location.hash.slice(1)));
  function setCondition(next){
    condition=next;
    $('conditionOriginal').classList.toggle('active',next==='clean');
    $('conditionAblated').classList.toggle('active',next==='ablated');
    $('conditionOriginal').setAttribute('aria-pressed',next==='clean'?'true':'false');
    $('conditionAblated').setAttribute('aria-pressed',next==='ablated'?'true':'false');
    renderProbeResult();
  }
  $('conditionOriginal').addEventListener('click',()=>setCondition('clean'));
  $('conditionAblated').addEventListener('click',()=>setCondition('ablated'));
  function initPlay(){
    const siteIds=Object.keys(probes);
    probeSite=siteIds.find(s=>s.startsWith('qwen3_1_7b_base:')) || siteIds[0];
    const selection=$('probeModel');selection.replaceChildren();
    siteIds.forEach(id=>{let m=metrics.find(x=>x.site_id===id);if(!m)return;const option=el('option',null,nameOf(m)+' · Layer '+m.layer);option.value=id;selection.append(option)});
    selection.value=probeSite;
    selection.addEventListener('change',()=>setProbeSite(selection.value));
    const presets=$('modelPresets');presets.replaceChildren();
    PROBE_PRESETS.forEach(model=>{
      let m=metrics.find(x=>x.model===model && probes[x.site_id]);if(!m)return;
      const button=el('button','model-pill',nameOf(m));button.type='button';button.dataset.preset=model;
      button.addEventListener('click',()=>setProbeSite(m.site_id));presets.append(button);
    });
    setProbeSite(probeSite);
  }
  function setProbeSite(id){
    if(!probes[id]||!probes[id].length)return;
    probeSite=id;selectedCase=0;condition='clean';$('probeModel').value=id;
    document.querySelectorAll('[data-preset]').forEach(v=>v.classList.toggle('active',id.startsWith(v.dataset.preset+':')));
    $('conditionOriginal').classList.add('active');$('conditionOriginal').setAttribute('aria-pressed','true');
    $('conditionAblated').classList.remove('active');$('conditionAblated').setAttribute('aria-pressed','false');
    const sample=$('sampleButtons');sample.replaceChildren();
    probes[id].forEach((p,i)=>{let b=el('button','sample-btn');b.type='button';b.setAttribute('aria-label','Recorded sample '+(i+1));
      b.append(el('strong',null,['France','Science','Birds'][i]),el('small',null,'Recorded input'));b.addEventListener('click',()=>{selectedCase=i;renderProbeResult();});sample.append(b);
    });
    renderProbeResult();
  }
  function renderProbeResult(){
    const p=probes[probeSite][selectedCase], m=metrics.find(v=>v.site_id===probeSite);
    document.querySelectorAll('.sample-btn').forEach((b,i)=>{b.classList.toggle('active',i===selectedCase);b.setAttribute('aria-pressed',i===selectedCase?'true':'false');});
    $('probePrompt').textContent=['The capital of France is','A scientist explained the results carefully.','Small birds fly across the morning sky.'][selectedCase];
    $('recordedFeature').replaceChildren();$('recordedFeature').append('Recorded feature ',el('strong',null,'#'+p.feature_id),' · ',el('strong',null,nameOf(m)));
    const number=$('response-number');number.replaceChildren();number.classList.toggle('original',condition==='clean');
    number.append(document.createTextNode(condition==='clean'?'0.00':fmt(p.delta,2)),el('span',null,' L2'));
    $('response-caption').textContent='Change in output scores';
    $('response-desc').textContent=condition==='clean'?'Original output · no feature removed':'Real measurement · feature #'+p.feature_id+' removed';
    $('observationMeta').textContent='Feature #'+p.feature_id+' · '+p.l0+' active features · Sample '+(selectedCase+1);
    $('originalEvidenceLink').href=p.source;
    const max=Math.max(...probes[probeSite].map(x=>x.delta),1), chart=$('probeChart');chart.replaceChildren();
    probes[probeSite].forEach((v,i)=>{
      const col=el('button','probe-col'+(selectedCase===i?' active':''));col.type='button';col.title='Actual output change: '+fmt(v.delta,3)+' (logit L2)';
      col.setAttribute('aria-label','Sample '+(i+1)+' measured output change '+fmt(v.delta,2));
      const value=el('span','probe-value',short(v.delta));
      const bar=el('span','probe-bar');bar.style.height=Math.max(4,Math.round(119*(v.delta/max)))+'px';
      col.append(value,bar,el('span',null,'Sample '+(i+1)));col.addEventListener('click',()=>{selectedCase=i;renderProbeResult();});chart.append(col);
    });
  }
  function initFactors(){
    factorMask=0;
    const d=D.controls.real_model_six_factor_design;
    const grid=$('stateGrid');grid.replaceChildren();
    d.states.forEach((v,i)=>{let b=el('button','state-cell',String(i+1).padStart(2,'0'));b.type='button';b.title='State '+i+' · '+d.factors.map((f,axis)=>f.levels[(i>>axis)&1]).join(' / ');b.dataset.mask=String(i);b.setAttribute('aria-label','Choose factor state '+i);b.addEventListener('click',()=>{factorMask=i;renderFactor();});grid.append(b);});
    $('randomState').addEventListener('click',()=>{factorMask=(factorMask+1+Math.floor(Math.random()*63))%64;renderFactor();toast('Selected state '+(factorMask+1));});
    $('copyPrompt').addEventListener('click',()=>{
      const state=d.states.find(v=>Number(v.mask)===factorMask);
      const content=state.prompt;
      if(navigator.clipboard?.writeText){navigator.clipboard.writeText(content).then(()=>toast('Input copied')).catch(()=>downloadText(content));}
      else downloadText(content);
    });
    renderFactor();
  }
  function downloadText(txt){const blob=new Blob([txt],{type:'text/plain'});const url=URL.createObjectURL(blob),a=el('a');a.href=url;a.download='opensae-controlled-input.txt';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('Input downloaded');}
  function renderFactor(){
    const d=D.controls.real_model_six_factor_design;
    $('maskLabel').textContent='State '+String(factorMask+1).padStart(2,'0')+' / 64';
    const list=$('factorControls');list.replaceChildren();
    d.factors.forEach((f,i)=>{
      const row=el('div','factor-row'),name=el('div','factor-title',f.name==='voice'?'Narrator':f.name.charAt(0).toUpperCase()+f.name.slice(1)),options=el('div','factor-options');
      f.levels.forEach((choice,bit)=>{
        const button=el('button',((factorMask>>i)&1)===bit?'active':'',choice);button.type='button';
        button.setAttribute('aria-pressed',String(((factorMask>>i)&1)===bit));
        button.addEventListener('click',()=>{factorMask=(factorMask&~(1<<i))|(bit<<i);renderFactor();});options.append(button);
      });row.append(name,options);list.append(row);
    });
    const state=d.states.find(s=>Number(s.mask)===factorMask);
    if(!state){console.error('Missing controlled state',factorMask);return;}
    $('livePrompt').textContent=state.prompt;
    $('inputMeta').textContent='Authentic input · State '+factorMask+' · From complete 64-state controlled design';
    document.querySelectorAll('.state-cell').forEach(b=>{let sel=Number(b.dataset.mask)===factorMask;b.classList.toggle('active',sel);b.setAttribute('aria-pressed',String(sel));});
  }
  function initModels(){
    const models=[...new Set(metrics.map(x=>x.model))].sort((a,b)=>nameOf({model:a}).localeCompare(nameOf({model:b})));
    const select=$('compareModel');select.replaceChildren();models.forEach(m=>{const op=el('option',null,nameOf({model:m}));op.value=m;select.append(op);});
    select.value=compareModel;
    select.addEventListener('change',()=>{compareModel=select.value;selectedLayer=null;renderModels();});
    $('metricSelect').addEventListener('change',()=>{selectedMetric=$('metricSelect').value;renderModels();});
    renderModels();
  }
  const METRIC_LABELS={ev:'Explained variance',l0:'Active features (L0)',ablation_delta:'Change in output scores',mse:'Reconstruction error'};
  const readableMetric=(k,v)=>k==='ev'?fmt(v*100,1)+'%':k==='mse'?fmt(v,3):fmt(v,1);
  function renderModels(){
    const rows=metrics.filter(m=>m.model===compareModel).sort((a,b)=>a.layer-b.layer);
    const active=rows.find(x=>x.layer===selectedLayer)||rows[Math.floor(rows.length/2)]; if(!active)return;
    selectedLayer=active.layer;
    $('metricCaption').textContent=METRIC_LABELS[selectedMetric]+' across layers';
    const amounts=rows.map(r=>r[selectedMetric]);const max=Math.max(...amounts.map(x=>Math.abs(x)),selectedMetric==='ev'?1:0.00001);
    const chart=$('layerChart');chart.replaceChildren();
    rows.forEach(m=>{
      const button=el('button','layer-bar-group'+(m.layer===selectedLayer?' active':''));button.type='button';button.title='Layer '+m.layer+': '+readableMetric(selectedMetric,m[selectedMetric]);button.setAttribute('aria-label','Select layer '+m.layer);
      const bar=el('span','layer-bar');bar.style.height=Math.max(4,Math.round(Math.abs(m[selectedMetric])/max*214))+'px';
      button.append(el('span','layer-value',readableMetric(selectedMetric,m[selectedMetric])),bar,el('span','layer-lbl','L'+m.layer));
      button.addEventListener('click',()=>{selectedLayer=m.layer;renderModels();});chart.append(button);
    });
    $('layerName').textContent='Layer '+selectedLayer;
    $('metricStack').replaceChildren();[['ev','Explained variance'],['l0','Active SAE features'],['ablation_delta','Feature-off output change'],['mse','Reconstruction error']].forEach(([k,label])=>{
      const item=el('div','metric-item');item.append(el('span',null,label),el('strong',null,readableMetric(k,active[k])));$('metricStack').append(item);
    });
    $('layerSource').href=active.reference;
  }
  function initResearch(){
    const sar=$('kButtons');sar.replaceChildren();
    D.sar.forEach(r=>{let b=el('button',null,'K'+r.k);b.type='button';b.dataset.k=r.k;b.addEventListener('click',()=>renderSar(Number(r.k)));sar.append(b);});
    renderSar(16);
    const stack=$('rssStack');stack.replaceChildren();
    [['Selection changes',45000],['Held-out checks',90000],['Clean inputs',3000]].forEach(([label,count])=>{
      const row=el('div','rss-row'),head=el('div','rss-row-head');head.append(el('span',null,label),el('strong',null,short(count)));
      const track=el('div','rss-track'),fill=el('div','rss-fill');fill.style.width=(count/90000*100)+'%';track.append(fill);row.append(head,track);stack.append(row);
    });
  }
  function renderSar(k){
    const data=D.sar.find(x=>Number(x.k)===k);if(!data)return;
    const total=Number(data.successful_groups);
    document.querySelectorAll('[data-k]').forEach(b=>{const active=Number(b.dataset.k)===k;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
    $('sarTotal').replaceChildren();$('sarTotal').append(document.createTextNode(short(total)),el('span',null,' groups'));
    const groups=D.sar;const max=Math.max(...groups.map(x=>Number(x.successful_groups)));
    const viz=$('sarViz');viz.replaceChildren();groups.forEach(g=>{let b=el('div',Number(g.k)===k?'sar-col selected':'sar-col');b.style.height=Math.max(3,Math.round(90*Number(g.successful_groups)/max))+'px';b.title='K'+g.k+': '+g.successful_groups+' groups';viz.append(b);});
    $('sarReadout').textContent=`${data.m} measured coalition responses per valid group · ${data.fit} fit / ${data.held} held-out. These are real group counts, not generated coalition-level values.`;
  }
  async function start(){
    if(!D || !D.controls?.real_model_six_factor_design?.states || D.controls.real_model_six_factor_design.states.length!==64){throw Error('Experimental data absent; refusing synthetic fallback');}
    const [mr,pr]=await Promise.all([fetch('assets/metrics.json'),fetch('assets/probes.json')]);
    if(!mr.ok||!pr.ok)throw Error('Original measurement asset unavailable');
    metrics=await mr.json();probes=await pr.json();
    if(metrics.length!==48||Object.values(probes).some(x=>x.length!==3))throw Error('Actual measurement manifest incomplete');
    initPlay();initFactors();initModels();initResearch();navigate(location.hash.slice(1));
    window.playgroundDebug={metrics,probes,selectSite:setProbeSite,selectCondition:setCondition,selectProbe:(i)=>{selectedCase=i;renderProbeResult();},selectFactorMask:(i)=>{if(Number.isInteger(i)&&i>=0&&i<64){factorMask=i;renderFactor();}},getFactor:()=>factorMask,navigate};
  }
  start().catch(error=>{console.error(error);$('response-desc').textContent='Measurement data unavailable. Reload this page or open the source dataset.';$('response-number').textContent='—';toast('Could not load real experiment data');});
})();
