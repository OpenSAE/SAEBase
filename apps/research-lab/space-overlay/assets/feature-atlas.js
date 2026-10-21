/* OpenSAE Feature Atlas. Every example is hash-matched to a real checkpoint probe.
   Prompt topics are NOT model feature explanations. No fabricated token heats. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const el=(tag,cls,txt)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(txt!==undefined)n.textContent=txt;return n;};
  const fmt=(n)=>Number(n).toLocaleString('en-US',{maximumFractionDigits:2,minimumFractionDigits:2});
  const friendly={gemma1b:'Gemma 1B',gemma3_4b_pt:'Gemma 3 4B',gemma3_4b_it:'Gemma 3 4B IT',pythia410m:'Pythia 410M',llama31_8b:'Llama 3.1 8B',qwen3_1_7b_base:'Qwen3 1.7B',mistral7b_instruct:'Mistral 7B'};
  let features=[],active=null, activeExample=0;
  const storedKey=f=>'opensae-feature-note::'+f.id;
  function feedback(message){$('atlasSavedMsg').textContent=message;}
  function setHash(f){
    const val='features/'+encodeURIComponent(f.id);
    if(decodeURI(location.hash.slice(1))!==decodeURI(val))history.replaceState(null,'','#'+val);
  }
  function highlightPrompt(holder,example){
    const text=example.prompt;
    // We only have last-position readings from the original acceptance records.
    // Highlight the final word, clearly identified as the readout POSITION.
    const match=/^(.*?)(\S+)(\s*)$/s.exec(text);
    const p=el('div','atlas-prompt-line');
    if(match){p.append(document.createTextNode(match[1]));const mark=el('mark','atlas-target-token',match[2]);mark.title='Last visible word; measurement was at final MODEL TOKEN. Exact subword boundary was not stored.';p.append(mark,document.createTextNode(match[3]));}
    else p.textContent=text;
    holder.append(p);
  }
  function selectFeature(f,opt={}){
    active=f;activeExample=0;
    if(!opt.noHash)setHash(f);
    document.querySelectorAll('.atlas-item').forEach(n=>n.classList.toggle('active',n.dataset.id===f.id));
    $('atlasModel').textContent=(friendly[f.model]||f.model)+' · Layer '+f.layer;
    $('atlasFeatureTitle').textContent='Feature #'+f.feature_id;
    $('atlasHypothesis').textContent='Meaning not yet established';
    $('atlasExplanation').textContent=f.candidate_explanation;
    $('atlasEvidenceMeta').replaceChildren();
    const inf=[`${f.observed_count} recorded input${f.observed_count===1?'':'s'}`,...f.observed_topics.map(t=>'Prompt topic: '+t)];
    inf.forEach(a=>$('atlasEvidenceMeta').append(el('span','atlas-chip',a)));
    const list=$('atlasExamples');list.replaceChildren();
    f.examples.forEach((x,i)=>{
      const row=el('button','atlas-prompt-card'+(i===activeExample?' active':''));row.type='button';row.dataset.idx=i;
      const upper=el('div','atlas-prompt-top');upper.append(el('span','atlas-topic',x.topic),el('span','atlas-prompt-index','Input '+x.case));
      row.append(upper);highlightPrompt(row,x);
      row.append(el('p','atlas-prompt-meaning',x.prompt_meaning));
      const change=el('div','atlas-prompt-effect');change.append(el('span',null,'After removing feature'),el('strong',null,'Δ '+fmt(x.delta_logit_l2)+' L2'));
      row.append(change);row.addEventListener('click',()=>{activeExample=i;updateSelectedExample();});list.append(row);
    });
    updateSelectedExample();
    $('atlasDraft').value='';
    try{$('atlasDraft').value=localStorage.getItem(storedKey(f))||'';}catch(_){/* browsing still works without storage */}
    feedback('');
    $('atlasSource').href=f.source_url;
  }
  function updateSelectedExample(){
    if(!active)return;
    const ex=active.examples[activeExample];
    const linked=(window.OPENSAEEffectLinks||[]).some(row=>row.site_id===ex.site_id && Number(row.feature_id)===Number(active.feature_id) && row.prompt_sha256===ex.prompt_sha256);
    $('atlasTryEdit').textContent=linked?'Explore this intervention →':'Browse measured effects →';
    $('atlasMainEffect').replaceChildren();$('atlasMainEffect').append(document.createTextNode(fmt(ex.delta_logit_l2)),el('small',null,' L2'));
    document.querySelectorAll('.atlas-prompt-card').forEach((n,i)=>n.classList.toggle('active',i===activeExample));
    const bars=$('atlasEffectBars');bars.replaceChildren();
    const max=Math.max(...active.examples.map(x=>x.delta_logit_l2),.001);
    active.examples.forEach((x,i)=>{
      const row=el('button','atlas-effect-row'+(i===activeExample?' active':''));row.type='button';
      row.append(el('span','atlas-effect-name',x.topic_short));
      const track=el('span','atlas-bar-track');const bar=el('span','atlas-bar-fill');bar.style.width=(x.delta_logit_l2/max*100)+'%';track.append(bar);row.append(track);
      row.append(el('strong',null,fmt(x.delta_logit_l2)));row.addEventListener('click',()=>{activeExample=i;updateSelectedExample();});bars.append(row);
    });
    $('atlasSource').href=ex.source_url;
  }
  function renderList(query=''){
    query=query.trim().toLowerCase();const list=$('atlasList');list.replaceChildren();
    const found=features.filter(f=>[f.model,f.site_id,String(f.feature_id),f.candidate_label,...f.observed_topics,...f.examples.map(e=>e.prompt)].join(' ').toLowerCase().includes(query));
    $('atlasCount').textContent=found.length+' of '+features.length;
    for(const f of found){
      const button=el('button','atlas-item'+(active?.id===f.id?' active':''));button.type='button';button.dataset.id=f.id;
      const head=el('div','atlas-item-head');head.append(el('strong',null,'#'+f.feature_id),el('span',null,(friendly[f.model]||f.model)+' · L'+f.layer));
      button.append(head,el('div','atlas-item-desc','Observed feature · interpretation pending'),el('div','atlas-item-count',f.observed_count+' measured input'+(f.observed_count===1?'':'s')));
      button.addEventListener('click',()=>selectFeature(f));list.append(button);
    }
    if(!found.length)list.append(el('div','atlas-empty','No matching measured features. Try a model name or topic.'));
  }
  function share(){
    if(!active)return;
    const url=location.origin+location.pathname+'#features/'+encodeURIComponent(active.id);
    if(navigator.clipboard?.writeText){navigator.clipboard.writeText(url).then(()=>feedback('Feature link copied')).catch(()=>feedback('Your browser blocked copying'));}
    else feedback('Link: '+url);
  }
  function copyEvidence(){
    if(!active)return;
    const val=`${friendly[active.model]||active.model} · Layer ${active.layer} · SAE feature ${active.feature_id}\n${active.candidate_label}\n${active.candidate_explanation}\n\n`+active.examples.map(x=>`Prompt: ${x.prompt}\nPrompt topic: ${x.topic}\nFeature removal Δ logits: ${fmt(x.delta_logit_l2)} L2\nOriginal source: ${x.source_url}`).join('\n\n')+'\n\nInterpretation status: unverified candidate; per-token activation map not available.';
    if(navigator.clipboard?.writeText){navigator.clipboard.writeText(val).then(()=>feedback('Real evidence copied')).catch(()=>feedback('Copy unavailable'));}
    else feedback('Copy unavailable in this browser');
  }
  function moveToIntervention(){
    if(!active)return;
    const ex=active.examples[activeExample];
    const links=window.OPENSAEEffectLinks||[];
    const hit=links.find(row=>row.site_id===ex.site_id && Number(row.feature_id)===Number(active.feature_id) && row.prompt_sha256===ex.prompt_sha256);
    if(!hit){feedback('This example stays on its recorded card. No paired edit matches this site, feature, and prompt.');return;}
    location.href='lab-experiments/index.html?effect='+encodeURIComponent(hit.case_id)+'#effects';
  }
  async function init(){
    try{const linkResponse=await fetch('assets/effect-links.json');window.OPENSAEEffectLinks=linkResponse.ok?await linkResponse.json():[];}catch(_){window.OPENSAEEffectLinks=[];}
    const response=await fetch('assets/features.json');if(!response.ok)throw Error('Cannot load observed feature examples');
    const d=await response.json();
    features=d.features;if(!features||features.length!==14 || features.reduce((a,f)=>a+f.examples.length,0)!==21)throw Error('Feature evidence incomplete');
    window.OPENSAEAtlas={showMeasuredPrompt(sha, preferredSite){
      const matched=features.filter(f=>f.examples.some(e=>e.prompt_sha256===sha));
      const picked=matched.find(f=>f.site_id===preferredSite)||matched[0];
      if(!picked)return false;
      selectFeature(picked,{noHash:true});
      activeExample=Math.max(0,picked.examples.findIndex(e=>e.prompt_sha256===sha));
      updateSelectedExample();
      $('atlasDetailSpotlight')?.scrollIntoView({behavior:'smooth',block:'start'});
      return true;
    }};

    const activeFromHash=()=>{
      const hash=location.hash.slice(1);
      if(hash.startsWith('features/')){
        const id=decodeURIComponent(hash.slice(9));
        const chosen=features.find(f=>f.id===id);if(chosen){selectFeature(chosen,{noHash:true});return;}
      }
      if(!active)selectFeature(features.find(f=>f.model==='qwen3_1_7b_base')||features[0],{noHash:true});
    };
    $('atlasSearch').addEventListener('input',()=>renderList($('atlasSearch').value));
    $('atlasSave').addEventListener('click',()=>{if(!active)return;try{localStorage.setItem(storedKey(active),$('atlasDraft').value);feedback('Draft saved on this device');}catch(_){feedback('Browser storage unavailable — copy your draft');}});
    $('atlasShare').addEventListener('click',share);
    $('atlasCopy').addEventListener('click',copyEvidence);
    $('atlasTryEdit').addEventListener('click',moveToIntervention);
    window.addEventListener('hashchange',activeFromHash);
    activeFromHash();renderList();
    window.featureAtlasDebug={count:features.length,observations:21,selectFeature:id=>{const f=features.find(x=>x.id===id);if(f){selectFeature(f);return true;}return false;},getSelected:()=>active?.id,selectExample:i=>{activeExample=i;updateSelectedExample();}};
  }
  init().catch(err=>{console.error(err);$('atlasList').textContent='Feature examples could not be loaded. Please reload.';});
})();
