/* OpenResearch Atlas: centered, source-backed trees with node focus navigation. */
(function () {
  "use strict";


  const DATA = window.OPENRESEARCH_ATLAS || {};
  // The large public data.js is refreshed independently of the canonical
  // repository. This generated overlay carries only source-backed structure.
  const STRUCTURE = window.OPENRESEARCH_ATLAS_PATCH;
  if(STRUCTURE?.schema === "openresearch.atlas.structure.v1"){
    if(STRUCTURE.knowledge) DATA.knowledge = STRUCTURE.knowledge;
    for(const view of ["code","context"]){
      const graph = DATA[view], patch = STRUCTURE[view];
      if(!graph || !patch)continue;
      const ids = new Set((graph.nodes||[]).map(n=>n.id));
      for(const node of patch.nodes||[]){
        if(!ids.has(node.id)){graph.nodes.push(node);ids.add(node.id);}
      }
      if(view==="code"){
        const byId=new Map(graph.nodes.map(n=>[n.id,n]));
        for(const [id,science] of Object.entries(patch.science_links||{}))
          if(byId.has(id))byId.get(id).context_ids=science;
      }
      const moved = new Map(Object.entries(patch.parents||{})
        .filter(([child,parent])=>ids.has(child)&&ids.has(parent)));
      graph.edges = (graph.edges||[]).filter(edge=>
        edge.type!=="contains" || !moved.has(edge.target));
      for(const [child,parent] of moved)
        graph.edges.push({source:parent,target:child,type:"contains"});
    }
  }

  // Public GitHub Pages data snapshots are published independently of this
  // application. Upgrade only older, generic headings from actual node IDs,
  // Python symbols and paths, without touching the authoritative source files.
  const LEGACY_TITLES = {
    "core:existing":"SAEBase 研究架构与任务注册",
    "knowledge:existing":"LLM SAE 科研方法与实现依据",
    "context:root":"SAE 技术术语及源码依据",
    "env:root":"CPU、GPU 与任务执行环境",
    "code:root":"SAEBase Python 实现树",
    "q.modularity":"Dataset、Model、SAE Task Registry",
    "q.transparency":"Atlas 五棵 Tree 的职责与关联",
    "g.registry":"模型、数据与 SAE Task 插件注册",
    "g.pipeline":"SAE Task 运行前的兼容检查",
    "code:concrete:datasets":"Prompt 数据集",
    "code:concrete:models":"LLM 模型实现",
    "code:concrete:tasks":"LLM SAE 任务执行",
    "code:group:tasks":"SAE 研究任务",
  };
  const LEGACY_ONLY = new Set([
    "已有的项目功能","研究概念与依据","机器与执行条件",
    "现有代码能力","已有的模块化组件架构","已有的项目上下文组织",
    "已有的训练配置与产物管理",
    "定义训练损失","记录 Hydra 组合配置",
    "确定随机性与执行环境","保存可追溯的结果文件",
    "可插拔组件注册","Task 运行前检查"
  ]);
  // Code titles are authored in the Python semantic presentation layer.
  // Browsers must never append paths, class names or Python identifiers.
  function improveLegacyTitles(){
    for(const [view,tree] of Object.entries(DATA)){
      for(const node of tree?.nodes||[]){
        const old=node.title||"";
        const replacement=view==="code" ? null :
          (LEGACY_ONLY.has(old) ? LEGACY_TITLES[node.id] : null);
        if(!replacement||replacement===old)continue;
        node.title=replacement;
        if(node.reader_explanation)
          node.reader_explanation=node.reader_explanation.replaceAll(old,replacement);
      }
    }
  }
  improveLegacyTitles();
  // Legacy public snapshots may contain older Code source entities. Group only
  // nodes actually present after the source-backed Atlas structure overlay.
  if(typeof window.OPENRESEARCH_ATLAS_BALANCE==="function"){
    window.OPENRESEARCH_ATLAS_BALANCE(DATA.code,"code");
    window.OPENRESEARCH_ATLAS_BALANCE(DATA.context,"context");
  }
  const VIEWS = {
    core: { root: "core:existing", primary: ["contains", "refines", "addresses", "decomposes"] },
    knowledge: { root: "knowledge:existing", primary: ["contains", "refines", "addresses", "decomposes"] },
    code: { root: "code:root", primary: ["contains"] },
    environment: { root: "env:root", primary: ["contains"] },
    context: { root: "context:root", primary: ["contains"] }
  };
  const STATES = {
    unchecked: "#8255b5", denied: "#78818b",
    not_implemented: "#c85b61",
    partial: "#d5a238",
    implemented: "#429669"
  };
  // Node kind never dictates shading. State alone defines the surface.
  const NODE_THEME = {
    implemented: {rgb:"66,150,105",accent:"#76ad91",ink:"#416b56"},
    not_implemented: {rgb:"200,91,97",accent:"#d19a9e",ink:"#935d64"},
    partial: {rgb:"213,162,56",accent:"#d8be86",ink:"#886f39"},
    unchecked: {rgb:"130,85,181",accent:"#b09cc9",ink:"#775f91"},
    denied: {rgb:"120,129,139",accent:"#acb3ba",ink:"#636f78"},
    unknown: {rgb:"120,129,139",accent:"#bac0c6",ink:"#68747d"}
  };
  const STATE_FILL = Object.fromEntries(Object.entries(NODE_THEME).map(
    ([status,color])=>[status,"rgba("+color.rgb+","+
      (status==="unknown"?".035":".085")+")"]));

  const $ = id => document.getElementById(id);

  // Review is a mode inside Atlas. It reuses the real graph, cards and navigation.
  // Proposal decisions are NOT implementation results: unchecked purple,
  // approved red (still unimplemented), denied gray.
  const REVIEW_PALETTE = {
    unchecked: { fill: STATE_FILL.unchecked, stroke: "#8255b5" },
    approved: { fill: STATE_FILL.not_implemented, stroke: "#bb4656" },
    denied: { fill: STATE_FILL.denied, stroke: "#78818b" }
  };
  let review = null, reviewGraphs = {}, reviewBusy = false, reviewError = "";
  let reviewDraftMessage = "";
  const REVIEW_VIEWS=["knowledge","code","environment","context"];
  const REVIEW_TITLES={knowledge:"Knowledge",code:"Code",
    environment:"Environment",context:"Context"};
  let reviewQueue=[],reviewGroupOf=new Map(),reviewGroups=[];
  let reviewFilter={level:"all"},reviewMenuOpen=false;
  const reviewExpanded=new Set(["all",...REVIEW_VIEWS.map(v=>"view:"+v)]);

  function requestedReview() {
    const match = (window.location?.search || "").match(/[?&]review=([^&]+)/);
    if (!match) return null;
    try { return decodeURIComponent(match[1]); } catch (_) { return null; }
  }
  function reviewKey() {
    const match = (window.location?.hash || "").match(/^#key=([^&]+)/);
    try { return match ? decodeURIComponent(match[1]) : ""; }
    catch (_) { return ""; }
  }
  function proposalState(item) {
    return item?.proposal_review?.decision || "unchecked";
  }
  function itemFor(id, view=config.view) {
    return review?.plan?.items?.find(x => x.node_id === id && x.view === view);
  }
  function overlayGraphs() {
    reviewGraphs = {};
    if (!review) return;
    const anchor = review.context || { nodes:[], edges:[] };
    for(const view of ["knowledge","code","environment","context"]) {
      const base = DATA[view] || {nodes:[],edges:[]};
      const nodes = new Map((base.nodes || []).map(n=>[n.id,n]));
      const edges = [...(base.edges || [])];
      const addEdge = edge => {
        if (!edges.some(e => e.source === edge.source && e.target === edge.target &&
             e.type === edge.type)) edges.push(edge);
      };
      if (view === "knowledge") {
        for(const node of anchor.nodes || [])
          if (!nodes.has(node.id)) nodes.set(node.id,node);
        for(const edge of anchor.edges || []) addEdge(edge);
      }
      for(const item of review.plan.items.filter(x=>x.view===view)) {
        const baseNode = nodes.get(item.node_id) || {};
        const after = item.after || {};
        const origin = item.before || {};
        const displayed = {
          ...baseNode,
          // Keep the normal node's interface, device facts and terminology.
          // Proposed fields are never read as real implemented Code evidence.
          ...after,
          id: item.node_id,
          kind: baseNode.kind || after.type || origin.type || "goal",
          title: (() => {
            const original=after.title || origin.title || baseNode.title || item.node_id;
            const parentSymbol=item.parent_id?.startsWith("code:class:")
              ? item.parent_id.split(":").pop() : "";
            if(view==="code" && parentSymbol && !original.includes(parentSymbol))
              return parentSymbol+" · "+original;

            return original;
          })(),
          description: after.description || origin.description || baseNode.description || "",
          summary: after.summary || after.description || origin.description || baseNode.summary || "",
          status: proposalState(item) === "approved" ? "not_implemented" :
                  proposalState(item) === "denied" ? "denied" : "unchecked",
          reader_explanation: after.reader_explanation || after.description || origin.description || "",
          context_term_ids: after.context_term_ids || after.term_ids ||
                            baseNode.context_term_ids || [],
          review_item: true
        };
        nodes.set(item.node_id,displayed);
        const parent = item.parent_id;
        if (parent && nodes.has(parent)) {
          addEdge({source:parent,target:item.node_id,type:"contains"});
        } else if (item.operation === "add" && nodes.has(VIEWS[view].root)) {
          addEdge({source:VIEWS[view].root,target:item.node_id,type:"contains"});
        }
      }
      reviewGraphs[view] = {...base,nodes:[...nodes.values()],edges};
    }
  }
  function buildReviewQueue() {
    // DFS over the source tree's stored child edges, never manifest list order.
    reviewQueue=[]; reviewGroups=[]; reviewGroupOf=new Map();
    if(!review)return;
    const indexed=new Map(review.plan.items.map(x=>[x.key,x]));
    const visitedItems=new Set();
    for(const view of REVIEW_VIEWS) {
      const tree=reviewGraphs[view] || DATA[view] || {nodes:[],edges:[]};
      const nodes=new Map((tree.nodes||[]).map(x=>[x.id,x]));
      const kids=new Map(),parentsOf=new Map();
      const allowed=new Set(VIEWS[view].primary);
      for(const edge of tree.edges||[]) {
        if(!allowed.has(edge.type)||!nodes.has(edge.source)||
           !nodes.has(edge.target)||parentsOf.has(edge.target))continue;
        parentsOf.set(edge.target,edge.source);
        if(!kids.has(edge.source))kids.set(edge.source,[]);
        kids.get(edge.source).push(edge.target);
      }
      const roots=[];
      if(nodes.has(VIEWS[view].root))roots.push(VIEWS[view].root);
      for(const id of nodes.keys())if(!parentsOf.has(id)&&!roots.includes(id))roots.push(id);
      const seen=new Set(),stack=roots.reverse();
      while(stack.length){
        const id=stack.pop();
        if(seen.has(id))continue;
        seen.add(id);
        const key=view+":"+id;
        if(indexed.has(key)){reviewQueue.push(indexed.get(key));visitedItems.add(key);}
        const list=kids.get(id)||[];
        for(let i=list.length-1;i>=0;i--)stack.push(list[i]);
      }
      // Orphan proposals follow the reachable DFS nodes in their own view.
      for(const item of review.plan.items){
        if(item.view===view&&!visitedItems.has(item.key)){
          reviewQueue.push(item);visitedItems.add(item.key);
        }
      }
      const branches=new Map();
      for(const item of reviewQueue.filter(x=>x.view===view)){
        const parentId=parentsOf.get(item.node_id)||item.parent_id||VIEWS[view].root;
        const key=view+"\u0000"+parentId;
        reviewGroupOf.set(item.key,key);
        if(!branches.has(key)){
          branches.set(key,{key,view,parentId,
                            label:nodes.get(parentId)?.title||parentId,items:[]});
        }
        branches.get(key).items.push(item);
      }
      if(branches.size)reviewGroups.push({
        view,title:REVIEW_TITLES[view],groups:[...branches.values()]
      });
    }
  }
  function filteredReviewQueue(){
    if(reviewFilter.level==="all")return reviewQueue;
    if(reviewFilter.level==="view")
      return reviewQueue.filter(item=>item.view===reviewFilter.view);
    if(reviewFilter.level==="parent")
      return reviewQueue.filter(item=>reviewGroupOf.get(item.key)===reviewFilter.parentKey);
    if(reviewFilter.level==="node")
      return reviewQueue.filter(item=>item.key===reviewFilter.nodeKey);
    return reviewQueue;
  }
  function reviewPosition(){
    return filteredReviewQueue().findIndex(item=>
      item.node_id===config.focusId && item.view===config.view);
  }
  function reviewGo(index){
    const queue=filteredReviewQueue();
    if(!review||reviewBusy||index<0||index>=queue.length)return;
    reviewError="";
    navigate(queue[index].node_id,queue[index].view,false);
  }
  function findUnchecked(direction){
    const queue=filteredReviewQueue(),at=reviewPosition();
    if(at<0||queue.length<2)return -1;
    for(let offset=1;offset<queue.length;offset++){
      const index=(at+direction*offset+queue.length)%queue.length;
      if(proposalState(queue[index])==="unchecked")return index;
    }
    return -1;
  }
  function reviewGoUnchecked(direction){
    const index=findUnchecked(direction);
    if(index>=0)reviewGo(index);
  }
  function reviewStats(items){
    const counts={approved:0,denied:0,unchecked:0,total:items.length};
    for(const item of items){
      const key=proposalState(item);
      counts[key]=(counts[key]||0)+1;
    }
    return counts;
  }


  function reviewRing(items){
    const c=reviewStats(items),ring=el("span","review-ring");
    const total=c.total||1,red=360*c.approved/total,
          gray=360*(c.approved+c.denied)/total;
    ring.style.background="conic-gradient(#c9848b 0deg "+red+
      "deg,#a5adb5 "+red+"deg "+gray+
      "deg,#ab94c8 "+gray+"deg 360deg)";
    ring.title=(c.approved+c.denied)+" of "+c.total+" reviewed";
    ring.appendChild(el("span","review-ring-center",
                        (c.approved+c.denied)+"/"+c.total));
    return ring;
  }
  function reviewCountDots(items){
    const counts=reviewStats(items),row=el("span","review-state-counts");
    for(const [state,color] of [["approved","red"],["denied","gray"],["unchecked","purple"]]){
      const badge=el("span","review-count-chip");
      badge.title=state+": "+counts[state];
      badge.appendChild(el("i","review-count-dot "+color));
      badge.appendChild(el("span","",counts[state]));
      row.appendChild(badge);
    }
    return row;
  }
  function reviewFilterLabel(){
    if(reviewFilter.level==="all")return "All";
    if(reviewFilter.level==="view")return REVIEW_TITLES[reviewFilter.view];
    if(reviewFilter.level==="parent")return reviewGroups.flatMap(x=>x.groups)
      .find(x=>x.key===reviewFilter.parentKey)?.label || "Branch";
    if(reviewFilter.level==="node"){
      const item=reviewQueue.find(x=>x.key===reviewFilter.nodeKey);
      return reviewGraphs[item?.view]?.nodes?.find(n=>n.id===item?.node_id)?.title ||
        item?.after?.title || "Node";
    }
    return "All";
  }
  function selectedReviewCategory(level,value){
    if(reviewFilter.level!==level)return false;
    return level==="all" ? true : level==="view" ? reviewFilter.view===value :
      level==="parent" ? reviewFilter.parentKey===value : reviewFilter.nodeKey===value;
  }
  function reviewCategoryRow(container,level,value,label,items,expand,children){
    const row=el("div","review-category-row level-"+level);
    if(children){
      const disclosure=el("button","review-category-disclosure",
                          reviewExpanded.has(expand)?"▾":"▸");
      disclosure.type="button";
      disclosure.setAttribute("aria-label","Expand "+label);
      disclosure.setAttribute("aria-expanded",String(reviewExpanded.has(expand)));
      disclosure.addEventListener("click",event=>{
        event.stopPropagation?.();
        if(reviewExpanded.has(expand))reviewExpanded.delete(expand);
        else reviewExpanded.add(expand);
        renderReviewCategories();
      });
      row.appendChild(disclosure);
    }else row.appendChild(el("span","review-category-spacer"));
    const button=el("button","review-category-choice",label);
    button.type="button";
    button.title=label;
    button.setAttribute("data-review-scope",level+":"+value);
    button.setAttribute("aria-selected",String(selectedReviewCategory(level,value)));
    button.addEventListener("click",()=>selectReviewCategory(
      level==="all"?{level:"all"}:
      level==="view"?{level,view:value}:
      level==="parent"?{level,parentKey:value}:
      {level:"node",nodeKey:value}
    ));
    row.appendChild(button);
    row.appendChild(reviewRing(items));
    row.appendChild(reviewCountDots(items));
    container.appendChild(row);
    if(children && reviewExpanded.has(expand)){
      const nested=el("div","review-category-children");
      container.appendChild(nested);
      children(nested);
    }
    return row;
  }
  function renderReviewCategories(){
    if(!review)return;
    const menu=$("atlasReviewMenu");
    menu.replaceChildren();
    reviewCategoryRow(menu,"all","all","All",reviewQueue,"all",root=>{
      for(const section of reviewGroups){
        const contents=reviewQueue.filter(item=>item.view===section.view);
        reviewCategoryRow(root,"view",section.view,section.title,contents,
                          "view:"+section.view,branchContainer=>{
          for(const branch of section.groups){
            reviewCategoryRow(branchContainer,"parent",branch.key,branch.label,
                              branch.items,"parent:"+branch.key,leafContainer=>{
              for(const item of branch.items){
                reviewCategoryRow(leafContainer,"node",item.key,
                  reviewGraphs[item.view]?.nodes?.find(n=>n.id===item.node_id)?.title ||
                  item.after?.title||item.before?.title||item.node_id,
                  [item],null,null);
              }
            });
          }
        });
      }
    });
    $("atlasReviewFilterLabel").textContent=reviewFilterLabel();
    const ring=$("atlasReviewFilterRing");
    ring.replaceChildren(reviewRing(filteredReviewQueue()));
    $("atlasReviewFilter").setAttribute("aria-expanded",String(reviewMenuOpen));
    menu.classList.toggle("is-hidden",!reviewMenuOpen);
  }
  function selectReviewCategory(next){
    reviewFilter=next;
    reviewMenuOpen=false;
    const queue=filteredReviewQueue();
    if(queue.length&&!queue.some(item=>
      item.view===config.view && item.node_id===config.focusId)){
      const target=queue.find(item=>proposalState(item)==="unchecked")||queue[0];
      navigate(target.node_id,target.view,false);
    }else updateReviewBar();
  }

  // A public GitHub Pages page has no identity allowed to write to a repository.
  // Browser drafts really persist as JSON in this browser, with a visible export;
  // only authenticated loopback Atlas saves the authoritative Knowledge JSON.
  function browserDraftKey(plan) {
    return "openresearch:atlas:proposal:v1:" + plan.id + ":" + plan.preview_sha256;
  }
  function browserDraftStorage() {
    try { return window.localStorage || null; }
    catch (_) { return null; }
  }
  function restoreBrowserDraft(plan) {
    const store = browserDraftStorage();
    if (!store) return;
    try {
      const raw=store.getItem(browserDraftKey(plan));
      if(!raw) return;
      const saved=JSON.parse(raw);
      if(saved.schema!=="openresearch.atlas.proposal-draft.v1" ||
         saved.plan_id!==plan.id || saved.preview_sha256!==plan.preview_sha256)
         return;
      for(const item of plan.items){
        const decision=saved.decisions?.[item.key];
        if(["approved","denied"].includes(decision))
          item.proposal_review={decision,actor:"Browser draft"};
      }
    } catch (_) { reviewDraftMessage="Could not load saved draft"; }
  }
  function saveBrowserDraft(key,decision) {
    const store=browserDraftStorage();
    if(!store) throw Error("Browser storage unavailable. Use local Atlas to save JSON.");
    const choices={};
    for(const item of review.plan.items){
      const val=item.key===key?decision:proposalState(item);
      if(val==="approved"||val==="denied") choices[item.key]=val;
    }
    const data={schema:"openresearch.atlas.proposal-draft.v1",
      plan_id:review.plan.id,preview_sha256:review.plan.preview_sha256,
      decisions:choices,updated_at:new Date().toISOString(),
      authority_applied:false};
    // Synchronous storage completes before the visible node state changes.
    store.setItem(browserDraftKey(review.plan),JSON.stringify(data));
    review.plan.items.find(item=>item.key===key).proposal_review=
      decision==="unchecked"?null:
      {decision,actor:"Browser draft",at:data.updated_at};
    reviewDraftMessage="Saved in browser";
  }
  function exportBrowserDraft() {
    if(!review || review.editable) return;
    const store=browserDraftStorage();
    const raw=store?.getItem(browserDraftKey(review.plan));
    if(!raw){reviewError="Nothing saved";updateReviewBar();return;}
    const data=JSON.parse(raw);
    const payload=JSON.stringify(data,null,2)+"\n";
    const url=URL.createObjectURL(new Blob([payload],{type:"application/json"}));
    const link=el("a");
    link.href=url;link.download=review.plan.id+".proposal-draft.json";
    document.body.appendChild(link);link.click();link.remove();
    setTimeout(()=>URL.revokeObjectURL(url),0);
  }
  function updateReviewBar() {
    if(!review) return;
    const bar=$("atlasReviewBar");
    bar.classList.remove("is-hidden");
    const items=filteredReviewQueue(),current=reviewPosition();
    $("atlasReviewCount").textContent=(current>=0?current+1:"–")+" / "+items.length;
    $("atlasReviewPrevUnchecked").disabled=reviewBusy||findUnchecked(-1)<0;
    $("atlasReviewNextUnchecked").disabled=reviewBusy||findUnchecked(1)<0;
    $("atlasReviewPrev").disabled=reviewBusy||current<=0;
    $("atlasReviewNext").disabled=reviewBusy||current>=items.length-1;
    $("atlasReviewApprove").disabled=reviewBusy||current<0;
    $("atlasReviewDeny").disabled=reviewBusy||current<0;
    $("atlasReviewUnchecked").disabled=reviewBusy||current<0;
    $("atlasReviewExport").classList.toggle("is-hidden", !!review.editable);
    $("atlasReviewExport").disabled = reviewBusy;
    const currentItem=items[current];
    const status=proposalState(currentItem);
    $("atlasReviewApprove").classList.toggle("chosen",status==="approved");
    $("atlasReviewDeny").classList.toggle("chosen",status==="denied");
    $("atlasReviewUnchecked").classList.toggle("chosen",status==="unchecked");
    renderReviewCategories();
    $("atlasReviewMessage").textContent = reviewBusy ? "Saving…" : reviewError ? reviewError :
       (!review.editable ? (reviewDraftMessage || "Browser draft") :
       (status === "unchecked" ? "Unchecked" :
        status === "approved" ? "Not implemented" : "Denied"));
  }
  function applyReviewResponse(data,live) {
    if (!data?.plan || !Array.isArray(data.plan.items) ||
        data.plan.id !== requestedReview()) throw Error("Invalid Atlas plan");
    if(!live) restoreBrowserDraft(data.plan);
    review = {plan:data.plan,context:data.context || data.review_context ||
              {nodes:[],edges:[]},editable:!!live && !!reviewKey()};
    overlayGraphs();
    buildReviewQueue();
    rebuildHierarchy();
    config.expanded = initialExpanded();
    render(false);
  }
  async function loadReview() {
    if(!requestedReview() || config.view==="core") return;
    let payload,live=false;
    try {
      const response=await fetch("/api/review/state",{cache:"no-store"});
      if(!response.ok) throw Error("Local review service unavailable");
      payload=await response.json();
      live=true;
    } catch (_) {
      // GitHub Pages is read-only. It cannot secretly persist human decisions.
      if(!["OR-Knowledge-Demo-20261026","OR-Atlas-MultiTree-Demo-20261026"]
          .includes(requestedReview())) {
        $("focusHint").textContent="Local Atlas review service required";
        return;
      }
      const response=await fetch("./demo-knowledge-plan.json",{cache:"no-store"});
      if(!response.ok) throw Error("Preview JSON missing");
      payload=await response.json();
      if(!payload.plan) payload={plan:payload,context:payload.review_context};
    }
    applyReviewResponse(payload,live);
    const key=getSearchFocus(),item=reviewQueue.find(x=>x.node_id===key && x.view===config.view);
    if(item) navigate(item.node_id,item.view,true);
    else {
      const first=reviewQueue.find(x=>x.view===config.view)||reviewQueue[0];
      if(first) navigate(first.node_id,first.view,true);
    }
    const bar=$("atlasReviewBar");
    $("atlasReviewPrevUnchecked").addEventListener("click",()=>reviewGoUnchecked(-1));
    $("atlasReviewPrev").addEventListener("click",()=>reviewGo(reviewPosition()-1));
    $("atlasReviewNext").addEventListener("click",()=>reviewGo(reviewPosition()+1));
    $("atlasReviewNextUnchecked").addEventListener("click",()=>reviewGoUnchecked(1));
    $("atlasReviewApprove").addEventListener("click",()=>submitReview("approved"));
    $("atlasReviewDeny").addEventListener("click",()=>submitReview("denied"));
    $("atlasReviewUnchecked").addEventListener("click",()=>submitReview("unchecked"));
    $("atlasReviewFilter").addEventListener("click",event=>{
      event.stopPropagation?.();
      reviewMenuOpen=!reviewMenuOpen;
      renderReviewCategories();
    });
    $("atlasReviewExport").addEventListener("click",exportBrowserDraft);
    // Existing Atlas tabs remain available in review mode without reloading.
    if (typeof document.querySelectorAll === "function")
      for(const tab of document.querySelectorAll(".tabs [data-graph]"))
        tab.addEventListener("click", event => {
          if(!review || tab.dataset.graph === "core") return;
          event.preventDefault();
          const choice=reviewQueue.find(item=>item.view===tab.dataset.graph);
          if(choice){
            reviewFilter={level:"view",view:tab.dataset.graph};
            navigate(choice.node_id,choice.view,false);
          }
        });
    bar.classList.remove("is-hidden");
    updateReviewBar();
  }
  function handleReviewShortcut(event) {
    if(!review || reviewBusy || !event || event.repeat ||
       event.altKey || event.ctrlKey || event.metaKey)return false;
    const active=event.target || document.activeElement;
    const tag=String(active?.tagName||"").toUpperCase();
    if(["INPUT","TEXTAREA","SELECT"].includes(tag) ||
       active?.isContentEditable)return false;
    const key=event.key;
    if(key==="Escape" && reviewMenuOpen){
      event.preventDefault?.();
      reviewMenuOpen=false;
      renderReviewCategories();
      return true;
    }
    if(!["Enter","Backspace","u","U","ArrowLeft","ArrowRight"].includes(key))
      return false;
    // Keep the standard keyboard behavior for focused inputs, links and
    // node/attribute controls; elsewhere Enter approves the selected node.
    if(key==="Enter" &&
       (["BUTTON","A","SUMMARY"].includes(tag) ||
        active?.attributes?.role==="button" ||
        active?.getAttribute?.("role")==="button"))return false;
    event.preventDefault?.();
    if(key==="Enter")submitReview("approved");
    else if(key==="Backspace")submitReview("denied");
    else if(key==="u" || key==="U")submitReview("unchecked");
    else if(key==="ArrowLeft"){
      if(event.shiftKey)reviewGo(reviewPosition()-1);
      else reviewGoUnchecked(-1);
    }else if(key==="ArrowRight"){
      if(event.shiftKey)reviewGo(reviewPosition()+1);
      else reviewGoUnchecked(1);
    }
    return true;
  }

  async function submitReview(decision) {
    if(!review || reviewBusy) return;
    const at=reviewPosition();
    if(at<0)return;
    const current=filteredReviewQueue()[at];
    reviewError="";
    reviewBusy=true;
    updateReviewBar();
    try {
      // Local Atlas commits Knowledge and Ledger before returning success.
      // Public Atlas persists a clearly labeled browser-only JSON draft.
      if(review.editable){
        const response=await fetch("/api/review/decision",{
          method:"POST",credentials:"same-origin",keepalive:true,
          headers:{"Content-Type":"application/json","X-Atlas-Review-Key":reviewKey()},
          body:JSON.stringify({revision:review.plan.revision,key:current.key,decision})
        });
        const data=await response.json();
        if(!response.ok) throw Error(data.error || "Review not saved");
        applyReviewResponse(data,true);
      } else {
        saveBrowserDraft(current.key,decision);
        overlayGraphs();
        rebuildHierarchy();
        render(false);
      }
      // Keep a purple-reset node visible. Approve/Deny advances within scope.
      if(decision!=="unchecked"){
        const next=findUnchecked(1);
        reviewBusy=false;
        if(next>=0)reviewGo(next);
      }
    } catch(error) {
      reviewError=review.editable?"Save failed":"Draft not saved";
      $("atlasReviewMessage").title=error.message;
    } finally {
      reviewBusy=false;
      updateReviewBar();
    }
  }

  const config = {
    view: document.body?.dataset?.atlasView || "core",
    focusId: null,
    mode: "overview",
    expanded: new Set(),
    query: "",
    zoom: 1,
    zoomInitialized: false,
    outlineOpen: false,
    detailsOpen: false,
    diff: null,
    expandedAttributes: new Set(),
    positions: new Map(),
    scene: { width: 700, height: 440 },
    visible: [],
    animateNextRender: false,
    renderedView: null
  };
  let parents = new Map(), children = new Map(), byId = new Map();
  let rootIds = [];
  let drag = null, bounceFrame = null, wheelStamp = 0;
  let outgoingScene = null;
  let lastViewport = { width: 0, height: 0 };

  function el(tag, className, value) {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (value !== undefined) item.textContent = String(value);
    return item;
  }
  function svg(tag, attrs, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = String(text);
    return node;
  }
  function kind(node) { return node?.kind || node?.type || "goal"; }
  function graph() { return reviewGraphs[config.view] || DATA[config.view] || { nodes: [], edges: [], gates: [] }; }
  function plain(text) {
    return String(text || "").replace(/^\s*(问题|方案|目标|定义|现状)[:：]\s*/, "").replace(/\s+/g, " ").trim();
  }
  function matches(node) {
    const q = config.query.trim().toLocaleLowerCase();
    return !q || [node.id, node.title, node.source_symbol, node.description, node.signature, node.path,
      ...(node.aliases || []), ...(node.term_ids || []), ...(node.context_ids || [])]
      .join(" ").toLocaleLowerCase().includes(q);
  }
  function order(a, b) {
    const weights = { question: 0, solve: 1, goal: 2, root: 0,
      group: 1, module: 2, class: 3, function: 4, method: 5, term: 3 };
    return (weights[kind(a)] ?? 6) - (weights[kind(b)] ?? 6)
      || String(a.title).localeCompare(String(b.title), "zh");
  }
  function rebuildHierarchy() {
    byId = new Map((graph().nodes || []).map(n => [n.id, n]));
    parents = new Map(); children = new Map();
    for (const edge of graph().edges || []) {
      if (!VIEWS[config.view].primary.includes(edge.type)) continue;
      if (!byId.has(edge.source) || !byId.has(edge.target) || parents.has(edge.target)) continue;
      parents.set(edge.target, edge.source);
      if (!children.has(edge.source)) children.set(edge.source, []);
      children.get(edge.source).push(edge.target);
    }
    // Child edges are intentionally rendered in stored top-to-bottom order.
    const preferred = VIEWS[config.view].root;
    rootIds = byId.has(preferred) ? [preferred] : [...byId.keys()].filter(id => !parents.has(id));
  }
  function ancestors(id) {
    const result = []; const seen = new Set();
    let cursor = id;
    while (byId.has(cursor) && !seen.has(cursor) && result.length <= 16) {
      seen.add(cursor); result.push(cursor);
      cursor = parents.get(cursor);
    }
    return result.reverse();
  }
  function initialExpanded() {
    const result = new Set(rootIds);
    if (["core", "knowledge"].includes(config.view)) {
      for (const id of rootIds) for (const c of children.get(id) || []) result.add(c);
    }
    if (config.view === "environment") {
      for (const n of byId.values()) if (kind(n) === "group") result.add(n.id);
    }
    return result;
  }
  function getSearchFocus() {
    const raw = (window.location?.search || "").match(/[?&]focus=([^&]+)/);
    if (!raw) return null;
    try { return decodeURIComponent(raw[1].replace(/\+/g, "%20")); }
    catch (_) { return null; }
  }
  function setAddress(focusId, replace) {
    if (!window.history?.pushState || !window.location?.pathname) return;
    // Review mode switches the URL and Atlas graph in place; no new HTML load.
    const path = review ? config.view + ".html" : window.location.pathname;
    const params=[];
    if(requestedReview()) params.push("review="+encodeURIComponent(requestedReview()));
    if(focusId) params.push("focus="+encodeURIComponent(focusId));
    const url = path + (params.length?"?"+params.join("&"):"") +
      (requestedReview()?window.location.hash||"":"");
    try {
      window.history[replace ? "replaceState" : "pushState"](
        { view: config.view, focusId: focusId || null }, "", url
      );
    } catch (_) { /* file:// mode may reject history; still allow local navigation */ }
  }
  function locateView(id) {
    if (byId.has(id)) return config.view;
    for (const key of Object.keys(VIEWS)) {
      if ((reviewGraphs[key]?.nodes || DATA[key]?.nodes || []).some(n => n.id === id)) return key;
    }
    return null;
  }
  function activateReviewView(view) {
    if (!review || view === "core" || !VIEWS[view] || !DATA[view]) return false;
    config.view = view;
    if(document.body?.dataset) document.body.dataset.atlasView = view;
    rebuildHierarchy();
    config.expanded = initialExpanded();
    config.expandedAttributes.clear();
    // The same DOM and canvas stay mounted across different trees.
    if(typeof document.querySelectorAll === "function")
      for(const tab of document.querySelectorAll(".tabs [data-graph]"))
        tab.classList.toggle("active", tab.dataset.graph === view);
    return true;
  }
  function siblingsOf(id) {
    const parent = parents.get(id);
    return parent ? (children.get(parent) || []).filter(key => byId.has(key)) :
      rootIds.filter(key => byId.has(key));
  }
  function adjacentSibling(direction) {
    if (!config.focusId) return null;
    const siblings = siblingsOf(config.focusId);
    const at = siblings.indexOf(config.focusId);
    return at >= 0 ? siblings[at + direction] || null : null;
  }
  function moveSibling(direction) {
    if (reviewBusy) return false;
    const next = adjacentSibling(direction);
    if (!next) return false;
    return navigate(next, config.view, false);
  }
  function moveReviewOrSibling(direction) {
    if (review) {
      const at = reviewPosition();
      if(at < 0)return false;
      const next = at + direction;
      if(next < 0 || next >= filteredReviewQueue().length)return false;
      reviewGo(next);
      return true;
    }
    return moveSibling(direction);
  }
  // Navigation selects a compact node. Only an explicit card click toggles
  // its explanation; wheel, keyboard and review arrows never expand it.
  function navigate(id, view, replace, options={}) {
    const target = view || locateView(id);
    if (!target || !(reviewGraphs[target]?.nodes || DATA[target]?.nodes || []).some(n => n.id === id)) return false;
    const sameFocus = target===config.view && config.focusId===id;
    const wasExpanded = sameFocus && config.expandedAttributes.has(id);
    if (target !== config.view) {
      if (review) {
        // Fast cross-tree page turn: preserve scroll/zoom and shared draft state.
        if (!activateReviewView(target)) return false;
      } else {
        if(window.location) window.location.href = target+".html?"+
          (requestedReview()?"review="+encodeURIComponent(requestedReview())+"&":"")+
          "focus="+encodeURIComponent(id)+(requestedReview()?window.location.hash||"":"");
        return true;
      }
    }
    config.mode = "focus";
    config.focusId = id;
    config.query = "";
    $("search").value = "";
    config.outlineOpen = false;
    config.detailsOpen = false;
    for (const key of ancestors(id)) config.expanded.add(key);
    // A selected node is the only one permitted to hold open attributes.
    config.expandedAttributes.clear();
    if (options.toggle && !wasExpanded) config.expandedAttributes.add(id);
    config.animateNextRender = true;
    setAddress(id, !!replace || sameFocus);
    render(false);
    if(review) updateReviewBar();
    return true;
  }
  function overview(push) {
    config.expandedAttributes.clear();
    config.animateNextRender = true;
    config.mode = "overview"; config.focusId = null; config.query = "";
    config.detailsOpen = false;
    $("search").value = "";
    if (push) setAddress(null, false);
    render(true);
  }
  function openCross(id) { navigate(id, locateView(id), false); }
  function init() {
    if (!VIEWS[config.view] || !DATA[config.view]) config.view = "core";
    rebuildHierarchy();
    config.expanded = initialExpanded();
    const deep = getSearchFocus();
    if (deep && byId.has(deep)) {
      config.mode = "focus"; config.focusId = deep;
    }
    bind();
    render(true);
    if(requestedReview() && config.view!=="core") {
      loadReview().catch(error=>{$("focusHint").textContent=error.message;});
    }
  }

  // SVG text wrapping uses a conservative display-width estimate for CJK and Latin.
  function charWidth(character, fontSize) {
    const code = character.codePointAt(0) || 0;
    if (code >= 0x2e80 || code >= 0x1100 && code <= 0x11ff) return fontSize;
    if (character === " ") return fontSize * .36;
    if ("il.,:;'!|".includes(character)) return fontSize * .42;
    if ("MW@%".includes(character)) return fontSize * .91;
    return fontSize * .65;
  }
  function wrap(text, width, fontSize, maxLines) {
    const chars = Array.from(plain(text));
    if (!chars.length) return [];
    const lines = []; let line = "", used = 0, at = 0;
    while (at < chars.length && lines.length < maxLines) {
      const ch = chars[at], w = charWidth(ch, fontSize);
      if (line && used + w > width) {
        lines.push(line.trim()); line = ""; used = 0;
      } else {
        line += ch; used += w; at++;
      }
    }
    if (line && lines.length < maxLines) lines.push(line.trim());
    if (at < chars.length && lines.length) {
      const i = lines.length - 1;
      lines[i] = lines[i].replace(/.{1,2}$/u, "") + "…";
    }
    return lines;
  }
  function measure(node, focused) {
    // Pending/approved/denied nodes use the SAME size and expansion rules
    // as source-backed Atlas nodes. Status affects only their colors.
    const expanded = config.expandedAttributes.has(node.id);
    const w = expanded ? 525 : 330;
    const title = wrap(node.title || node.id, w - 45, 14.2, 3);
    const body = wrap(node.summary || node.description, w - 42, 11.8, 4);
    const base = Math.max(108, 22 + title.length * 19
      + (body.length ? 10 + body.length * 17 : 0) + 18);
    const isClass=(node.attributes||[]).some(x=>x.type==="class_methods");
    const count=(node.attributes||[]).length;
    const terms = (node.context_term_ids || []).length;
    const extraH = expanded ? Math.min(760, Math.max(340, 170 + terms * 65 +
      Math.min(count, 5) * 46 + (isClass ? 180 : 0))) : 0;
    return { w, h:base + extraH, title, body, baseH:base,
      extraH, expanded };
  }
  function visibleOverview() {
    const needed = new Set();
    if (config.query.trim()) {
      for (const n of byId.values()) if (matches(n)) {
        for (const id of ancestors(n.id)) needed.add(id);
      }
    }
    const seen = new Set(), list = [];
    function visit(id, depth) {
      if (!byId.has(id) || seen.has(id) || depth > 13) return;
      if (config.query && !needed.has(id)) return;
      seen.add(id); list.push({ id, depth, role: "overview" });
      if (config.query || config.expanded.has(id)) {
        for (const child of children.get(id) || []) visit(child, depth + 1);
      }
    }
    for (const id of rootIds) visit(id, 0);
    return list;
  }

  function layoutOverview() {
    const visible = visibleOverview();
    const allowed = new Set(visible.map(v => v.id));
    const sizes = new Map(), blocks = new Map(), positions = new Map();
    const VERTICAL_GAP = 24, HORIZONTAL_GAP = 72, ORIGIN_X = 36;
    const columnWidths=new Map();
    for(const entry of visible){
      const width=measure(byId.get(entry.id),false).w;
      columnWidths.set(entry.depth,Math.max(columnWidths.get(entry.depth)||270,width));
    }
    const xColumns=[ORIGIN_X];
    const maxDepth=Math.max(0,...visible.map(x=>x.depth));
    for(let depth=1;depth<=maxDepth;depth++){
      xColumns[depth]=xColumns[depth-1]+(columnWidths.get(depth-1)||270)+HORIZONTAL_GAP;
    }
    function height(id) {
      if (blocks.has(id)) return blocks.get(id);
      const kids = (children.get(id) || []).filter(x => allowed.has(x));
      let total = kids.reduce((sum, child) => sum + height(child), 0)
        + Math.max(0, kids.length - 1) * VERTICAL_GAP;
      const own = sizes.get(id) || measure(byId.get(id), false);
      sizes.set(id, own);
      const result = Math.max(own.h, total);
      blocks.set(id, result);
      return result;
    }
    function place(id, depth, top) {
      const size = sizes.get(id), block = blocks.get(id);
      const centerY = top + block / 2;
      positions.set(id, { x: xColumns[depth],
        y: centerY - size.h / 2, ...size, depth });
      const kids = (children.get(id) || []).filter(x => allowed.has(x));
      const total = kids.reduce((sum, child) => sum + blocks.get(child), 0)
        + Math.max(0, kids.length - 1) * VERTICAL_GAP;
      let childTop = top + (block - total) / 2;
      for (const child of kids) {
        place(child, depth + 1, childTop);
        childTop += blocks.get(child) + VERTICAL_GAP;
      }
    }
    let y = 34;
    for (const id of rootIds) if (allowed.has(id)) {
      const h = height(id); place(id, 0, y); y += h + 28;
    }
    return { visible, positions };
  }

  function layoutFocus() {
    const id = config.focusId;
    if (!byId.has(id)) return layoutOverview();
    const chain = ancestors(id);
    const ancestorsShown = chain.slice(Math.max(0, chain.length - 3), -1);
    const kids = (children.get(id) || []).filter(x => byId.has(x));
    const siblings = siblingsOf(id), at = siblings.indexOf(id);
    const peerIds = [siblings[at-1],siblings[at+1]].filter(Boolean)
      .filter(key => !ancestorsShown.includes(key) && !kids.includes(key));
    const ids = [...ancestorsShown, id, ...kids, ...peerIds];
    const positions = new Map();
    const horizontalGap = 72, xStart = 36;
    const focusSize = measure(byId.get(id), true);
    const childSizes = kids.map(k => measure(byId.get(k), false));
    const childTotal = childSizes.reduce((sum, s) => sum + s.h, 0)
      + Math.max(0, childSizes.length - 1) * 20;
    const center = 38 + Math.max(focusSize.h, childTotal) / 2;
    const depthOffset = ancestorsShown.length;
    const ancestorSizes=ancestorsShown.map(key=>measure(byId.get(key),false));
    const columns=[xStart];
    for(let i=1;i<=depthOffset;i++){
      columns[i]=columns[i-1]+ancestorSizes[i-1].w+horizontalGap;
    }
    const ownX=columns[depthOffset];
    positions.set(id, { x: ownX, y: center - focusSize.h / 2,
      ...focusSize, depth: depthOffset, role: "focus" });
    ancestorsShown.forEach((key, i) => {
      const size=ancestorSizes[i];
      positions.set(key, { x: columns[i],
        y: center - size.h / 2, ...size, depth: i, role: "ancestor" });
    });
    let y = center - childTotal / 2;
    kids.forEach((child, index) => {
      const size = childSizes[index];
      positions.set(child, { x: ownX + focusSize.w + horizontalGap,
        y, ...size, depth: depthOffset + 1, role: "child" });
      y += size.h + 20;
    });
    // Ghost siblings share the selected node's parent, not a fictional edge.
    for(const peer of peerIds) {
      const size = measure(byId.get(peer), false);
      const direction = siblings.indexOf(peer) < at ? -1 : 1;
      positions.set(peer, {x:ownX,
        y:center - focusSize.h/2 + (direction < 0 ?
          -size.h-26 : focusSize.h+26),
        ...size, depth:depthOffset, role:"sibling-preview"});
    }
    return { visible: ids.map(key => ({
      id: key, depth: positions.get(key).depth, role: positions.get(key).role
    })), positions };
  }
  function normalizeLayout(layout) {
    let minX = 0, minY = 0, maxX = 0, maxY = 0;
    for (const p of layout.positions.values()) {
      minX = Math.min(minX,p.x); minY = Math.min(minY,p.y);
      maxX = Math.max(maxX,p.x+p.w); maxY = Math.max(maxY,p.y+p.h);
    }
    if(minX<0 || minY<0) {
      const dx = minX<0 ? 35-minX : 0, dy = minY<0 ? 35-minY : 0;
      for(const p of layout.positions.values()){p.x+=dx;p.y+=dy;}
      maxX+=dx;maxY+=dy;
    }
    // Give the selected card enough scene padding to be exactly centered,
    // even for the first/last node or a tree narrower than the viewport.
    if(config.mode==="focus" && layout.positions.has(config.focusId)){
      const v=$("canvasWrap"),zoom=config.zoom||1;
      const halfW=(Number(v.clientWidth)||1150)/(2*zoom);
      const halfH=(Number(v.clientHeight)||630)/(2*zoom);
      const focus=layout.positions.get(config.focusId);
      const dx=Math.max(0,halfW-focus.x-focus.w/2);
      const dy=Math.max(0,halfH-focus.y-focus.h/2);
      if(dx||dy){
        for(const p of layout.positions.values()){p.x+=dx;p.y+=dy;}
        maxX+=dx;maxY+=dy;
      }
      maxX=Math.max(maxX,focus.x+focus.w/2+halfW);
      maxY=Math.max(maxY,focus.y+focus.h/2+halfH);
    }
    return { ...layout,width:Math.max(600,maxX+35),
      height:Math.max(300,maxY+35) };
  }
  function textLines(group, lines, x, startY, lineHeight, className) {
    for (let i = 0; i < lines.length; i++) {
      group.appendChild(svg("text", { x, y: startY + i * lineHeight,
        class: className }, lines[i]));
    }
  }
  function drawCard(node, p, layer, positions, focus) {
    const k = kind(node), chosen = node.id === config.focusId;
    const classes = ["graph-node", "node-" + k, chosen ? "node-selected" : "",
      p.expanded ? "node-expanded" : "",
      p.role === "sibling-preview" ? "node-sibling-preview" : "",
      config.diff && config.view === "code" &&
        (config.diff.added_ids || []).includes(node.id) ? "node-added" : "",
      config.diff && config.view === "code" &&
        (config.diff.changed || []).some(x => x.id === node.id) ? "node-changed" : ""
    ].filter(Boolean).join(" ");
    const attrs = {
      class: classes, transform: "translate(" + p.x + "," + p.y + ")",
      role: "button", tabindex: "0",
      "aria-label": node.title + ": " + plain(node.description || ""),
      "data-node-id": node.id, "data-status": node.status || "unknown"
    };
    // Even an empty data-review attribute activates review CSS. Only reviewed
    // nodes may have it; normal Atlas cards must keep their original font size.
    if (review && itemFor(node.id)) attrs["data-review"] = proposalState(itemFor(node.id));
    const card = svg("g", attrs);
    const state=review && itemFor(node.id) ? proposalState(itemFor(node.id)) : node.status || "unknown";
    const theme=NODE_THEME[state==="approved"?"not_implemented":state] || NODE_THEME.unknown;
    card.style.setProperty?.("--node-rgb",theme.rgb);
    card.style.setProperty?.("--node-accent",theme.accent);
    card.style.setProperty?.("--node-ink",theme.ink);
    card.appendChild(svg("title", {}, (node.title || "") +
      (node.status ? " — " + node.status : "") + ". " + plain(node.description || "")));
    card.appendChild(svg("rect", {
      class: "node-surface", rx: 10, width: p.w, height: p.h,
      fill: review && itemFor(node.id) ?
        REVIEW_PALETTE[proposalState(itemFor(node.id))].fill :
        STATE_FILL[node.status] || STATE_FILL.unknown
    }));
    // Context ancestors stay neutral during a proposal review. In particular,
    // a partial child must not paint its parent branch yellow.
    const status = review && itemFor(node.id) ?
      REVIEW_PALETTE[proposalState(itemFor(node.id))].stroke :
      STATES[node.status];
    if (status) card.appendChild(svg("rect", {
      class: "node-status", rx: 2, x: 0, y: 9,
      width: 4, height: p.h - 18, fill: status
    }));
    // Typography and text layout are identical in normal and review modes.
    const titleX=17, top=23;
    textLines(card, p.title, titleX, top, 19, "node-title");
    if (p.body.length) {
      const contentY=top+p.title.length*19+7;
      textLines(card,p.body,titleX,contentY,17,"node-description");
    }
    const onOpen = () => navigate(node.id,config.view,false,{toggle:true});
    if(p.expanded){
      const embedded=svg("foreignObject",{
        x:10,y:p.baseH+8,width:p.w-20,height:p.extraH-41,
        class:"node-attributes-foreign"
      });
      const content=el("div","node-attributes-inner");
      content.setAttribute("aria-label",node.title+" 的完整解释与证据");
      const intro=attributeSection(content,"这个节点在做什么");
      intro.appendChild(el("p","node-attribute-text",
        node.reader_explanation || plain(node.description || node.summary || "")));
      const termIds=node.context_term_ids || [];
      if(termIds.length){
        const terms=attributeSection(content,"专业词汇 · Context");
        for(const id of termIds){
          const term=(reviewGraphs.context?.nodes || DATA.context?.nodes || []).find(x=>x.id===id);
          if(!term)continue;
          const row=el("div","node-term-row");
          const link=el("a","node-term-link",term.title);
          link.href="context.html?"+(review?"review="+encodeURIComponent(requestedReview())+"&":"")+
                    "focus="+encodeURIComponent(id)+(review?window.location.hash||"":"");
          link.title="在 Context 中查看“"+term.title+"”";
          if(review) link.addEventListener("click",event=>{
            event.preventDefault();
            navigate(id,"context",false);
          });
          row.appendChild(link);
          row.appendChild(el("p","node-term-definition",plain(term.description)));
          terms.appendChild(row);
        }
      }
      for(const block of node.attributes||[])renderAttributeBlock(content,block,node);
      embedded.appendChild(content);
      // An expanded foreignObject covers much of the card. Its old unconditional
      // stopPropagation swallowed blank-area clicks, leaving the card open.
      // Explicitly route those clicks through the SAME toggle as the SVG body.
      embedded.addEventListener("click",event=>{
        event.stopPropagation?.();
        if(event.defaultPrevented || (event.button!==undefined && event.button!==0))return;
        // Links, accordions, form controls and selectable code keep their
        // native behavior rather than closing the whole card.
        if(event.target?.closest?.(
          "a,button,summary,input,select,textarea,pre,code,[role='button'],[contenteditable]"
        ))return;
        // Drag-selecting explanation text must not collapse its container.
        const selection=window.getSelection?.();
        if(selection && !selection.isCollapsed && String(selection.toString?.()||"").length)return;
        onOpen();
      });
      embedded.addEventListener("keydown",event=>event.stopPropagation?.());
      card.appendChild(embedded);
    }
    const count = (children.get(node.id) || []).length;
    if (count) {
      const label = svg("g", { class: "child-count" });
      label.appendChild(svg("rect", {
        x: p.w - 41, y: p.h - 25, width: 29, height: 18, rx: 9
      }));
      label.appendChild(svg("text", {
        x: p.w - 26.5, y: p.h - 12, "text-anchor": "middle"
      }, String(count)));
      // The count badge expands this branch without changing zoom or focus.
      label.setAttribute("role", "button");
      label.setAttribute("tabindex", "0");
      label.setAttribute("aria-label", "展开或收起子节点");
      label.addEventListener("click", event => {
        event.stopPropagation?.();
        if (config.expanded.has(node.id)) config.expanded.delete(node.id);
        else config.expanded.add(node.id);
        render(false);
      });
      label.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault?.();
          event.stopPropagation?.();
          if (config.expanded.has(node.id)) config.expanded.delete(node.id);
          else config.expanded.add(node.id);
          render(false);
        }
      });
      card.appendChild(label);
    }
    {
      const toggle = svg("g", {
        class: "attribute-trigger", role: "button", tabindex: "0",
        "data-attribute-toggle": node.id,
        "aria-label": "展开 " + node.title + " 的属性",
      });
      const countWidth = (children.get(node.id) || []).length ? 48 : 13;
      const buttonWidth = p.w - countWidth - 14;
      toggle.appendChild(svg("rect",{
        class:"attr-hit",x:11,y:p.h-28,width:buttonWidth,height:22,rx:6
      }));
      const shortFunction = node.key_function ?
        node.key_function.split(".").slice(-1)[0] : null;
      toggle.appendChild(svg("text",{
        x:20,y:p.h-13
      },p.expanded ? "收起节点属性" :
         shortFunction ? "查看功能说明与源码依据" :
         "展开节点说明与依据"));
      const show = event => {
        event.preventDefault?.();
        event.stopPropagation?.();
        toggleAttributes(node.id);
      };
      toggle.addEventListener("click",show);
      toggle.addEventListener("keydown",event=>{
        if (event.key==="Enter" || event.key===" ") show(event);
      });
      card.appendChild(toggle);
    }
    card.addEventListener("click", onOpen);
    // Wheel on a compact node turns actual siblings. Expanded HTML content
    // retains its native scroll; blank canvas keeps native scrolling.
    card.addEventListener("wheel", event => {
      if(!config.focusId || !event.deltaY || event.ctrlKey ||
         event.target?.closest?.(".node-attributes-inner,textarea,input,select,pre"))return;
      if(node.id !== config.focusId &&
         !["sibling-preview"].includes(p.role))return;
      const now = Date.now();
      if(now-wheelStamp<180){event.preventDefault?.();return;}
      const next = adjacentSibling(event.deltaY>0?1:-1);
      if(!next)return;
      event.preventDefault?.();
      wheelStamp=now;
      navigate(next,config.view,false);
    },{passive:false});
    card.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault(); onOpen();
      }
    });
    layer.appendChild(card);
  }

  function drawSVG(layout) {
    const drawing = $("diagram");
    const prior = new Map();
    const oldCards=drawing.children?.[1]?.children || [];
    for(const old of oldCards){
      const id=old.getAttribute?.("data-node-id")||old.attributes?.["data-node-id"];
      if(id)prior.set(id,old.getBoundingClientRect?.()||null);
    }
    const transition=!!config.animateNextRender && config.renderedView===config.view &&
      !window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
    drawing.replaceChildren();
    const lines = svg("g", { class: "edges" });
    const cards = svg("g", { class: "cards" });
    const position = layout.positions;
    function link(source, target) {
      const a = position.get(source), b = position.get(target);
      if (!a || !b) return;
      const x1 = a.x + a.w, x2 = b.x;
      const y1 = a.y + a.h / 2, y2 = b.y + b.h / 2;
      if (x2 <= x1) return;
      const mid = (x1 + x2) / 2;
      lines.appendChild(svg("path", {
        class: "edge-main",
        d: "M " + x1 + " " + y1 + " C " + mid + " " + y1 +
          ", " + mid + " " + y2 + ", " + x2 + " " + y2
      }));
    }
    for (const item of layout.visible) {
      const p = parents.get(item.id);
      if (p) link(p, item.id);
    }
    // In focused mode siblings are vertically stacked and share a parent.
    if(config.mode==="focus")for(const item of layout.visible){
      if(item.role!=="sibling-preview")continue;
      const peer=position.get(item.id),selected=position.get(config.focusId);
      if(!peer||!selected)continue;
      const x=selected.x+selected.w/2, a=peer.y+peer.h/2,
            b=selected.y+selected.h/2;
      lines.appendChild(svg("path",{class:"edge-sibling",
        d:"M "+x+" "+a+" L "+x+" "+b}));
    }
    for (const item of layout.visible) {
      const n = byId.get(item.id), p = position.get(item.id);
      if (n && p) drawCard(n, p, cards, position, config.mode === "focus");
    }
    drawing.append(lines, cards);
    drawing.setAttribute("viewBox", "0 0 " + layout.width + " " + layout.height);
    // Stable-card FLIP: move existing identities from their previous screen
    // location without scaling text or sacrificing native card interactions.
    if(transition && typeof window.requestAnimationFrame==="function"){
      const moved=[];
      for(const node of cards.children){
        const id=node.getAttribute?.("data-node-id")||node.attributes?.["data-node-id"];
        const a=prior.get(id), b=node.getBoundingClientRect?.();
        if(!a||!b||!Number.isFinite(a.left)||!Number.isFinite(b.left))continue;
        const dx=a.left-b.left,dy=a.top-b.top;
        if(Math.abs(dx)<1&&Math.abs(dy)<1)continue;
        node.style.transition="none";
        node.style.translate=dx+"px "+dy+"px";
        moved.push(node);
      }
      window.requestAnimationFrame(()=>{
        for(const node of moved){
          node.style.transition="translate 380ms cubic-bezier(.22,1,.36,1), opacity 280ms ease";
          node.style.translate="0px 0px";
        }
      });
    }
    config.renderedView=config.view;
    config.animateNextRender=false;
    config.positions = position;
    config.scene = { width: layout.width, height: layout.height };
  }
  function fitZoom() {
    const v = $("canvasWrap");
    const w = Number(v.clientWidth) || 1150;
    const h = Number(v.clientHeight) || 630;
    const { width, height } = config.scene;
    const a = (w - 56) / width, b = (h - 48) / height;
    return Math.max(0.82, Math.min(1, a, b));
  }
  function centerSelected() {
    const v = $("canvasWrap");
    const p = config.positions.get(config.focusId);
    if (!p) return;
    const width = Number(v.clientWidth) || 1150;
    const height = Number(v.clientHeight) || 630;
    const x = config.zoom * (p.x + p.w / 2) - width / 2;
    const y = config.zoom * (p.y + p.h / 2) - height / 2;
    if (typeof v.scrollTo === "function") {
      v.scrollTo({ left: Math.max(0, x), top: Math.max(0, y),
        behavior: window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ? "auto" : "smooth" });
    }
  }
  function setScale() {
    const drawing = $("diagram");
    drawing.setAttribute("width", Math.round(config.scene.width * config.zoom));
    drawing.setAttribute("height", Math.round(config.scene.height * config.zoom));
    $("zoomValue").textContent = Math.round(config.zoom * 100) + "%";
  }
  function rowButton(id, label, indent) {
    const button = el("button", "outline-item" +
      (config.focusId === id ? " selected" : ""), label);
    button.style.paddingLeft = (13 + indent * 12) + "px";
    button.title = byId.get(id)?.description || label;
    button.addEventListener("click", () => navigate(id, config.view, false));
    return button;
  }
  function renderOutline() {
    const output = $("outline");
    output.replaceChildren();
    const list = visibleOverview();
    if (config.mode === "focus" && config.focusId) {
      const path = ancestors(config.focusId);
      const ids = new Set([...path, ...(children.get(config.focusId) || [])]);
      for (const id of ids) output.appendChild(rowButton(id, byId.get(id).title,
        Math.min(4, path.indexOf(id) >= 0 ? path.indexOf(id) : path.length)));
    } else {
      for (const item of list) output.appendChild(rowButton(item.id, byId.get(item.id).title, item.depth));
    }
    if (!output.children.length) output.appendChild(el("p", "empty-message", "尝试其他检索词"));
  }
  function setDrawerState() {
    $("sidebar").classList.toggle("is-hidden", !config.outlineOpen);
    $("detailDrawer").classList.toggle("is-hidden", !config.detailsOpen);
    $("detailsToggle").disabled = !config.focusId;
    $("overviewBack").classList.toggle("is-hidden", !config.focusId);
  }
  function updateBreadcrumb() {
    const area = $("breadcrumb");
    area.replaceChildren();
    if (!config.focusId) {
      area.appendChild(el("span", "crumb-current", config.view[0].toUpperCase() + config.view.slice(1)));
      return;
    }
    const path = ancestors(config.focusId);
    const visiblePath = path.slice(-3);
    if (path.length > 3) area.appendChild(el("span", "crumb-ellipsis", "…"));
    for (let i = 0; i < visiblePath.length; i++) {
      if (i) area.appendChild(el("span", "crumb-divider", "/"));
      const id = visiblePath[i], n = byId.get(id);
      if (id === config.focusId) area.appendChild(el("span", "crumb-current", n.title));
      else {
        const button = el("button", "crumb-link", n.title);
        button.addEventListener("click", () => navigate(id, config.view, false));
        area.appendChild(button);
      }
    }
  }
  function section(area, label) {
    const block = el("section", "context-section");
    block.appendChild(el("h4", "", label));
    area.appendChild(block);
    return block;
  }
  function refLink(id, label) {
    const target = locateView(id);
    const button = el("button", "context-link", label);
    button.addEventListener("click", () => openCross(id));
    button.title = target ? target + " · " + id : "Unknown node";
    return button;
  }
  function sourceHref(path, line) {
    if (typeof path !== "string" || !path ||
        path.startsWith("/") || path.includes("..") || path.includes("\\"))
      return null;
    const suffix = typeof line === "number" && line > 0 ? "#L" + line : "";
    return "https://github.com/OpenSAE/SAEBase/blob/main/" +
      path.split("/").map(encodeURIComponent).join("/") + suffix;
  }
  function diagramShape(group, tag, attributes, value) {
    const element = svg(tag,attributes,value);
    group.appendChild(element);
    return element;
  }
  function attributeDiagram(type) {
    const diagram = svg("svg",{viewBox:"0 0 390 128",class:"attribute-diagram",
      role:"img", "aria-label":"可核对的科研概念示意"});
    if (type==="residual") {
      diagramShape(diagram,"rect",{x:10,y:46,width:62,height:34,rx:7,class:"diagram-box"});
      diagramShape(diagram,"text",{x:41,y:67,"text-anchor":"middle"},"输入 x");
      diagramShape(diagram,"path",{d:"M72 63 L106 63 L106 31 L128 31",class:"diagram-edge"});
      diagramShape(diagram,"rect",{x:128,y:14,width:129,height:36,rx:7,class:"diagram-box"});
      diagramShape(diagram,"text",{x:192,y:36,"text-anchor":"middle"},"残差变换 F(x)");
      diagramShape(diagram,"path",{d:"M257 32 L288 32 L288 63",class:"diagram-edge"});
      diagramShape(diagram,"path",{d:"M72 63 L288 63",class:"diagram-edge"});
      diagramShape(diagram,"rect",{x:275,y:52,width:28,height:24,rx:12,class:"diagram-sum"});
      diagramShape(diagram,"text",{x:289,y:68,"text-anchor":"middle"},"+");
      diagramShape(diagram,"path",{d:"M303 64 L329 64",class:"diagram-edge"});
      diagramShape(diagram,"text",{x:356,y:67,"text-anchor":"middle"},"输出");
      diagramShape(diagram,"text",{x:185,y:89,"text-anchor":"middle"},"捷径连接");
    }
    return diagram;
  }

  function attributeSection(parent,title) {
    const group=el("section","node-attribute-section");
    group.appendChild(el("h4","node-attribute-heading",title));
    parent.appendChild(group);
    return group;
  }
  function attributeCode(parent,value) {
    const pre=el("pre","node-attribute-code",value||"");
    parent.appendChild(pre);
    return pre;
  }
  function sourceAnchor(parent,path,line,title) {
    const href=sourceHref(path,line);
    if(!href)return;
    const a=el("a","node-attribute-link",title||"查看实体源码");
    a.href=href;a.target="_blank";a.rel="noopener noreferrer";
    parent.appendChild(a);
  }
  function renderParameterList(parent,parameters) {
    if(!parameters?.length) return;
    const section=attributeSection(parent,"参数与作用");
    const list=el("div","node-parameter-list");
    for(const parameter of parameters) {
      const row=el("div","node-parameter-row");
      const name=el("code","node-parameter-name",parameter.name);
      row.appendChild(name);
      const body=el("div","node-parameter-info");
      body.appendChild(el("p","node-parameter-meaning",parameter.meaning||"输入参数"));
      const tech=[parameter.annotation,parameter.default!==null&&parameter.default!==undefined?
        "默认值 "+parameter.default:""].filter(Boolean);
      if(tech.length)body.appendChild(el("p","node-parameter-extra",tech.join(" · ")));
      row.appendChild(body);list.appendChild(row);
    }
    section.appendChild(list);
  }
  function renderInterface(parent,info,sourcePath) {
    if(!info)return;
    attributeSection(parent,"主要作用").appendChild(
      el("p","node-attribute-text",info.purpose||"现有源代码实现的处理流程。"));
    attributeSection(parent,"函数签名").appendChild(
      el("code","node-attribute-signature",info.signature||""));
    renderParameterList(parent,info.parameters);
    if(info.returns){
      const ret=attributeSection(parent,"返回值");
      const description=(info.returns.name||"return")+" · "+
        (info.returns.annotation||"源代码表达式");
      ret.appendChild(el("code","node-attribute-signature",description));
      ret.appendChild(el("p","node-attribute-text",info.returns.meaning||""));
      if(info.returns.expressions?.length){
        attributeCode(ret,info.returns.expressions.join("\n"));
      }
    }
    if(info.code){
      const block=attributeSection(parent,"实际代码主要部分");
      attributeCode(block,info.code);
      sourceAnchor(block,sourcePath,info.line,"查看完整函数源码");
    }
  }
  function renderAttributeBlock(parent,attr,node) {
    if(!attr || typeof attr.label!=="string")return;
    if(attr.type==="class_methods") {
      const section=attributeSection(parent,attr.label);
      for(const item of attr.methods||[]) {
        const details=el("details","node-method-details");
        const heading=el("summary","node-method-summary",item.name);
        details.appendChild(heading);
        const body=el("div","node-method-body");
        renderInterface(body,item,node.path);
        details.appendChild(body);
        section.appendChild(details);
      }
      return;
    }
    if(attr.type==="fields") {
      const section=attributeSection(parent,attr.label);
      for(const field of attr.fields||[]){
        const row=el("div","node-field-row");
        row.appendChild(el("code","node-parameter-name",field.name));
        row.appendChild(el("span","node-attribute-text",field.meaning||"成员数据"));
        if(field.assignment)row.appendChild(
          el("code","node-field-assignment",field.assignment));
        section.appendChild(row);
      }
      return;
    }
    if(attr.type==="module_contents") {
      const section=attributeSection(parent,attr.label);
      for(const [label,values] of [
        ["类",attr.classes||[]],["函数",attr.functions||[]]]){
        if(values.length)section.appendChild(
          el("p","node-attribute-text",label+"："+values.join(" · ")));
      }
      return;
    }
    if(attr.type==="interface"){
      renderInterface(parent,attr.interface,node.path);
      return;
    }
    if(attr.type==="code_reference"){
      const section=attributeSection(parent,attr.label);
      if(attr.interface)renderInterface(section,attr.interface,attr.path);
      else if(attr.code)attributeCode(section,attr.code);
      sourceAnchor(section,attr.path,attr.line,"查看源文件");
      if(attr.symbol_id){
        const link=el("a","node-attribute-link","在 Code Tree 中查看对应实现");
        link.href="code.html?focus="+encodeURIComponent(attr.symbol_id);
        section.appendChild(link);
      }
      return;
    }
    const section=attributeSection(parent,attr.label);
    if(attr.type==="diagram"){
      section.appendChild(attributeDiagram(attr.diagram));
      if(attr.caption)section.appendChild(
        el("p","node-attribute-note",attr.caption));
    } else if(attr.type==="formula") {
      section.appendChild(el("div","node-attribute-formula",attr.value||""));
      if(attr.note)section.appendChild(el("p","node-attribute-note",attr.note));
    } else if(attr.type==="code") {
      attributeCode(section,attr.value||"");
    } else if(attr.type==="function") {
      section.appendChild(el("code","node-attribute-signature",attr.signature||attr.value||""));
    } else if(attr.type==="reference"){
      sourceAnchor(section,attr.path,attr.line,"查看完整来源");
    } else {
      section.appendChild(el("p","node-attribute-text",attr.value||""));
    }
    if(attr.source)sourceAnchor(section,attr.source,attr.line,"查看依据");
  }
  function toggleAttributes(id){
    navigate(id,config.view,false,{toggle:true});
    return $("diagram");
  }

  function renderDetails() {
    const area = $("detailBody"); area.replaceChildren();
    const n = byId.get(config.focusId);
    if (!n) {
      area.appendChild(el("p", "empty-message", "选择节点查看证据与来源"));
      return;
    }
    area.appendChild(el("h2", "detail-title", n.title));
    const status = el("div", "detail-status");
    const swatch = el("span", "detail-status-swatch");
    const proposal=itemFor(n.id);
    swatch.style.background = proposal
      ? REVIEW_PALETTE[proposalState(proposal)].stroke
      : STATES[n.status] || "#adb2b9";
    status.appendChild(swatch);
    status.appendChild(el("span", "", config.view==="environment" && graph().synthetic
      ? "演示数据（未实测）"
      : ({unchecked:"Unchecked",denied:"Denied",not_implemented:"Not Implemented",
          partial:"Partial",implemented:"Implemented"})[n.status] || "Unknown"));
    area.appendChild(status);
    area.appendChild(el("p", "detail-description",
      n.reader_explanation || plain(n.description || "")));
    if(n.context_term_ids?.length){
      const context=section(area,"专业词汇 · Context");
      for(const id of n.context_term_ids){
        const term=(reviewGraphs.context?.nodes || DATA.context?.nodes || []).find(x=>x.id===id);
        if(!term)continue;
        const row=el("div","node-term-row");
        row.appendChild(refLink(id,term.title));
        row.appendChild(el("p","node-term-definition",plain(term.description)));
        context.appendChild(row);
      }
    }
    if (n.source_symbol && ["module","class","method","function"].includes(kind(n))) {
      const technical = section(area, "源代码标识");
      technical.appendChild(el("code", "signature", n.source_symbol));
    }
    if (n.signature) section(area, "方法签名").appendChild(el("code", "signature", n.signature));
    if (n.facts) {
      const details = section(area, "设备与执行信息");
      for (const [key,value] of Object.entries(n.facts)) {
        details.appendChild(el("p", "", key + "：" + value));
      }
    }
    if (n.why_it_matters) section(area, "Why this matters").appendChild(el("p", "", n.why_it_matters));
    if (n.aliases?.length) section(area, "Also known as").appendChild(el("p", "", n.aliases.join(" · ")));
    if (n.next) section(area, "Next step").appendChild(el("p", "", n.next));
    for (const [title, keys] of [
      ["Scientific knowledge", n.context_ids], ["Project context", n.core_ids],
      ["Scientific terms", n.term_ids]
    ]) {
      if (keys?.length) {
        const block = section(area, title);
        for (const key of keys) {
          const targetView = locateView(key);
          const source = (reviewGraphs[targetView]?.nodes || DATA[targetView]?.nodes || []).find(x=>x.id===key);
          block.appendChild(refLink(key, source?.title || key));
        }
      }
    }
    if (n.foundation) {
      const block = section(area, "Background");
      const a = el("a", "context-link", n.foundation);
      a.href = "https://github.com/OpenSAE/SAEBase/blob/main/" + n.foundation;
      a.target = "_blank"; a.rel = "noopener noreferrer"; block.appendChild(a);
    }
    if (n.evidence?.length) {
      const block = section(area, "Evidence");
      for (const source of n.evidence.slice(0, 12)) {
        const a = el("a", "context-link", source);
        const path = source.split(":")[0];
        if (path.startsWith("https://")) a.href = path;
        else if (!path.startsWith("/") && !path.includes("..")) {
          a.href = "https://github.com/OpenSAE/SAEBase/blob/main/" + path;
        }
        a.target = "_blank"; a.rel = "noopener noreferrer"; block.appendChild(a);
      }
    }
    if (config.view === "code" && config.diff) {
      const changed = config.diff.changed?.find(x => x.id === n.id);
      if (changed) section(area, "Observed change").appendChild(el("p", "", changed.fields.join(", ")));
    }
  }
  function render(autoFit) {
    const layout = normalizeLayout(config.mode === "focus" ? layoutFocus() : layoutOverview());
    config.visible = layout.visible;
    drawSVG(layout);
    // Preserve the user’s visual scale across focus, expansion, and search.
    // The tree grows in scrollable space rather than squeezing every card.
    if (!config.zoomInitialized) {
      config.zoom = 1;
      config.zoomInitialized = true;
    }
    setScale();
    updateBreadcrumb(); renderOutline(); renderDetails(); setDrawerState();
    $("canvasWrap").classList.toggle("in-focus", config.mode === "focus");
    $("diffControl").classList.toggle("is-hidden", config.view !== "code");
    $("focusHint").textContent = config.view === "environment" && graph().synthetic
      ? "演示环境 · 合成设备配置" : "";
    $("statusLegend").classList.toggle("implemented-only", config.view !== "environment");
    $("statusLegend").classList.toggle("is-hidden", !!review);
    $("collapse").disabled = config.mode === "focus";
    $("expand").disabled = config.mode === "focus";
    if (review) updateReviewBar();
    if (config.mode === "focus") centerSelected();
    else if (autoFit && typeof $("canvasWrap").scrollTo === "function") {
      $("canvasWrap").scrollTo({ left: 0, top: 0 });
    }
  }

  function bindCanvasPan() {
    const canvas=$("canvasWrap");
    const boundary=()=>({
      x:Math.max(0,(Number(canvas.scrollWidth)||config.scene.width*config.zoom)-
        (Number(canvas.clientWidth)||1150)),
      y:Math.max(0,(Number(canvas.scrollHeight)||config.scene.height*config.zoom)-
        (Number(canvas.clientHeight)||630))
    });
    const easeBack=()=>{
      if(!drag)return;
      const b=boundary(),left=Math.max(0,Math.min(b.x,canvas.scrollLeft)),
        top=Math.max(0,Math.min(b.y,canvas.scrollTop));
      canvas.scrollTo?.({left,top,behavior:"smooth"});
      drag=null;
      canvas.classList.remove("is-dragging");
      canvas.classList.add("is-bouncing");
      $("diagram").style.transform="translate(0px,0px)";
    };
    // Thumb buttons on five-button mice mirror forced previous/next.
    // This does not replace Shift+Left/Right on the keyboard.
    canvas.addEventListener("mousedown",event=>{
      if(event.button!==3 && event.button!==4)return;
      if(moveReviewOrSibling(event.button===3?-1:1))
        event.preventDefault?.();
    });
    canvas.addEventListener("auxclick",event=>{
      if(event.button===3 || event.button===4)event.preventDefault?.();
    });
    canvas.addEventListener("pointerdown",event=>{
      if(event.button!==0 || event.target?.closest?.(
        ".graph-node,button,a,input,textarea,select,[role=button]"))return;
      if(bounceFrame && window.cancelAnimationFrame)window.cancelAnimationFrame(bounceFrame);
      canvas.classList.remove("is-bouncing");
      $("diagram").style.transform="translate(0px,0px)";
      drag={id:event.pointerId,x:event.clientX,y:event.clientY,
        left:canvas.scrollLeft,top:canvas.scrollTop};
      canvas.setPointerCapture?.(event.pointerId);
      canvas.classList.add("is-dragging");
    });
    canvas.addEventListener("pointermove",event=>{
      if(!drag || event.pointerId!==drag.id)return;
      const bounds=boundary();
      const rubber=(value,max)=>value<0?value*.18:value>max?
        max+(value-max)*.18:value;
      const wantedX=drag.left+drag.x-event.clientX,
        wantedY=drag.top+drag.y-event.clientY;
      canvas.scrollLeft=Math.max(0,Math.min(bounds.x,wantedX));
      canvas.scrollTop=Math.max(0,Math.min(bounds.y,wantedY));
      // Native scroll cannot overshoot its bounds: translate the SVG for
      // visible resistance and spring it back on pointer release.
      const offsetX=rubber(wantedX,bounds.x)-canvas.scrollLeft;
      const offsetY=rubber(wantedY,bounds.y)-canvas.scrollTop;
      $("diagram").style.transform="translate("+(-offsetX)+"px,"+(-offsetY)+"px)";
      event.preventDefault?.();
    });
    $("diagram").addEventListener("transitionend",()=>{
      canvas.classList.remove("is-bouncing");
    });
    for(const key of ["pointerup","pointercancel","lostpointercapture"])
      canvas.addEventListener(key,event=>{
        if(!drag || event.pointerId!==drag.id)return;
        canvas.releasePointerCapture?.(drag.id);
        easeBack();
      });
  }
  function bind() {
    bindCanvasPan();
    // Each public HTML is a genuine URL; same-view focus changes use browser history.
    $("overviewBack").addEventListener("click", () => overview(true));
    $("outlineToggle").addEventListener("click", () => {
      config.outlineOpen = !config.outlineOpen;
      config.detailsOpen = false; setDrawerState();
    });
    $("outlineClose").addEventListener("click", () => {
      config.outlineOpen = false; setDrawerState();
    });
    $("detailsToggle").addEventListener("click", () => {
      if (!config.focusId) return;
      config.detailsOpen = !config.detailsOpen;
      config.outlineOpen = false; renderDetails(); setDrawerState();
    });
    $("detailsClose").addEventListener("click", () => {
      config.detailsOpen = false; setDrawerState();
    });
    $("search").addEventListener("input", event => {
      config.query = String(event.target.value || "");
      config.expandedAttributes.clear();
      config.mode = "overview"; config.focusId = null; config.detailsOpen = false;
      config.outlineOpen = !!config.query.trim();
      if (window.location?.search?.includes("focus=")) setAddress(null, true);
      render(true);
    });
    $("collapse").addEventListener("click", () => {
      if (config.mode === "focus") return;
      config.expanded = new Set(rootIds);
      render(true);
    });
    $("expand").addEventListener("click", () => {
      if (config.mode === "focus") return;
      config.expanded = new Set(byId.keys());
      render(true);
    });
    $("fit").addEventListener("click", () => {
      config.zoom = fitZoom(); config.zoomInitialized = true; render(false);
    });
    $("zoomIn").addEventListener("click", () => {
      config.zoom = Math.min(1.8, Math.round((config.zoom + 0.15) * 100) / 100);
      render(false);
    });
    $("zoomOut").addEventListener("click", () => {
      config.zoom = Math.max(0.45, Math.round((config.zoom - 0.15) * 100) / 100);
      render(false);
    });
    $("diffFile").addEventListener("change", async event => {
      const file = event.target.files?.[0];
      if (!file) return;
      try {
        const result = JSON.parse(await file.text());
        if (result.kind !== "code_diff") throw new Error("Expected a code_diff JSON report");
        config.diff = result;
        if (config.view !== "code") {
          if (window.location) window.location.href = "code.html";
        } else render(true);
      } catch (error) {
        if (typeof alert === "function") alert("Cannot load code-tree diff: " + error.message);
      }
      event.target.value = "";
    });
    if (window.addEventListener) {
      window.addEventListener("popstate", () => {
        if(review) {
          const current = (window.location?.pathname || "").match(/\/(core|knowledge|code|environment|context)\.html$/);
          if(current && current[1] !== config.view && current[1] !== "core")
            activateReviewView(current[1]);
        }
        const next = getSearchFocus();
        config.expandedAttributes.clear();
        config.focusId = next && byId.has(next) ? next : null;
        config.mode = config.focusId ? "focus" : "overview";
        config.detailsOpen = false;
        config.query = ""; $("search").value = "";
        render(true);
      });
      window.addEventListener("keydown", event => {
        if(handleReviewShortcut(event))return;
        const active=event.target||document.activeElement;
        const tag=String(active?.tagName||"").toUpperCase();
        const editing=["INPUT","TEXTAREA","SELECT"].includes(tag) ||
          active?.isContentEditable;
        if(!editing && !event.altKey && !event.ctrlKey && !event.metaKey &&
           event.shiftKey && ["ArrowLeft","ArrowRight"].includes(event.key)){
          const direction=event.key==="ArrowLeft"?-1:1;
          if(moveReviewOrSibling(direction))event.preventDefault?.();
          return;
        }
        if (event.key === "Escape") {
          if(config.expandedAttributes.size) {
            config.expandedAttributes.clear();render(true);return;
          }
          if (config.detailsOpen || config.outlineOpen) {
            config.detailsOpen = false; config.outlineOpen = false; setDrawerState();
          } else if (config.mode === "focus") overview(true);
        }
      });
      window.addEventListener("resize", () => {
        const v = $("canvasWrap"), next = {
          width: Number(v.clientWidth) || 0, height: Number(v.clientHeight) || 0
        };
        if (Math.abs(next.width - lastViewport.width) > 90 ||
            Math.abs(next.height - lastViewport.height) > 90) {
          lastViewport = next;
          render(false);
        }
      });
    }
  }

  init();
})();

