/* OpenResearch Atlas: centered, source-backed trees with node focus navigation. */
(function () {
  "use strict";


  const DATA = window.OPENRESEARCH_ATLAS || {};
  if (DATA.context && !String(DATA.context.project || "").startsWith("SAEBase"))
    throw new Error("Foreign Atlas project loaded into SAEBase");
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
  // Discard navigation-only nodes in legacy static data and structure patches.
  // Reattach real children to their nearest real ancestor; never create a
  // synthetic node merely to cap the number of visible cards.
  for (const tree of Object.values(DATA)) {
    window.SAEBASE_ATLAS_REMOVE_FAKE_NAVIGATION?.(tree);
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
    // Top-level tabs use the unified switcher, including review mode.
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
    panX: 0,
    panY: 0,
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
  // Every real child remains available without paging.
  const childPages = new Map();
  // Independent focus, expansion, zoom and scroll for each Atlas tree.
  const viewStates = new Map();
  let parents = new Map(), children = new Map(), byId = new Map();
  let rootIds = [];
  let drag = null;
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
  function pagedChildren(id) {
    return (children.get(id) || []).filter(key => byId.has(key));
  }
  function updateChildPager() {
    $("atlasChildPager")?.classList.add("is-hidden");
  }
  function ancestors(id) {
    const result = []; const seen = new Set();
    let cursor = id;
    while (byId.has(cursor) && !seen.has(cursor)) {
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
    const base = window.location.pathname.replace(
      /(?:index|core|knowledge|code|environment|context)\.html$/, ""
    );
    const path = (base.endsWith("/") ? base : base + "/") + config.view + ".html";
    const params = new URLSearchParams(window.location.search || "");
    if (focusId) params.set("focus", focusId);
    else params.delete("focus");
    const url = path + (params.toString() ? "?" + params : "") +
      (window.location.hash || "");
    try {
      window.history[replace ? "replaceState" : "pushState"](
        { view: config.view, focusId: focusId || null }, "", url
      );
    } catch (_) { /* file:// mode may reject history */ }
  }
  function locateView(id) {
    if (byId.has(id)) return config.view;
    if (DATA.code?.source_api_index?.[id]) return "code";
    for (const key of Object.keys(VIEWS)) {
      if ((reviewGraphs[key]?.nodes || DATA[key]?.nodes || []).some(n => n.id === id)) return key;
    }
    return null;
  }
  function saveViewState() {
    viewStates.set(config.view, {
      mode: config.mode, focusId: config.focusId,
      expanded: new Set(config.expanded),
      expandedAttributes: new Set(config.expandedAttributes),
      query: config.query, zoom: config.zoom,
      zoomInitialized: config.zoomInitialized,
      outlineOpen: config.outlineOpen, detailsOpen: config.detailsOpen,
      panX: config.panX, panY: config.panY
    });
  }
  function activateView(view, options = {}) {
    if (!VIEWS[view] || !DATA[view]) return false;
    if (view === config.view) return true;
    saveViewState();
    config.view = view;
    if (document.body?.dataset) document.body.dataset.atlasView = view;
    rebuildHierarchy();
    const prior = viewStates.get(view);
    config.mode = prior?.mode || "overview";
    config.focusId = prior?.focusId || null;
    if (config.focusId && !byId.has(config.focusId)) {
      config.focusId = null;
      config.mode = "overview";
    }
    config.expanded = prior ? new Set(prior.expanded) : initialExpanded();
    config.expandedAttributes = prior ? new Set(prior.expandedAttributes) : new Set();
    config.query = prior?.query || "";
    config.zoom = prior?.zoom ?? 1;
    config.zoomInitialized = prior?.zoomInitialized ?? false;
    config.panX = prior?.panX ?? 0;
    config.panY = prior?.panY ?? 0;
    config.outlineOpen = prior?.outlineOpen || false;
    config.detailsOpen = !!prior?.detailsOpen && !!config.focusId;
    config.animateNextRender = false;
    $("search").value = config.query;
    for (const tab of document.querySelectorAll(".tabs [data-graph]")) {
      const active = tab.dataset.graph === view;
      tab.classList.toggle("active", active);
      if (active) tab.setAttribute("aria-current", "page");
      else tab.removeAttribute("aria-current");
    }
    if (review && view !== "core") reviewFilter = {level:"view", view};
    $("atlasReviewBar").classList.toggle("is-hidden", !review || view === "core");
    render(false, true);
    if (!prior && config.focusId) centerSelected();
    if (options.history !== false) setAddress(config.focusId, false);
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
    const foldedOwner = target === "code" && DATA.code?.source_api_index?.[id];
    if (foldedOwner) {
      id = foldedOwner;
      options = {...options, toggle:true};
    }
    if (!target || !(reviewGraphs[target]?.nodes || DATA[target]?.nodes || []).some(n => n.id === id)) return false;
    const sameFocus = target===config.view && config.focusId===id;
    const wasExpanded = sameFocus && config.expandedAttributes.has(id);
    if (target !== config.view && !activateView(target, {history:false}))
      return false;
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
    render(false,true);
    centerSelected();
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
    fitCanvas();
  }
  function openCross(id) { navigate(id, locateView(id), false); }
  function init() {
    if (!VIEWS[config.view] || !DATA[config.view]) config.view = "core";
    rebuildHierarchy();
    config.expanded = initialExpanded();
    const deep = getSearchFocus();
    const sourceOwner = deep && DATA[config.view]?.source_api_index?.[deep];
    if (deep && (byId.has(deep) || (sourceOwner && byId.has(sourceOwner)))) {
      config.mode = "focus"; config.focusId = byId.has(deep) ? deep : sourceOwner;
      if (sourceOwner) config.expandedAttributes.add(sourceOwner);
    }
    bind();
    render(true);
    if (config.focusId) centerSelected();
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
  function densityMode() {
    return config.zoom <= .80 ? "compact" : config.zoom >= 1.16 ? "detail" : "standard";
  }
  function measure(node, focused) {
    const density=densityMode();
    const expanded=config.expandedAttributes.has(node.id);
    const w=expanded?525:density==="compact"?278:density==="detail"?336:310;
    const title=wrap(node.title||node.id,w-45,density==="compact"?16:14.2,3);
    const body=density==="compact"&&!expanded?[]:
      wrap(node.summary||node.description,w-42,12,density==="detail"?5:expanded?4:2);
    const context=density==="detail"&&!expanded?
      wrap((node.context_term_ids||[]).slice(0,3).map(id=>
        DATA.context?.nodes?.find(term=>term.id===id)?.title
      ).filter(Boolean).join(" · "),w-42,11,1):[];
    const base=Math.max(density==="compact"?86:98,22+title.length*19+
      (body.length?10+body.length*17:0)+(context.length?8+context.length*17:0)+18);
    const isClass=(node.attributes||[]).some(x=>x.type==="class_methods");
    const count=(node.attributes||[]).length;
    const terms=(node.context_term_ids||[]).length;
    const extraH=expanded?Math.min(760,Math.max(340,170+terms*65+
      Math.min(count,5)*46+(isClass?180:0))):0;
    return {w,h:base+extraH,title,body,context,baseH:base,extraH,expanded};
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
      if (!byId.has(id) || seen.has(id)) return;
      if (config.query && !needed.has(id)) return;
      seen.add(id); list.push({ id, depth, role: "overview" });
      if (config.query || config.expanded.has(id)) {
        for (const child of pagedChildren(id)) visit(child, depth + 1);
      }
    }
    for (const id of rootIds) visit(id, 0);
    return list;
  }

  function layoutOverview(visible = visibleOverview()) {
    const allowed = new Set(visible.map(v => v.id));
    const sizes = new Map(), blocks = new Map(), positions = new Map();
    const VERTICAL_GAP = 18, HORIZONTAL_GAP = 48, ORIGIN_X = 36;
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

  // Focus shows every preceding level. At the focused depth, only
  // siblings sharing its parent appear; the focused node's direct
  // children remain available for deeper exploration.
  function visibleFocus() {
    const path=ancestors(config.focusId);
    if(!path.length)return visibleOverview();
    const depthLimit=path.length-1;
    const siblings=new Set(siblingsOf(config.focusId));
    const roots=[...new Set([...rootIds,path[0]])];
    const list=[],seen=new Set(),onPath=new Set(path);
    function visit(id,depth){
      if(!byId.has(id)||seen.has(id)||depth>depthLimit+1)return;
      if(depth===depthLimit&&!siblings.has(id))return;
      if(depth===depthLimit+1&&parents.get(id)!==config.focusId)return;
      seen.add(id);
      list.push({id,depth,role:id===config.focusId?"focus":
        onPath.has(id)?"ancestor":depth===depthLimit?"sibling":
        depth===depthLimit+1?"child":"context"});
      if(depth<depthLimit){
        for(const child of children.get(id)||[])visit(child,depth+1);
      } else if(id===config.focusId){
        for(const child of children.get(id)||[])visit(child,depth+1);
      }
    }
    for(const root of roots)visit(root,0);
    return list;
  }
  function layoutFocus(){
    const layout=layoutOverview(visibleFocus());
    for(const item of layout.visible){
      const p=layout.positions.get(item.id);
      if(p)p.role=item.role;
    }
    return layout;
  }
  function normalizeLayout(layout) {
    // Camera coordinates are unbounded. No selected-node padding or
    // invisible scroll areas are needed to reposition the tree.
    let maxX=0,maxY=0;
    for(const p of layout.positions.values()){
      maxX=Math.max(maxX,p.x+p.w);
      maxY=Math.max(maxY,p.y+p.h);
    }
    return {...layout,width:Math.max(600,maxX+35),
      height:Math.max(300,maxY+35)};
  }
  function textLines(group, lines, x, startY, lineHeight, className) {
    for (let i = 0; i < lines.length; i++) {
      group.appendChild(svg("text", { x, y: startY + i * lineHeight,
        class: className }, lines[i]));
    }
  }
  function drawCard(node, p, layer, positions, focus) {
    const k = kind(node), chosen = node.id === config.focusId;
    const classes = ["graph-node", "node-" + k, "density-" + densityMode(), chosen ? "node-selected" : "",
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
    if(p.context?.length) {
      const cy=top+p.title.length*19+7+(p.body.length?p.body.length*17+9:0);
      textLines(card,p.context,titleX,cy,17,"node-context-preview");
    }
    const onOpen = () => navigate(node.id,config.view,false,{toggle:true});
    if(p.expanded){
      const embedded=svg("foreignObject",{
        x:10,y:p.baseH+8,width:p.w-20,height:p.extraH-41,
        class:"node-attributes-foreign"
      });
      const content=el("div","node-attributes-inner");
      content.setAttribute("aria-label",node.title+" 的完整解释与证据");
      const extraExplanation=plain(node.reader_explanation || "");
      if(extraExplanation &&
          extraExplanation.replace(/\s+/g,"") !==
          plain(node.summary || "").replace(/\s+/g,"")){
        const intro=attributeSection(content,"这个节点在做什么");
        intro.appendChild(el("p","node-attribute-text",extraExplanation));
      }
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
         densityMode() === "compact" ? "查看详情" :
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
    const focusPath=new Set(config.mode==="focus"&&config.focusId?ancestors(config.focusId):[]);
    function link(source, target) {
      const a = position.get(source), b = position.get(target);
      if (!a || !b) return;
      const x1 = a.x + a.w, x2 = b.x;
      const y1 = a.y + a.h / 2, y2 = b.y + b.h / 2;
      if (x2 <= x1) return;
      const mid = (x1 + x2) / 2;
      lines.appendChild(svg("path", {
        class: "edge-main"+(focusPath.has(target)?" is-current-path":""),
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
  // Fit the actual node positions, not the invisible padding that keeps
  // a focused card centered during normal browsing.
  function fitBounds() {
    const positions = [...config.positions.values()];
    if (!positions.length) return {minX:0,minY:0,maxX:600,maxY:300};
    return {
      minX:Math.min(...positions.map(p => p.x)),
      minY:Math.min(...positions.map(p => p.y)),
      maxX:Math.max(...positions.map(p => p.x + p.w)),
      maxY:Math.max(...positions.map(p => p.y + p.h))
    };
  }
  function fitZoom() {
    const v=$("canvasWrap"), bounds=fitBounds();
    const w=Math.max(100,bounds.maxX-bounds.minX);
    const h=Math.max(100,bounds.maxY-bounds.minY);
    return Math.max(.04,Math.min(1,(v.clientWidth-56)/w,(v.clientHeight-62)/h));
  }
  function centerSelected() {
    const v=$("canvasWrap"),p=config.positions.get(config.focusId);
    if(!p)return;
    config.panX=v.clientWidth/2-config.zoom*(p.x+p.w/2);
    config.panY=v.clientHeight/2-config.zoom*(p.y+p.h/2);
    setScale();
  }
  function setScale() {
    const drawing=$("diagram");
    drawing.setAttribute("width",Math.round(config.scene.width));
    drawing.setAttribute("height",Math.round(config.scene.height));
    drawing.style.transform="translate("+config.panX+"px,"+
      config.panY+"px) scale("+config.zoom+")";
    $("zoomValue").textContent=Math.round(config.zoom*100)+"%";
  }
  function changeZoom(next,point=null) {
    const v=$("canvasWrap"),rect=v.getBoundingClientRect();
    const oldZoom=config.zoom||1;
    // Toolbar zoom anchors the selected card or root, keeping it visible.
    // Wheel zoom instead uses the pointer on the actual canvas.
    const home=!point?config.positions.get(config.mode==="focus"?
      config.focusId:config.visible[0]?.id):null;
    const px=point?point.clientX-rect.left:
      home?config.panX+oldZoom*(home.x+home.w/2):v.clientWidth/2;
    const py=point?point.clientY-rect.top:
      home?config.panY+oldZoom*(home.y+home.h/2):v.clientHeight/2;
    const oldX=(px-config.panX)/oldZoom;
    const oldY=(py-config.panY)/oldZoom;
    const targetId=point?.nodeId||null;
    const before=config.positions.get(targetId);
    const localX=before?(oldX-before.x)/before.w:0;
    const localY=before?(oldY-before.y)/before.h:0;
    const zoom=Math.min(2.2,Math.max(.04,next));
    if(Math.abs(zoom-oldZoom)<.0001)return;
    config.zoom=zoom;
    config.zoomInitialized=true;
    render(false,true);
    const after=config.positions.get(targetId);
    const worldX=before&&after?after.x+localX*after.w:oldX;
    const worldY=before&&after?after.y+localY*after.h:oldY;
    config.panX=px-worldX*zoom;
    config.panY=py-worldY*zoom;
    setScale();
  }
  function fitCanvas() {
    changeZoom(fitZoom());
    changeZoom(fitZoom());
    const v=$("canvasWrap"),b=fitBounds();
    config.panX=v.clientWidth/2-(b.minX+b.maxX)*config.zoom/2;
    config.panY=v.clientHeight/2-(b.minY+b.maxY)*config.zoom/2;
    setScale();
  }
  function renderPathTrace() {
    const trace = $("atlasPathTrace");
    if (!trace) return;
    trace.replaceChildren();
    const focused = config.mode === "focus" && byId.has(config.focusId);
    trace.classList.toggle("is-hidden", !focused);
    if (!focused) return;
    trace.classList.toggle("is-compact", densityMode() === "compact");
    for (const [index, id] of ancestors(config.focusId).entries()) {
      const n = byId.get(id);
      if (index) trace.appendChild(el("span", "atlas-path-separator", "›"));
      const step = el("button", "atlas-path-step" +
        (id === config.focusId ? " is-current" : ""), n.title);
      step.type = "button";
      step.title = n.title;
      step.setAttribute("aria-current", id === config.focusId ? "step" : "false");
      step.addEventListener("click", () => {
        if (id !== config.focusId) navigate(id, config.view, false);
      });
      trace.appendChild(step);
    }
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
      for(const item of config.visible){
        const node=byId.get(item.id);
        if(node)output.appendChild(rowButton(item.id,node.title,Math.min(5,item.depth)));
      }
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
    if(attr.type==="source_interfaces") {
      const section=attributeSection(parent,attr.label);
      // Source symbols are technical detail, not fictional research cards.
      for(const item of attr.members||[]) {
        const details=el("details","node-method-details");
        const heading=el("summary","node-method-summary",
          (item.kind||"source")+" · "+(item.source_symbol||item.id));
        details.appendChild(heading);
        const body=el("div","node-method-body");
        const originalPath=item.path||node.path||"";
        if(item.signature)body.appendChild(el("code","node-attribute-signature",item.signature));
        for(const block of item.attributes||[]) {
          renderAttributeBlock(body,block,{path:originalPath,id:item.id});
        }
        if(originalPath)sourceAnchor(body,originalPath,item.line||1,"查看真实源码");
        details.appendChild(body);
        section.appendChild(details);
      }
      return;
    }
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


  // All five source-backed Atlas trees share the search index. Rebuild from
  // current graphs each time so locale and review overlays remain authoritative.
  const ATLAS_VIEW_LABELS={core:"Core",knowledge:"Knowledge",code:"Code",
    environment:"Environment",context:"Context"};
  let searchDialog=null,searchInput=null,searchResults=null,searchFilters=null;
  let searchOpen=false,searchView="all",searchHits=[],searchSelection=0;
  let searchReturnFocus=null;
  const searchText=(zh,en)=>document.documentElement.lang.toLowerCase().startsWith("zh")?zh:en;
  function globalSearchEntries(query) {
    const q=String(query||"").trim().toLocaleLowerCase();
    const result=[];
    for(const view of Object.keys(ATLAS_VIEW_LABELS)){
      const nodes=(reviewGraphs[view]||DATA[view])?.nodes||[];
      for(const node of nodes){
        if(!node?.id || !node?.title)continue;
        const title=String(node.title).toLocaleLowerCase();
        const identity=[node.id,node.source_symbol,node.signature,node.path,
          ...(node.aliases||[])].filter(Boolean).join(" ").toLocaleLowerCase();
        const summary=String(node.summary||node.description||"").toLocaleLowerCase();
        const other=String(node.reader_explanation||"").toLocaleLowerCase();
        let score=0;
        if(!q)score=1;
        else if(title===q||String(node.id).toLocaleLowerCase()===q)score=150;
        else if(title.startsWith(q))score=105;
        else if(title.includes(q))score=80;
        else if(identity.includes(q))score=65;
        else if(summary.includes(q))score=40;
        else if(other.includes(q))score=20;
        if(!score)continue;
        result.push({view,id:node.id,title:node.title,
          summary:plain(node.summary||node.description||""),
          identity:node.source_symbol||node.path||"",score});
      }
    }
    // Exact identity must win over a merely related description.
    result.sort((a,b)=>b.score-a.score||
      a.title.localeCompare(b.title,document.documentElement.lang.startsWith("zh")?"zh":"en"));
    return result;
  }
  function updateGlobalSearch() {
    if(!searchDialog||!searchOpen)return;
    const q=searchInput.value;
    const matches=globalSearchEntries(q);
    searchFilters.replaceChildren();
    for(const [view,label] of [["all",searchText("全部","All")],
      ...Object.entries(ATLAS_VIEW_LABELS)]){
      const count=matches.filter(x=>view==="all"||x.view===view).length;
      const button=el("button","atlas-search-filter"+(searchView===view?" is-active":""),label);
      button.type="button";
      button.setAttribute("aria-pressed",String(searchView===view));
      button.appendChild(el("span","atlas-search-filter-count",String(count)));
      button.addEventListener("click",()=>{
        searchView=view;searchSelection=0;updateGlobalSearch();searchInput.focus();
      });
      searchFilters.appendChild(button);
    }
    const filtered=matches.filter(x=>searchView==="all"||x.view===searchView);
    searchHits=searchView==="all"?
      Object.keys(ATLAS_VIEW_LABELS).flatMap(view=>
        filtered.filter(hit=>hit.view===view).slice(0,q.trim()?14:4)):
      filtered.slice(0,q.trim()?65:24);
    searchSelection=Math.max(0,Math.min(searchSelection,searchHits.length-1));
    searchResults.replaceChildren();
    const grouped=new Map();
    for(const hit of searchHits){
      if(!grouped.has(hit.view))grouped.set(hit.view,[]);
      grouped.get(hit.view).push(hit);
    }
    for(const [view,hits] of grouped){
      const group=el("div","atlas-search-result-group");
      group.appendChild(el("p","atlas-search-group-name",ATLAS_VIEW_LABELS[view]));
      for(const hit of hits){
        const index=searchHits.indexOf(hit);
        const button=el("button","atlas-search-result"+(index===searchSelection?" is-selected":""));
        button.dataset.index=String(index);
        button.type="button";button.setAttribute("role","option");
        button.setAttribute("aria-selected",String(index===searchSelection));
        button.addEventListener("mouseenter",()=>{
          searchSelection=index;highlightGlobalSearch();
        });
        button.addEventListener("click",()=>selectGlobalSearch(index));
        const top=el("span","atlas-search-result-heading");
        top.appendChild(el("strong","",hit.title));
        const caption=el("span","atlas-search-result-summary",hit.summary||hit.identity||hit.id);
        button.append(top,caption);group.appendChild(button);
      }
      searchResults.appendChild(group);
    }
    if(!searchHits.length){
      searchResults.appendChild(el("p","atlas-search-empty",
        searchText("没有匹配的节点","No matching nodes")));
    }
    const counter=$("atlasGlobalSearchCount");
    if(counter)counter.textContent=searchText("显示 ","Showing ")+
      searchHits.length+" / "+filtered.length;
  }
  function highlightGlobalSearch() {
    for(const [index,node] of [...searchResults.querySelectorAll(".atlas-search-result")].entries()){
      const key=Number(node.dataset.index);
      node.classList.toggle("is-selected",key===searchSelection);
      node.setAttribute("aria-selected",String(key===searchSelection));
      if(key===searchSelection)node.scrollIntoView({block:"nearest"});
    }
  }
  function selectGlobalSearch(index) {
    const hit=searchHits[index];
    if(!hit)return;
    closeGlobalSearch(false);
    // Explicit target view handles node IDs which appear in several trees.
    navigate(hit.id,hit.view,false);
  }
  function closeGlobalSearch(restore=true) {
    if(!searchDialog||!searchOpen)return;
    searchOpen=false;
    searchDialog.classList.add("is-hidden");
    searchDialog.setAttribute("aria-hidden","true");
    if(restore)searchReturnFocus?.focus?.();
  }
  function openGlobalSearch() {
    if(!searchDialog)return;
    if(searchOpen){searchInput.focus();return;}
    searchReturnFocus=document.activeElement;
    searchOpen=true;searchView="all";searchSelection=0;
    searchInput.value="";
    searchInput.placeholder=searchText("搜索所有节点、术语、函数…","Search nodes, terms, functions…");
    searchInput.dataset.atlasOriginalPlaceholder=searchInput.placeholder;
    const help=searchDialog.querySelectorAll(".atlas-search-footer > span");
    if(help[0])help[0].textContent="↑ ↓ "+searchText("选择","Select")+"   ↵ "+searchText("跳转","Open");
    if(help[1])help[1].textContent="Esc "+searchText("关闭","Close");
    searchDialog.classList.remove("is-hidden");
    searchDialog.setAttribute("aria-hidden","false");
    updateGlobalSearch();searchInput.focus();
  }
  function installGlobalSearch() {
    const launcher=el("button","atlas-global-launch", searchText("五树搜索","All trees"));
    launcher.type="button";launcher.id="atlasGlobalSearchButton";
    launcher.setAttribute("aria-label","Search all Atlas trees");
    launcher.title=searchText("搜索五棵树 · Ctrl K","Search five trees · Ctrl K");
    const toolbar=$("search")?.closest(".search-box");
    toolbar?.insertAdjacentElement("afterend",launcher);
    launcher.addEventListener("click",openGlobalSearch);
    searchDialog=el("div","atlas-search-overlay is-hidden");
    searchDialog.id="atlasGlobalSearch";
    searchDialog.setAttribute("aria-hidden","true");
    searchDialog.addEventListener("mousedown",event=>{
      if(event.target===searchDialog)closeGlobalSearch();
    });
    const panel=el("div","atlas-search-dialog");
    panel.setAttribute("role","dialog");panel.setAttribute("aria-modal","true");
    panel.setAttribute("aria-label","Search across Atlas trees");
    const heading=el("div","atlas-search-dialog-top");
    heading.appendChild(el("span","atlas-search-dialog-icon","⌕"));
    searchInput=el("input","atlas-search-input");
    searchInput.id="atlasGlobalSearchInput";
    searchInput.type="search";
    searchInput.autocomplete="off";
    searchInput.placeholder=searchText("搜索所有节点、术语、函数…","Search nodes, terms, functions…");
    searchInput.dataset.atlasOriginalPlaceholder=searchInput.placeholder;
    heading.appendChild(searchInput);
    const dismiss=el("button","atlas-search-dismiss","Esc");
    dismiss.type="button";dismiss.setAttribute("aria-label","Close search");
    dismiss.addEventListener("click",()=>closeGlobalSearch());
    heading.appendChild(dismiss);
    searchFilters=el("div","atlas-search-filters");
    searchResults=el("div","atlas-search-results");
    searchResults.id="atlasGlobalSearchResults";
    searchResults.setAttribute("role","listbox");
    searchResults.setAttribute("aria-label","Atlas search results");
    const footer=el("div","atlas-search-footer");
    footer.appendChild(el("span","","↑ ↓ "+searchText("选择","Select")+"   ↵ "+searchText("跳转","Open")));
    footer.appendChild(el("span","", "Esc "+searchText("关闭","Close")));
    footer.appendChild(el("span","atlas-search-total",""));
    footer.lastChild.id="atlasGlobalSearchCount";
    panel.append(heading,searchFilters,searchResults,footer);
    searchDialog.appendChild(panel);document.body.appendChild(searchDialog);
    searchInput.addEventListener("input",()=>{searchSelection=0;updateGlobalSearch();});
    document.addEventListener("keydown",event=>{
      const openShortcut=(event.ctrlKey||event.metaKey)&&
        String(event.key).toLocaleLowerCase()==="k";
      if(openShortcut){
        event.preventDefault();event.stopImmediatePropagation();
        openGlobalSearch();return;
      }
      if(!searchOpen)return;
      if(["ArrowDown","ArrowUp","Enter","Escape"].includes(event.key)){
        event.preventDefault();event.stopImmediatePropagation();
        if(event.key==="Escape")closeGlobalSearch();
        if(event.key==="ArrowDown"&&searchHits.length){
          searchSelection=(searchSelection+1)%searchHits.length;highlightGlobalSearch();
        }
        if(event.key==="ArrowUp"&&searchHits.length){
          searchSelection=(searchSelection+searchHits.length-1)%searchHits.length;
          highlightGlobalSearch();
        }
        if(event.key==="Enter")selectGlobalSearch(searchSelection);
      }
    },true);
  }
  function nodeDeepLink(id) {
    const url=new URL(window.location.href);
    url.pathname=url.pathname.replace(
      /(?:index|core|knowledge|code|environment|context)\.html$/,
      config.view+".html");
    url.search="";
    const lang=new URLSearchParams(window.location.search).get("lang");
    if(lang==="zh"||lang==="en")url.searchParams.set("lang",lang);
    url.searchParams.set("focus",id);
    url.hash="";
    return url.href;
  }
  async function copyDeepLink(id,button) {
    const value=nodeDeepLink(id);
    try {
      if(navigator.clipboard?.writeText)await navigator.clipboard.writeText(value);
      else {
        const input=el("textarea","");input.value=value;input.style.position="fixed";
        input.style.opacity="0";document.body.appendChild(input);input.select();
        if(!document.execCommand("copy"))throw new Error("Clipboard unavailable");
        input.remove();
      }
      button.textContent=searchText("已复制","Copied");
    } catch(_) {
      // An ordinary shareable URL remains available if clipboard permission is blocked.
      button.textContent=searchText("选择链接复制","Select link to copy");
      button.title=value;
      const input=el("input","atlas-copy-fallback");input.value=value;
      button.after(input);input.focus();input.select();
    }
  }
  function relatedNode(view,id) {
    const tree=reviewGraphs[view]||DATA[view];
    return tree?.nodes?.find(node=>node.id===id)||null;
  }
  function detailLayer(area,index,label) {
    const block=el("section","atlas-detail-layer");
    const heading=el("h3","atlas-detail-layer-heading");
    heading.appendChild(el("span","atlas-detail-layer-index",index));
    heading.appendChild(el("span","",label));
    block.appendChild(heading);area.appendChild(block);
    return block;
  }


  function renderDetails() {
    const area=$("detailBody");area.replaceChildren();
    const n=byId.get(config.focusId);
    if(!n){
      area.appendChild(el("p","empty-message",
        searchText("选择节点查看关系与依据","Select a node for relationships and evidence")));
      return;
    }
    const nodeLayer=detailLayer(area,"01",searchText("当前节点","NODE"));
    const titleRow=el("div","atlas-detail-title-row");
    titleRow.appendChild(el("h2","detail-title",n.title));
    const copy=el("button","atlas-copy-link",
      searchText("复制链接","Copy link"));
    copy.type="button";copy.title=nodeDeepLink(n.id);
    copy.addEventListener("click",()=>copyDeepLink(n.id,copy));
    titleRow.appendChild(copy);nodeLayer.appendChild(titleRow);
    const status=el("div","detail-status"),swatch=el("span","detail-status-swatch");
    const proposal=itemFor(n.id);
    swatch.style.background=proposal?REVIEW_PALETTE[proposalState(proposal)].stroke:
      STATES[n.status]||"#adb2b9";
    status.appendChild(swatch);
    const label=config.view==="environment"&&graph().synthetic?
      searchText("演示配置","Synthetic configuration"):
      ({unchecked:"Unchecked",denied:"Denied",not_implemented:"Not Implemented",
         partial:"Partial",implemented:"Implemented"})[n.status]||"Unknown";
    status.appendChild(el("span","",label));nodeLayer.appendChild(status);
    const summary=plain(n.summary||n.description||"");
    if(summary)nodeLayer.appendChild(el("p","detail-description",summary));
    const extra=plain(n.reader_explanation||"");
    if(extra&&extra.replace(/\s+/g,"")!==summary.replace(/\s+/g,""))
      nodeLayer.appendChild(el("p","detail-description",extra));
    if(n.why_it_matters)
      section(nodeLayer,searchText("研究意义","Why this matters")).appendChild(
        el("p","",n.why_it_matters));
    if(n.aliases?.length)
      section(nodeLayer,searchText("别名","Aliases")).appendChild(
        el("p","",n.aliases.join(" · ")));
    if(n.next)section(nodeLayer,searchText("下一步","Next step"))
      .appendChild(el("p","",n.next));

    const relations=detailLayer(area,"02",searchText("跨树关联","RELATIONS"));
    const specs=[
      ["context_term_ids","context",searchText("相关术语","Context term")],
      ["context_ids","knowledge",searchText("科研知识","Knowledge")],
      ["core_ids","core",searchText("项目目标","Core")],
      ["term_ids","context",searchText("术语引用","Term")]
    ];
    let count=0;
    const seen=new Set();
    for(const [field,targetView,label] of specs){
      for(const id of n[field]||[]){
        const identity=targetView+":"+id;
        if(seen.has(identity))continue;
        const target=relatedNode(targetView,id);
        if(!target)continue;
        seen.add(identity);count++;
        const row=el("div","atlas-relation-row");
        const descriptor=el("div","atlas-relation-metadata");
        descriptor.appendChild(el("span","atlas-relation-view",ATLAS_VIEW_LABELS[targetView]));
        descriptor.appendChild(el("span","atlas-relation-field",field));
        row.appendChild(descriptor);
        const jump=el("button","atlas-relation-link",target.title);
        jump.type="button";jump.title=id;
        jump.addEventListener("click",()=>navigate(id,targetView,false));
        row.appendChild(jump);
        relations.appendChild(row);
      }
    }
    if(!count)relations.appendChild(el("p","atlas-detail-muted",
      searchText("此节点暂无明确记录的跨树关联。",
        "No explicit cross-tree relation is recorded for this node.")));

    const evidence=detailLayer(area,"03",searchText("来源与依据","EVIDENCE"));
    if(n.context_term_ids?.length){
      const terms=section(evidence,searchText("术语定义","Terminology"));
      for(const id of n.context_term_ids){
        const term=relatedNode("context",id);
        if(!term)continue;
        const row=el("div","node-term-row");
        const link=el("button","context-link",term.title);
        link.type="button";link.addEventListener("click",()=>navigate(id,"context",false));
        row.appendChild(link);
        row.appendChild(el("p","node-term-definition",plain(term.description)));
        terms.appendChild(row);
      }
    }
    if(n.source_symbol&&["module","class","method","function"].includes(kind(n)))
      section(evidence,searchText("源码标识","Source symbol")).appendChild(
        el("code","signature",n.source_symbol));
    if(n.signature)section(evidence,searchText("方法签名","Signature"))
      .appendChild(el("code","signature",n.signature));
    if(n.facts){
      const details=section(evidence,searchText("设备与执行信息","Execution facts"));
      for(const [key,value] of Object.entries(n.facts))
        details.appendChild(el("p","",key+"："+value));
    }
    if(n.path){
      const direct=sourceHref(n.path,n.line);
      if(direct){
        const block=section(evidence,searchText("直接源码","Direct source"));
        const anchor=el("a","context-link",n.path+(n.line?":"+n.line:""));
        anchor.href=direct;anchor.rel="noopener noreferrer";anchor.target="_blank";
        block.appendChild(anchor);
      }
    }
    if(n.foundation){
      const direct=sourceHref(n.foundation);
      if(direct){
        const block=section(evidence,searchText("背景依据","Background"));
        const a=el("a","context-link",n.foundation);
        a.href=direct;a.target="_blank";a.rel="noopener noreferrer";
        block.appendChild(a);
      }
    }
    if(n.evidence?.length){
      const block=section(evidence,searchText("证据来源","Evidence"));
      for(const source of n.evidence.slice(0,16)){
        const link=el("a","context-link",source);
        const raw=String(source),parts=raw.split(":");
        const href=raw.startsWith("https://")?raw:
          sourceHref(parts[0],/^\d+$/.test(parts[1]||"")?Number(parts[1]):null);
        if(!href)continue;
        link.href=href;link.target="_blank";link.rel="noopener noreferrer";
        block.appendChild(link);
      }
    }
    if(config.view==="code"&&config.diff){
      const changed=config.diff.changed?.find(x=>x.id===n.id);
      if(changed)section(evidence,searchText("变化证据","Observed change"))
        .appendChild(el("p","",changed.fields.join(", ")));
    }
    if(!evidence.querySelector(".context-section"))evidence.appendChild(
      el("p","atlas-detail-muted",
        searchText("此节点的来源信息请查看展开节点属性。",
          "Open node attributes to inspect additional source evidence.")));
  }
  function render(autoFit, preserveViewport = false) {
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
    updateBreadcrumb(); renderPathTrace(); updateChildPager(); renderOutline(); renderDetails(); setDrawerState();
    $("canvasWrap").classList.toggle("in-focus", config.mode === "focus");
    $("diffControl").classList.toggle("is-hidden", config.view !== "code");
    $("focusHint").textContent = config.view === "environment" && graph().synthetic
      ? "演示环境 · 合成设备配置" : "";
    $("statusLegend").classList.toggle("implemented-only", config.view !== "environment");
    $("statusLegend").classList.toggle("is-hidden", !!review);
    $("collapse").disabled = config.mode === "focus";
    $("expand").disabled = config.mode === "focus";
    if (review) updateReviewBar();
    // Renders preserve the freely moved camera. Center once only on an
    // explicit navigation action, never while dragging or changing scale.
  }

  function bindCanvasPan() {
    const canvas=$("canvasWrap");
    // A single wheel gesture now zooms the graph, including when hovering
    // a card. Scrollable source explanations keep their own native wheel.
    let pendingWheel = 0, wheelPoint = null, wheelFrame = 0;
    canvas.addEventListener("wheel", event => {
      if (event.target?.closest?.(".node-attributes-inner,textarea,input,select,pre") &&
          !event.ctrlKey) return;
      event.preventDefault();
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 :
        event.deltaMode === 2 ? canvas.clientHeight : 1);
      pendingWheel += delta;
      const card = event.target?.closest?.(".graph-node");
      wheelPoint = {clientX:event.clientX,clientY:event.clientY,
        nodeId:card?.getAttribute?.("data-node-id") || null};
      if (wheelFrame) return;
      wheelFrame = window.requestAnimationFrame(() => {
        wheelFrame = 0;
        const amount = Math.max(-700, Math.min(700, pendingWheel));
        pendingWheel = 0;
        changeZoom(config.zoom * Math.exp(-amount * .0017), wheelPoint);
      });
    }, {passive:false});

    // Free camera panning in screen coordinates, without scroll boundaries,
    // rubber-band resistance, or a forced Focus center.
    let suppressClick=false;
    canvas.addEventListener("click",event=>{
      if(!suppressClick)return;
      event.preventDefault();
      event.stopImmediatePropagation();
      suppressClick=false;
    },true);
    canvas.addEventListener("pointerdown",event=>{
      if(event.button!==0||event.target?.closest?.(
        ".node-attributes-inner,.attribute-trigger,.child-count,button,a,input,textarea,select,summary,pre,code"))
        return;
      drag={id:event.pointerId,startX:event.clientX,startY:event.clientY,
        panX:config.panX,panY:config.panY,moved:false};
    });
    canvas.addEventListener("pointermove",event=>{
      if(!drag||event.pointerId!==drag.id)return;
      const dx=event.clientX-drag.startX,dy=event.clientY-drag.startY;
      if(!drag.moved&&Math.hypot(dx,dy)<4)return;
      if(!drag.moved){
        drag.moved=true;
        canvas.setPointerCapture?.(drag.id);
        canvas.classList.add("is-dragging");
      }
      config.panX=drag.panX+dx;
      config.panY=drag.panY+dy;
      setScale();
      event.preventDefault();
    });
    for(const key of ["pointerup","pointercancel","lostpointercapture"]){
      canvas.addEventListener(key,event=>{
        if(!drag||event.pointerId!==drag.id)return;
        const moved=drag.moved,pointer=drag.id;
        drag=null;
        if(key==="pointerup"&&canvas.hasPointerCapture?.(pointer))
          canvas.releasePointerCapture(pointer);
        canvas.classList.remove("is-dragging");
        if(moved){
          suppressClick=true;
          window.setTimeout(()=>{suppressClick=false;},0);
        }
      });
    }
  }
  function bind() {
    const pathTrace = el("nav", "atlas-path-trace is-hidden");
    pathTrace.id = "atlasPathTrace";
    pathTrace.setAttribute("aria-label", "Full node ancestry");
    $("canvasWrap").parentNode.insertBefore(pathTrace, $("canvasWrap"));
    bindCanvasPan();
    installGlobalSearch();
    // All five tabs switch the mounted graph; links still work without JS
    // and support browser-native new-tab gestures.
    for (const tab of document.querySelectorAll(".tabs [data-graph]")) {
      tab.addEventListener("click", event => {
        if (event.defaultPrevented || event.button !== 0 ||
            event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
        const target = tab.dataset.graph;
        if (!VIEWS[target] || !DATA[target]) return;
        event.preventDefault();
        if (target !== config.view) activateView(target);
        if (review && target !== "core" && !config.focusId) {
          const first = reviewQueue.find(item => item.view === target);
          if (first) navigate(first.node_id, target, true);
        }
      });
    }
    // Source and Context links use the same in-place navigation as top tabs.
    // Capture before expanded SVG cards stop bubbling; modified clicks keep native behavior.
    document.addEventListener("click", event => {
      if (event.defaultPrevented || event.button !== 0 ||
          event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
      const link = event.target?.closest?.("a[href]");
      if (!link || link.closest(".tabs") || link.target === "_blank") return;
      const url = new URL(link.href, window.location.href);
      const matched = url.pathname.match(
        /\/(core|knowledge|code|environment|context)\.html$/
      );
      if (!matched || url.origin !== window.location.origin) return;
      const target = matched[1], focus = url.searchParams.get("focus");
      if (!VIEWS[target] || !DATA[target]) return;
      if (focus && !((reviewGraphs[target]?.nodes || DATA[target].nodes || [])
          .some(node => node.id === focus) || DATA.code?.source_api_index?.[focus])) return;
      event.preventDefault();
      if (focus) navigate(focus, target, false);
      else if (target !== config.view) activateView(target);
    }, true);
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
    $("fit").addEventListener("click", fitCanvas);
    $("zoomIn").addEventListener("click", () => changeZoom(config.zoom * 1.2));
    $("zoomOut").addEventListener("click", () => changeZoom(config.zoom / 1.2));
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
        const match = (window.location?.pathname || "").match(
          /\/(core|knowledge|code|environment|context)\.html$/
        );
        const target = match?.[1] || "core";
        if (target !== config.view) activateView(target, {history:false});
        const next = getSearchFocus();
        const sourceOwner = next && DATA[config.view]?.source_api_index?.[next];
        const focusId = next && byId.has(next) ? next :
          sourceOwner && byId.has(sourceOwner) ? sourceOwner : null;
        if (config.focusId !== focusId) {
          config.focusId = focusId;
          config.mode = focusId ? "focus" : "overview";
          config.expandedAttributes.clear();
          if (sourceOwner && focusId) config.expandedAttributes.add(focusId);
          config.detailsOpen = false;
          config.query = ""; $("search").value = "";
          render(false);
        }
        if (review) updateReviewBar();
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

  // Refresh translated labels without discarding focus, zoom or review state.
  window.addEventListener("saebase-atlas-locale", () => {
    const launcher=$("atlasGlobalSearchButton");
    if(launcher)launcher.textContent=searchText("五树搜索","All trees");
    if(searchOpen){
      searchInput.placeholder=searchText("搜索所有节点、术语、函数…","Search nodes, terms, functions…");
    searchInput.dataset.atlasOriginalPlaceholder=searchInput.placeholder;
      updateGlobalSearch();
    }
    render(false);
    if (config.detailsOpen) renderDetails();
    updateBreadcrumb();
  });
  init();
})();

