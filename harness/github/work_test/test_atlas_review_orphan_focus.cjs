/* Regression: stale public Atlas Core snapshots must still focus HC07/08/09. */
"use strict";
const assert=require("node:assert/strict"),fs=require("node:fs");
const path=require("node:path"),vm=require("node:vm");
const root=path.join(__dirname,"../../..");
const app=fs.readFileSync(path.join(root,"harness/visualization/app.js"),"utf8");
const authority=JSON.parse(fs.readFileSync(
  path.join(root,"harness/core/project_graph.json"),"utf8"));
const preview=JSON.parse(fs.readFileSync(
  path.join(root,"harness/visualization/review-proposals/HC-Archive-20261101.json"),"utf8"));
assert(preview.core_context?.nodes?.some(n=>n.id==="q.reproducibility"),
  "published HC proposal must include the real Core authority graph");
assert(preview.core_context?.edges?.some(e=>e.source==="q.research"&&
  e.target==="q.reproducibility"),"real Core parent chain is required");
assert.equal(preview.core_context.nodes.length,authority.nodes.length);
function method(start,next){
  const a=app.indexOf("  function "+start+"(");
  const b=app.indexOf("  function "+next+"(",a+3);
  assert(a>=0&&b>a,"Missing "+start+" or "+next);
  return app.slice(a,b);
}
function inspect(useCoreAuthority){
  // Mimic an old public data.js with root only: q.reproducibility and
  // q.agent_flow are absent even though the authoritative Core has them.
  const data={
    core:{nodes:[{id:"core:existing",kind:"root",title:"SAEBase Core"}],edges:[]}
  };
  const review={
    plan:preview.plan,context:{nodes:[],edges:[]},
    core_context:useCoreAuthority?preview.core_context:null
  };
  const ctx={
    review,reviewGraphs:{},DATA:data,REVIEW_VIEWS:["core"],
    VIEWS:{core:{root:"core:existing",
      primary:["contains","refines","addresses","decomposes"]}},
    proposalState:item=>item?.proposal_review?.decision||"unchecked",
    Map,Set
  };
  vm.createContext(ctx);
  vm.runInContext(
    method("overlayGraphs","buildReviewQueue")+"\n" +
    method("visibleFocus","layoutFocus")+"\n" +
    "globalThis.run={overlayGraphs,visibleFocus}",
    ctx,{timeout:1500}
  );
  ctx.run.overlayGraphs();
  const core=ctx.reviewGraphs.core;
  assert(core && core.nodes.length>1,"overlay must restore review nodes");
  const byId=new Map(core.nodes.map(n=>[n.id,n]));
  const parents=new Map(),children=new Map();
  for(const edge of core.edges){
    if(!ctx.VIEWS.core.primary.includes(edge.type))continue;
    if(!byId.has(edge.source)||!byId.has(edge.target)||parents.has(edge.target))continue;
    parents.set(edge.target,edge.source);
    if(!children.has(edge.source))children.set(edge.source,[]);
    children.get(edge.source).push(edge.target);
  }
  ctx.byId=byId;ctx.parents=parents;ctx.children=children;
  ctx.rootIds=["core:existing"];
  ctx.config={mode:"focus",focusId:"q.reproducibility"};
  ctx.ancestors=id=>{
    const out=[],seen=new Set();
    for(let cursor=id;byId.has(cursor)&&!seen.has(cursor);cursor=parents.get(cursor)){
      out.push(cursor);seen.add(cursor);
    }
    return out.reverse();
  };
  ctx.siblingsOf=id=>parents.has(id)?children.get(parents.get(id))||[]:ctx.rootIds;
  ctx.visibleOverview=()=>[];
  for(const id of [
    "q.reproducibility", "g.archive_wp_reports", "g.archive_wp_paths",
    "q.agent_flow", "g.archive_wt_history", "g.archive_wt_ci"
  ]){
    ctx.config.focusId=id;
    const nodes=ctx.run.visibleFocus();
    assert(nodes.some(n=>n.id===id),id+" focused node must appear in its real tree");
    assert(nodes.some(n=>n.id==="core:existing"),
      id+" must not collapse the tree into a detached orphan");
    assert.equal(ctx.ancestors(id)[0],"core:existing",
      id+" must have a traversable root ancestry");
  }
  if(useCoreAuthority){
    assert.equal(parents.get("q.reproducibility"),"q.research",
      "existing Core node must retain real parent");
    assert.equal(parents.get("g.archive_wp_reports"),"q.reproducibility",
      "proposed children must descend from their actual HC parent");
  }
}
inspect(true);
inspect(false); // Older review plans still get a safe root fallback.
console.log("Core HC07-HC12 focus, restored ancestry and stale-snapshot fallback passed");
