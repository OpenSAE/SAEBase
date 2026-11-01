/* Focused Atlas review: group-first traversal, unchecked arrows and uncluttered UI. */
"use strict";
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const assert=require("node:assert/strict");
const app=fs.readFileSync(path.join(__dirname,"../../visualization/app.js"),"utf8");
const css=fs.readFileSync(path.join(__dirname,"../../visualization/style.css"),"utf8");
function method(start,next){
  const a=app.indexOf("  function "+start+"("),b=app.indexOf("  function "+next+"(",a+2);
  assert(a>=0&&b>a,"Atlas method "+start+" must precede "+next);
  return app.slice(a,b);
}
function item(view,node,key){
  return {view,node_id:node,key,proposal_review:null};
}
const items=[
  item("core","g.alpha","core:g.alpha"),
  item("core","g.gamma","core:g.gamma"),
  item("core","g.beta","core:g.beta"),
  item("code","code:new","code:code:new"),
];
const groups=[
  {id:"HC01",title:"One logical change",item_keys:["core:g.alpha","code:code:new","core:g.beta"]},
  {id:"HC02",title:"Another change",item_keys:["core:g.gamma"]},
];
const review={plan:{items,groups,viewed:{}}};
const graph={
  core:{nodes:[{id:"core:existing"},{id:"g.alpha"},{id:"g.beta"},{id:"g.gamma"}],
    edges:[{source:"core:existing",target:"g.alpha",type:"contains"},
           {source:"core:existing",target:"g.gamma",type:"contains"},
           {source:"core:existing",target:"g.beta",type:"contains"}]},
  code:{nodes:[{id:"code:root"},{id:"code:new"}],
    edges:[{source:"code:root",target:"code:new",type:"contains"}]}
};
const config={view:"core",focusId:"g.alpha"};
const ctx={
  review,reviewQueue:[],reviewGroups:[],reviewGroupOf:new Map(),reviewGraphs:graph,
  REVIEW_VIEWS:["core","code"],REVIEW_TITLES:{core:"Core",code:"Code"},
  VIEWS:{core:{root:"core:existing",primary:["contains"]},
         code:{root:"code:root",primary:["contains"]}},
  DATA:graph,reviewFilter:{level:"all"},config,Map,Set,
  proposalState:x=>x?.proposal_review?.decision||"unchecked",
  logicalReviewGroup:key=>groups.find(g=>g.item_keys.includes(key))
};
vm.createContext(ctx);
vm.runInContext([
  method("buildReviewQueue","filteredReviewQueue"),
  method("filteredReviewQueue","reviewPosition"),
  method("reviewPosition","reviewGo"),
  method("findUnchecked","reviewGoUnchecked"),
  "globalThis.exposed={buildReviewQueue,filteredReviewQueue,reviewPosition,findUnchecked}"
].join("\n"),ctx,{timeout:1200});
const test=ctx.exposed;
test.buildReviewQueue();
assert.deepEqual(Array.from(ctx.reviewQueue,x=>x.key),[
  "core:g.alpha","code:code:new","core:g.beta","core:g.gamma"
], "all parts of HC01 must precede HC02 even across different trees");
assert.equal(test.findUnchecked(1),1,"single next reaches next HC01 part");
config.view="code";config.focusId="code:new";
review.plan.viewed.HC01={"core:g.alpha":{},"code:code:new":{}};
assert.equal(test.findUnchecked(1),2,"next unchecked part stays in HC01");
config.view="core";config.focusId="g.beta";
review.plan.viewed.HC01["core:g.beta"]={};
assert.equal(test.findUnchecked(1),3,"outer arrow reaches next unchecked HC after all current parts were viewed");
for(const x of items.filter(x=>groups[0].item_keys.includes(x.key)))
  x.proposal_review={decision:"approved"};
assert.equal(test.findUnchecked(1),3,"approved HC01 permits HC02");
assert(!app.includes("atlasReviewGroupScope"),"no HC01/02 group body box");
assert(!app.includes("Next linked part"),"no linked-part review button");
assert(app.includes('reviewDoneShown=true')&&app.includes('"All done"'),"completion dialog");
assert(app.includes('config.zoom=1.1')&&app.includes('config.expanded.add(id)'),
       "normal Focus must expand and slightly zoom on selected review node");
assert(app.includes('config.expandedAttributes.add(id)'),"selected node explanation expands");
assert(css.includes(".atlas-review-done"),"completion dialog styling");
assert(css.includes(".graph-node.node-selected[data-review]"),"review focus highlight");
assert(app.includes('"atlasReviewPrev","‹","Previous item'),"single prev navigates directly");
assert(app.includes('"atlasReviewNext","›","Next item'),"single next navigates directly");
assert(app.includes('"atlasReviewPrevUnchecked","«","Previous unchecked'),"outer prev skips unchecked");
assert(app.includes('"atlasReviewNextUnchecked","»","Next unchecked'),"outer next skips unchecked");
assert(app.includes('"atlasReviewPrev").addEventListener("click",()=>reviewGo(reviewPosition()-1))'),
       "inner left must force adjacent previous node");
assert(app.includes('"atlasReviewNext").addEventListener("click",()=>reviewGo(reviewPosition()+1))'),
       "inner right must force adjacent next node");
assert(app.includes('"atlasReviewPrevUnchecked").addEventListener("click",()=>reviewGoUnchecked(-1))'),
       "outer left must skip to prior unchecked node");
assert(app.includes('"atlasReviewNextUnchecked").addEventListener("click",()=>reviewGoUnchecked(1))'),
       "outer right must skip to following unchecked node");
assert(css.includes(".review-arrow.review-skip")&&css.includes("border:2px solid #8255b5"),
       "outer unchecked arrows need a prominent purple border");
console.log("Atlas HC ordering, navigation, zoom and completion UI passed");
