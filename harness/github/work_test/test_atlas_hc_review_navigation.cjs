/* Focused Atlas review: group-first traversal, unchecked arrows and uncluttered UI. */
"use strict";
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const assert=require("node:assert/strict");
const app=fs.readFileSync(path.join(__dirname,"../../visualization/app.js"),"utf8");
const css=fs.readFileSync(path.join(__dirname,"../../visualization/style.css"),"utf8");
const html=fs.readFileSync(path.join(__dirname,"../../visualization/index.html"),"utf8");
const i18n=fs.readFileSync(path.join(__dirname,"../../visualization/i18n.js"),"utf8");
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

// The very first node must be decidable, without switching trees.
function extract(start,next){
  const a=app.indexOf("  function "+start+"("),b=app.indexOf("  function "+next+"(",a+2);
  assert(a>=0&&b>a,"Missing "+start);
  return app.slice(a,b);
}
const savedDrafts=new Map();
const decisionItems=[
  {view:"core",node_id:"g.alpha",key:"core:g.alpha",proposal_review:null},
  {view:"code",node_id:"code:new",key:"code:code:new",proposal_review:null}
];
const decisionPlan={plan:{id:"HC-test-001",base_sha:"a".repeat(40),
  preview_sha256:"stable",viewed:{},items:decisionItems}};
const mock={
  review:decisionPlan,Map,Set,Date,JSON,
  browserDraftKey:plan=>"draft:"+plan.id,
  browserDraftStorage:()=>({setItem:(k,v)=>savedDrafts.set(k,v)}),
  proposalState:item=>item?.proposal_review?.decision||"unchecked",
  reviewDraftMessage:""
};
vm.createContext(mock);
vm.runInContext(extract("saveBrowserDraft","reviewHandoffData")+"\n"+
  extract("reviewHandoffData","reviewHandoffURL")+"\n"+
  extract("reviewHandoffURL","maybeShowReviewDone")+"\n"+
  "globalThis.inspect={saveBrowserDraft,reviewHandoffData,reviewHandoffURL}",
  mock,{timeout:1200});
mock.inspect.saveBrowserDraft("core:g.alpha","approved");
assert.equal(decisionItems[0].proposal_review.decision,"approved");
assert.equal(decisionItems[1].proposal_review,null,"other HC item must stay unchecked");
const savedReceipt=JSON.parse(savedDrafts.get("draft:HC-test-001"));
assert.equal(savedReceipt.decisions["core:g.alpha"],"approved");
assert.equal(savedReceipt.decisions["code:code:new"],undefined);
assert.equal(mock.inspect.reviewHandoffData().decisions["code:code:new"],"unchecked");
assert(mock.inspect.reviewHandoffURL().startsWith("https://github.com/OpenSAE/SAEBase/issues/new?title="));
assert(!app.includes("||!groupComplete"),"A group visit must not lock current approval buttons");


assert.equal((html.match(/id="atlasReviewSendReport"/g)||[]).length,1,"one report link");
assert(!html.includes("atlasReviewExport"),"remove JSON export");
assert(!app.includes("atlasReviewShare")&&!app.includes("atlasHumanCheckReport"),"no redundant report links");
assert(app.includes('review.plan.kind==="human_check" && review.editable'),"hide nonfunctional public toggle");
assert(app.includes('searchText("发送报告","Send report")'),"report label tracks language");
assert(!html.includes('id="atlasReviewSendReport" class="review-send-report is-hidden" href="#"'),
  "report must not be a fragment link");
assert(html.includes('id="atlasReviewSendReport" type="button"'),
  "report is an explicit button, not a reloadable anchor");
assert(app.includes("window.location.assign(url);"),
  "report must navigate to a prefilled GitHub issue form in the current tab");
assert(i18n.includes('"Send report":"发送报告"'),"English -> Chinese");
assert(i18n.includes('"发送报告":"Send report"'),"Chinese -> English");
assert(css.includes(".atlas-review-bar .review-send-report")&&css.includes("order:4"),"report positioning");
require("./test_atlas_node_reader.cjs");
assert(app.includes('"atlasReviewSendReport")?.addEventListener("click",sendBrowserReview)'),
  "single send report handler is wired");
assert(app.includes("persistBrowserDraft();"),"send report persists current draft");
const palette=fs.readFileSync(path.join(__dirname,"../../visualization/atlas-theme.css"),"utf8");
for(const state of ["unchecked","approved","denied"]){
  assert(palette.includes('node-selected .node-surface'),"selected node rules exist");
  assert(palette.includes('[data-review="'+state+'"].node-selected .node-surface'),
    "selected "+state+" retains the proper review color");
}
// Execute the real send handler. It must persist the receipt, then navigate
// to the prefilled GitHub Issue form; never stay on the Atlas '#' fragment.
let sentUrl="",persisted=0,prevented=0,stopped=0;
mock.window={location:{assign:url=>{sentUrl=url;}}};
mock.persistBrowserDraft=()=>{persisted++;};
mock.updateReviewBar=()=>{};
vm.runInContext(extract("sendBrowserReview","maybeShowReviewDone")+
  "\nthis.send=sendBrowserReview;",mock,{timeout:1200});
mock.send({preventDefault:()=>prevented++,stopPropagation:()=>stopped++});
assert.equal(persisted,1,"preserve the existing browser receipt");
assert.equal(prevented,1);
assert.equal(stopped,1);
assert(sentUrl.startsWith("https://github.com/OpenSAE/SAEBase/issues/new?"),
  "the click must leave Atlas for the GitHub form");
const destination=new URL(sentUrl);
assert.equal(destination.searchParams.get("title"),
  "Atlas Human Check decisions: HC-test-001");
const issueBody=destination.searchParams.get("body");
assert(issueBody.includes('"core:g.alpha": "approved"'),
  "send must carry the user's latest decision");
assert(issueBody.includes('"code:code:new": "unchecked"'),
  "send must carry unchecked items too");
console.log("Atlas HC ordering, palette and verified GitHub report redirect passed");
