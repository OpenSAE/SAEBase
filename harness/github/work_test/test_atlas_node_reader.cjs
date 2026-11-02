/* Reader semantics are testable without opening a new browser page. */
"use strict";
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const assert=require("node:assert/strict");
const root=path.resolve(__dirname,"../../..");
const js=fs.readFileSync(path.join(root,"harness/visualization/node-reader.js"),"utf8");
const css=fs.readFileSync(path.join(root,"harness/visualization/node-reader.css"),"utf8");
const app=fs.readFileSync(path.join(root,"harness/visualization/app.js"),"utf8");
const html=fs.readFileSync(path.join(root,"harness/visualization/index.html"),"utf8");
const env={window:{SAEBASE_ATLAS_LANGUAGE:"zh"},URLSearchParams,document:{documentElement:{lang:"zh"}}};
vm.createContext(env);vm.runInContext(js,env,{timeout:1200});
const {presentation,differs,create}=env.window.SAEBASE_ATLAS_NODE_READER;
assert.equal(typeof create,"function");
const base={id:"example",title:"Existing",description:"Current meaning",attributes:[{label:"Fields",value:"v1"}]};
const newItem={...base,title:"Updated",description:"Proposed meaning"};
let result=presentation({operation:"add",node:newItem,after:newItem});
assert.equal(result.operation,"add");
assert.equal(result.cards.length,1);
assert.equal(result.cards[0].side,"after");
result=presentation({operation:"remove",node:base,before:base,after:null});
assert.equal(result.cards.length,1);
assert.equal(result.cards[0].side,"before");
result=presentation({operation:"change",node:newItem,before:base,after:newItem});
assert.equal(result.cards.length,2);
assert.equal(result.cards[0].side,"before");
assert.equal(result.cards[1].side,"after");
assert(differs(base.description,newItem.description));
assert(!differs(base.attributes,newItem.attributes));
assert(app.includes('"node-operation-"+itemFor(node.id,config.view).operation'));
assert(app.includes("showNodeReader()")&&app.includes("nodeReader?.close()"));
assert(js.includes("data.evidence")&&js.includes("renderAttributes"));
assert(js.includes("event.deltaY")&&js.includes(".atlas-reader-scroll"));
assert(html.includes("node-reader.js")&&html.includes("node-reader.css"));
assert(css.includes(".atlas-reader-pages.is-compare")&&css.includes(".atlas-reader-card.is-removing"));
assert(css.includes("pointer-events:none")&&css.includes("pointer-events:auto"),
  "outside reader panel must pass wheel events to graph");
assert(css.includes("body.atlas-reader-active #atlasPathTrace"),
  "reader must remove redundant ancestor strip");
assert(css.includes(".atlas-reader-diff-remove")&&
  css.includes("text-decoration:line-through")&&
  css.includes(".atlas-reader-diff-add"),"red editorial additions/deletions");
assert(!js.includes("threshold+=event.deltaY"),"reader must not exit on content-scroll boundary");
assert(!js.includes("atlas-reader-hint"),"obsolete instruction microcopy absent");
assert(js.includes('setReviewState(entry.reviewState || "")'),"Reader palette follows review");
assert(css.includes('[data-review-state="unchecked"]')&&
  css.includes('[data-review-state="approved"]')&&
  css.includes('[data-review-state="denied"]'),"Reader highlights correspond to decision");
// All three Human Check operations use the same Graph-first landing.
assert(!app.includes('reviewTarget && reviewItem.operation==="change"'),
  "Change must not open Reader automatically");
assert(app.includes('!!options.toggle'),"explicit card click opens Reader");
assert(app.includes('if(!shouldOpenReader)nodeReader?.close()'),
  "Graph navigation dismisses any previous Reader");
assert(app.includes('config.detailsOpen = !nodeReader || (!reviewTarget && !shouldOpenReader)'),
  "Graph review does not open the old detail drawer");
assert(app.includes('if(decision==="approved" ||'),
  "approval always advances, even when Reader was open");
assert(app.includes('nodeReader?.close();\n        const next=findUnchecked(1);'),
  "approval closes Reader before selecting the next unchecked node");
assert(app.includes('if(next>=0)reviewGo(next);'),
  "approval navigates to the actual next unchecked node");
assert(app.includes('nodeReader.setReviewState(status)'),
  "the review bar synchronizes live state on the active Reader");
const setter=js.match(/    function setReviewState\(state\)\{[\s\S]*?\n    \}/);
assert(setter,"Reader must expose state-only update");
const stateHarness={shell:{dataset:{}}};
vm.createContext(stateHarness);
vm.runInContext(setter[0]+"\nthis.apply=setReviewState;",stateHarness,{timeout:1200});
for(const value of ["unchecked","approved","denied","unchecked"]){
  stateHarness.apply(value);
  assert.equal(stateHarness.shell.dataset.reviewState,value,
    "state toggle must update without closing or re-rendering Reader");
}
for(const value of ["unchecked","approved","denied"]){
  const scope='.atlas-reader-shell[data-review-state="'+value+'"]';
  assert(css.includes(scope),"missing review palette "+value);
}
assert(css.includes('color:var(--hc-mark')&&css.includes('border-color:var(--hc-card'),
  "diff text and both reader frames share the live palette");
console.log("SAEBase Atlas Reader: operation defaults and live HC palette PASS");
