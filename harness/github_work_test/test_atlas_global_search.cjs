/* Five-tree search relevance, explicit relations and node-link contract. */
"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs"),vm=require("node:vm"),path=require("node:path");
const app=fs.readFileSync(path.join(__dirname,"..","visualization","app.js"),"utf8");
function extract(name,next) {
  const first=app.indexOf("  function "+name+"(");
  const last=app.indexOf("  function "+next+"(",first+5);
  assert(first>=0&&last>first,"could not extract "+name);
  return app.slice(first,last);
}
const DATA={
  core:{nodes:[{id:"c1",title:"SAE research architecture",description:"research platform"}]},
  knowledge:{nodes:[{id:"k1",title:"Feature compositionality",description:"SAE interpretability"}]},
  code:{nodes:[{id:"code.capture",title:"Capture activation",
    source_symbol:"saebase.capture",signature:"capture(model)",description:"token activations"}]},
  environment:{nodes:[{id:"e1",title:"GPU runtime",description:"execution conditions"}]},
  context:{nodes:[{id:"t1",title:"Sparse Autoencoder",aliases:["SAE"],description:"feature encoder"}]}
};
const context={DATA,reviewGraphs:{},Map,Set,document:{documentElement:{lang:"en"}},
  plain:x=>String(x||""),URL,URLSearchParams,
  window:{location:{href:"https://example.org/opensae/saebase/atlas/code.html?lang=en&review=private#key=secret",
    search:"?lang=en&review=private"}},
  config:{view:"code"},String};
vm.createContext(context);
vm.runInContext([
 'const ATLAS_VIEW_LABELS={core:"Core",knowledge:"Knowledge",code:"Code",environment:"Environment",context:"Context"};',
 extract("globalSearchEntries","updateGlobalSearch"),
 extract("nodeDeepLink","copyDeepLink").replace("  async function copyDeepLink(id,button) {",""),
 'globalThis.api={globalSearchEntries,nodeDeepLink};'
].join("\n"),context,{timeout:1000});
const f=context.api;
assert.equal(f.globalSearchEntries("").length,5,"all trees must be indexed");
assert.equal(f.globalSearchEntries("saebase.capture")[0].id,"code.capture");
assert.equal(f.globalSearchEntries("saebase.capture")[0].view,"code");
assert.equal(f.globalSearchEntries("sae").some(x=>x.view==="context"),true);
assert.equal(f.globalSearchEntries("execution")[0].view,"environment");
assert.equal(f.globalSearchEntries("nonexistent").length,0);
context.reviewGraphs.knowledge={nodes:[{id:"draft1",title:"Draft research feature"}]};
assert.equal(f.globalSearchEntries("draft")[0].id,"draft1",
  "review overlay should be searchable when active");
const url=f.nodeDeepLink("code.capture");
const u=new URL(url);
assert.equal(u.pathname,"/opensae/saebase/atlas/code.html");
assert.equal(u.searchParams.get("focus"),"code.capture");
assert.equal(u.searchParams.get("lang"),"en");
assert(!u.searchParams.has("review"),"review keys must never leak into copied links");
assert(!u.hash,"review hash must not leak");
assert(app.includes("installGlobalSearch();"));
assert(app.includes('event.key==="ArrowDown"'));
assert(app.includes('event.key==="ArrowUp"'));
assert(app.includes('event.key==="Enter"'));
assert(app.includes('closeGlobalSearch()'));
for(const token of ["context_term_ids","context_ids","core_ids","term_ids",
  "atlas-relation-field","atlas-copy-link","atlas-detail-layer"]){
  assert(app.includes(token),"missing relation/evidence feature "+token);
}
console.log("Atlas global five-tree search, relation fields, and safe deep links passed");
