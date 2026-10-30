/* Atlas navigation regression: synthetic 20-deep path, without changing source graphs. */
"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const path=require("node:path");
const src=fs.readFileSync(path.join(__dirname,"..","visualization","app.js"),"utf8");
function section(from,to){
  const a=src.indexOf("  function "+from+"("),b=src.indexOf("  function "+to+"(",a+5);
  assert(a>=0&&b>a,"missing function boundary: "+from);
  return src.slice(a,b);
}
const ids=Array.from({length:21},(_,i)=>"deep:"+i);
const byId=new Map(ids.map((id,i)=>[id,{id,title:"Deep level "+i}]));
const parents=new Map(ids.slice(1).map((id,i)=>[id,ids[i]]));
const children=new Map(ids.slice(0,-1).map((id,i)=>[id,[ids[i+1]]]));
const config={
  mode:"focus",focusId:"deep:20",zoom:.16,expanded:new Set(ids),query:"",
  positions:new Map()
};
const ctx={
  byId,parents,children,config,rootIds:[ids[0]],Map,Set,
  pagedChildren:id=>children.get(id)||[],
  siblingsOf:id=>parents.has(id)?children.get(parents.get(id))||[]:[ids[0]],
  measure:()=>({w:210,h:88,expanded:false}),
  $:()=>({clientWidth:1240,clientHeight:720})
};
vm.createContext(ctx);
vm.runInContext([
 section("ancestors","initialExpanded"),
 section("densityMode","measure"),
 section("visibleOverview","layoutOverview"),
 section("layoutFocus","normalizeLayout"),
 section("fitBounds","fitZoom"),
 section("fitZoom","centerSelected"),
 'globalThis.testExports={ancestors,densityMode,visibleOverview,layoutFocus,fitBounds,fitZoom};'
].join("\n"),ctx,{timeout:1000});
const f=ctx.testExports;
assert.deepEqual(Array.from(f.ancestors("deep:20")),ids,
  "Focus must retain every ancestor through level 20");
const l=f.layoutFocus();
assert.equal(l.visible.length,21,"all ancestors including root and focus must be displayed");
assert.deepEqual(Array.from(l.visible.map(n=>n.id)),ids);
assert(Array.from(l.visible).every(v=>l.positions.has(v.id)));
for(let i=1;i<ids.length;i++)
  assert(l.positions.get(ids[i]).x>l.positions.get(ids[i-1]).x,
    "all ancestors must occupy successive columns");
config.positions=l.positions;
const fitted=f.fitZoom();
assert(fitted>.08&&fitted<.5,"deep path must be fittable in one zoom gesture");
assert.equal(f.densityMode(),"compact");
config.mode="overview";
assert.equal(f.visibleOverview().length,21,"Overview must not truncate deep paths");
parents.set(ids[0],ids[20]);
assert.equal(f.ancestors(ids[20]).length,21,"cycle protection must still terminate");
assert(src.includes('canvas.addEventListener("wheel", event =>'));
assert(!src.includes('card.addEventListener("wheel", event =>'));
assert(src.includes("renderPathTrace();"));
assert(src.includes("Math.exp(-amount * .0017)"));
console.log("Atlas twenty-level ancestry, fit and wheel contracts passed");
