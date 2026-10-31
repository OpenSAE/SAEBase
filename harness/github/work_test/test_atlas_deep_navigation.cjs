/* Full-depth visible Focus levels and boundary-free camera regression. */
"use strict";
const assert=require("node:assert/strict");
const fs=require("node:fs");
const vm=require("node:vm");
const path=require("node:path");
const source=fs.readFileSync(path.join(__dirname,"..","..","visualization","app.js"),"utf8");
function section(from,to){
  const i=source.indexOf("  function "+from+"(");
  const j=source.indexOf("  function "+to+"(",i+5);
  assert(i>=0&&j>i,"missing method "+from+" or "+to);
  return source.slice(i,j);
}
const ids=Array.from({length:21},(_,i)=>"deep:"+i);
const byId=new Map(ids.map((id,i)=>[id,{id,title:"Depth "+i}]));
const parents=new Map(),children=new Map();
function edge(p,c){parents.set(c,p);if(!children.has(p))children.set(p,[]);children.get(p).push(c);if(!byId.has(c))byId.set(c,{id:c,title:c})}
for(let i=1;i<ids.length;i++)edge(ids[i-1],ids[i]);
for(let i=1;i<ids.length;i++)edge(ids[i-1],"side:"+i);
for(let i=0;i<12;i++)edge(ids[19],"more:20:"+i);
edge("side:19","exclude:20");
edge(ids[20],"child:21:0");
edge(ids[20],"child:21:1");
const config={mode:"focus",focusId:ids[20],zoom:.16,
  query:"",expanded:new Set(byId.keys()),positions:new Map(),
  panX:-123,panY:278};
const view={clientWidth:1240,clientHeight:720,
  getBoundingClientRect(){return {left:0,top:0}}};
const diagram={style:{},setAttribute(){}};
const ctx={byId,parents,children,config,rootIds:[ids[0]],Map,Set,
  pagedChildren:id=>children.get(id)||[],
  siblingsOf:id=>parents.has(id)?children.get(parents.get(id))||[]:[ids[0]],
  measure:()=>({w:210,h:88,expanded:false}),
  $:id=>id==="canvasWrap"?view:id==="diagram"?diagram:{textContent:""},
  render:()=>{},Math};
vm.createContext(ctx);
vm.runInContext([
  section("ancestors","initialExpanded"),
  section("densityMode","measure"),
  section("visibleOverview","layoutOverview"),
  section("layoutOverview","visibleFocus"),
  section("visibleFocus","layoutFocus"),
  section("layoutFocus","normalizeLayout"),
  section("fitBounds","fitZoom"),
  section("fitZoom","centerSelected"),
  section("centerSelected","renderPathTrace"),
  "globalThis.exposed={ancestors,densityMode,visibleOverview,visibleFocus,layoutFocus,fitBounds,fitZoom,centerSelected,changeZoom,setScale};"
].join("\n"),ctx,{timeout:1200});
const f=ctx.exposed;
assert.deepEqual(Array.from(f.ancestors(ids[20])),ids,"all 20 ancestors retained");
const nodes=Array.from(f.visibleFocus()),visible=new Set(nodes.map(n=>n.id));
assert.equal(nodes.filter(n=>n.depth===19).length,2,
  "all previous-level nodes must be visible, including other branches");
for(let i=0;i<20;i++)assert(visible.has(ids[i])&&
  (i===0||visible.has("side:"+i)),"missing previous level "+i);
const at20=nodes.filter(n=>n.depth===20);
assert.equal(at20.length,14,"the focused depth must include every one of its 14 siblings");
assert(at20.every(n=>parents.get(n.id)===ids[19]));
assert(!visible.has("exclude:20"),"other parents' nodes at the focused depth must be hidden");
assert(visible.has("child:21:0")&&visible.has("child:21:1"),
  "direct children remain available for drilling");
const layout=f.layoutFocus();
assert.equal(layout.positions.size,nodes.length,"every visible node must be laid out");
config.positions=layout.positions;
assert(f.fitZoom()>=.04&&f.fitZoom()<.5,"Fit must include all visible levels");
assert.equal(f.densityMode(),"compact");
// The world camera must permit positions outside the old scroll rectangle.
config.scene={width:layout.positions.size*210,height:300};
f.centerSelected();
const expectedX=view.clientWidth/2-config.zoom*(layout.positions.get(config.focusId).x+105);
assert.equal(config.panX,expectedX);
config.panX=-1800;config.panY=2200;
f.setScale();
assert(diagram.style.transform.includes("-1800px,2200px"),
  "free camera positions may be negative or exceed any content bounds");
const oldX=config.panX,oldY=config.panY;
f.changeZoom(.22,{clientX:350,clientY:175});
assert.notEqual(config.panX,oldX);
assert.notEqual(config.panY,oldY);
assert(source.includes('canvas.addEventListener("pointermove",event=>'));
assert(source.includes("config.panX=drag.panX+dx"));
assert(!source.includes("const boundary=()=>"));
assert(!source.includes("PAGE_SIZE"));
console.log("Complete previous levels, 14 siblings, 20 layers, and free camera passed");
