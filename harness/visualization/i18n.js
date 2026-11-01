/* Atlas language control. Language-neutral graph IDs and state are untouched. */
(function(){
"use strict";
const KEY="saebase.atlas.locale";
const params=new URLSearchParams(location.search);
const requested=params.get("lang");
const saved=(()=>{try{return localStorage.getItem(KEY)}catch{return null}})();
let locale=["en","zh"].includes(requested)?requested:(["en","zh"].includes(saved)?saved:(navigator.language?.toLowerCase().startsWith("zh")?"zh":"en"));
window.SAEBASE_ATLAS_LANGUAGE=locale;
const CJK=/[\u3400-\u9fff]/;
const EXACT={
"展开节点属性与依据":"Expand attributes and evidence","这个节点在做什么":"What this node does",
"专业词汇 · Context":"Technical terms · Context","科研术语定义与源码依据":"Research concepts and source evidence",
"OpenResearch 双核心与任务注册":"OpenResearch architecture and task registration",
"OpenResearch Python 实现树":"OpenResearch Python implementation tree",
"CPU、GPU 与科研任务执行环境":"CPU, GPU, and research execution environment",
"Atlas 五棵 Tree 的职责与关联":"Responsibilities and connections of the five Atlas trees",
"Dataset、Model、Task 插件注册":"Dataset, Model, and Task plugin registration",
"Task 执行前的注册与兼容检查":"Registration and compatibility checks before task execution",
"Hydra 配置、随机种子与实验结果追溯":"Hydra configuration, random seeds, and experiment traceability",
"训练随机种子与运行环境记录":"Training random seeds and runtime records",
"LLM 语言模型":"LLM language models","LLM 稀疏特征任务":"LLM sparse feature tasks",
"具体数据集":"Specific datasets","具体模型":"Specific models","具体任务":"Specific tasks",
"暂无经过验证的功能说明，请结合源代码审核。":"Source-linked description pending review.",
"已有实现":"Implemented","待审阅":"Pending review","已批准":"Approved","已否决":"Denied",
"发送报告":"Send report","发送当前审阅报告":"Send current review report",
"在 GitHub 确认并提交报告":"Review and submit the report on GitHub"
};
Object.assign(EXACT, {
  "收起节点属性": "Collapse node attributes",
  "查看详情": "View details",
  "查看功能说明与源码依据": "View function and source evidence",
  "展开节点说明与依据": "Expand node explanation and evidence",
  "展开或收起子节点": "Expand or collapse child nodes",
  "在 Code Tree 中查看对应实现": "View implementation in Code Tree",
  "查看真实源码": "View source code",
  "查看源文件": "View source file",
  "查看完整来源": "View complete source",
  "查看依据": "View evidence",
  "尝试其他检索词": "Try another search",
  "五树搜索": "Search all trees",
  "搜索五棵树 · Ctrl K": "Search five trees · Ctrl K",
  "复制链接": "Copy link",
  "已复制": "Copied",
  "选择链接复制": "Select link to copy",
  "选择节点查看关系与依据": "Select a node for relationships and evidence",
  "当前节点": "Node",
  "跨树关联": "Relations",
  "来源与依据": "Evidence",
  "被引用": "Referenced by",
  "引用": "References",
  "收起": "Show fewer",
  "显示全部关联": "Show all relations",
  "此节点暂无明确记录的跨树关联。": "No explicit cross-tree relation is recorded for this node.",
  "此节点的来源信息请查看展开节点属性。": "Open node attributes to inspect additional source evidence.",
  "演示配置": "Synthetic configuration",
  "研究意义": "Why this matters",
  "别名": "Aliases",
  "下一步": "Next step",
  "相关术语": "Context terms",
  "科研知识": "Knowledge",
  "项目目标": "Core goals",
  "术语引用": "Term reference",
  "术语定义": "Terminology",
  "源码标识": "Source symbol",
  "方法签名": "Signature",
  "设备与执行信息": "Execution facts",
  "直接源码": "Direct source",
  "背景依据": "Background",
  "证据来源": "Evidence sources",
  "变化证据": "Observed changes",
  "全部": "All",
  "没有匹配的节点": "No matching nodes",
  "选择": "Select",
  "跳转": "Open",
  "关闭": "Close",
  "搜索所有节点、术语、函数…": "Search nodes, terms, functions…",
  "在 Context 中查看": "View in Context",
  "成员数据": "Member data",
  "主要作用": "Purpose",
  "返回值": "Return value",
  "函数签名": "Function signature",
  "输入参数": "Inputs",
  "默认值": "Default value",
  "类": "Class",
  "函数": "Function",
  "查看完整函数源码": "View complete function source",
  "可核对的科研概念示意": "Verifiable research concept diagram",
  "输入 x": "Input x",
  "输出": "Output",
  "捷径连接": "Shortcut connection",
  "演示环境 · 合成设备配置": "Demo environment · synthetic device configuration",
  "显示 ": "Showing ",
  "保存的字段": "Stored fields",
});
const UI={"Guide":"Guide","Overview":"Overview","Details":"Details","Outline":"Outline","Fit":"Fit","Approve":"Approve","Deny":"Deny","Unchecked":"Unchecked","Partial":"Partial","Implemented":"Implemented","Not Implemented":"Not Implemented","Compare code changes":"Compare code changes","Find a node…":"Find a node…"};
const ZH_UI={
"Guide":"指南","Overview":"总览","Details":"详情","Outline":"目录","Fit":"适应画布",
"Send report":"发送报告","Send current review report":"发送当前审阅报告",
"Review and submit the report on GitHub":"在 GitHub 确认并提交报告",
"All done":"全部完成","All Human Check review items are complete.":"所有 Human Check 节点均已审阅。",
"Approve":"批准","Deny":"否决","Unchecked":"待审阅","Partial":"部分完成",
"Implemented":"已实现","Not Implemented":"尚未实现",
"Compare code changes":"比较代码变化","Find a node…":"搜索节点…",
"Return to overview":"返回总览","Toggle outline":"切换目录",
"Show node details and links":"显示节点详情与关联",
"Expand tree":"展开树","Collapse tree":"收起树",
"Previous unchecked node":"上一个待审节点","Next unchecked node":"下一个待审节点",
"Close details":"关闭详情","Close outline":"关闭目录",
"Review nodes":"审阅节点","Project tree outline":"项目树目录"
};
const ATTRS=["title","aria-label","placeholder"];
const native=new WeakMap();
const technical=(el)=>!!el?.closest?.("code,pre,kbd,.code-source,.source-code,.signature,.code-block,[data-i18n-verbatim]");
function localize(raw,el){
 if(!raw||!CJK.test(raw))return raw;
 if(EXACT[raw.trim()])return raw.replace(raw.trim(),EXACT[raw.trim()]);
 const fullEvidence=raw.trim().match(/^(.+) 的完整解释与证据$/);
 if(fullEvidence)return "Full explanation and evidence for " + fullEvidence[1];
 const contextual=raw.trim().match(/^在 Context 中查看“(.+)”$/);
 if(contextual)return 'View "' + contextual[1] + '" in Context';
 const memberList=raw.trim().match(/^(类|函数)：(.+)$/);
 if(memberList)return (memberList[1]==="类"?"Classes: ":"Functions: ")+memberList[2];

 const known=window.OPENRESEARCH_ATLAS_I18N?.en?.[raw.trim()];
 if(typeof known==="string"&&known.trim())return raw.replace(raw.trim(),known);
 // Preserve exact code excerpts and recorded source facts.
 if(technical(el))return raw;
 // Localized overlays can provide translated strings without changing underlying graph data.
 const translated=window.SAEBASE_ATLAS_UI?.[raw.trim()];
 if(typeof translated==="string"&&translated.trim())return raw.replace(raw.trim(),translated);
 // A missing English mapping is a build-time localization defect, not user-facing prose.
 // Hide only the unknown explanatory text; technical source remains untouched.
 return raw;
}
function updateText(node){
 const el=node.parentElement;if(!el||el.closest("#atlasLanguageSwitch")||["SCRIPT","STYLE","NOSCRIPT"].includes(el.tagName))return;
 if(!native.has(node))native.set(node,node.nodeValue);
 const source=native.get(node);
 const next=locale==="en"?localize(source,el):(ZH_UI[source.trim()]||source);
 if(node.nodeValue!==next)node.nodeValue=next;
}
// Documentation links follow stable source document IDs across languages.
function translateLinks(root=document){
 const english=window.SAEBASE_ATLAS_LOCALES?.docs||{};
 for(const anchor of root.querySelectorAll?.("a[href]")||[]){
  if(!anchor.dataset.atlasOriginalHref)anchor.dataset.atlasOriginalHref=anchor.getAttribute("href");
  const raw=anchor.dataset.atlasOriginalHref;
  let resolved=raw;
  const hit=raw.match(/^(https:\/\/github\.com\/OpenSAE\/SAEBase\/blob\/main\/)([^?#]+)(.*)$/);
  if(hit&&locale==="en"&&english[hit[2]])resolved=hit[1]+english[hit[2]]+hit[3];
  if(resolved!==anchor.getAttribute("href"))anchor.setAttribute("href",resolved);
 }
}
function translate(root=document){
 const walk=document.createTreeWalker(root,NodeFilter.SHOW_TEXT);let node;
 while((node=walk.nextNode()))updateText(node);
 const elements=root.querySelectorAll?.("[title],[aria-label],[placeholder]")||[];
 for(const el of elements){
  if(el.closest("#atlasLanguageSwitch"))continue;
  for(const attr of ATTRS){
   if(!el.hasAttribute(attr))continue;
   const key="atlasOriginal"+attr.replace("-","").replace(/^./,x=>x.toUpperCase());
   if(!el.dataset[key])el.dataset[key]=el.getAttribute(attr);
   const source=el.dataset[key];const value=locale==="en"?localize(source,el):(ZH_UI[source.trim()]||source);
   if(el.getAttribute(attr)!==value)el.setAttribute(attr,value);
  }
 }
 translateLinks(root);
 document.documentElement.lang=locale==="zh"?"zh-CN":"en";
 document.querySelectorAll("#atlasLanguageSwitch button").forEach(button=>{
  const selected=button.dataset.lang===locale;
  button.setAttribute("aria-pressed",String(selected));
  button.classList.toggle("active",selected);
 });
}
let active=false;
const watcher=new MutationObserver(records=>{
 if(active)return;
 active=true;watcher.disconnect();
 for(const record of records){
  if(record.type==="characterData"){
   const n=record.target;
   if(native.has(n)&&n.nodeValue!==native.get(n)&&n.nodeValue!=="")native.set(n,n.nodeValue);
  }
 }
 translate();
 watcher.observe(document.body,{subtree:true,childList:true,characterData:true});
 active=false;
});
function setLanguage(next){
 if(!["en","zh"].includes(next))return;
 locale=next;window.SAEBASE_ATLAS_LANGUAGE=next;try{localStorage.setItem(KEY,next)}catch{}
 const url=new URL(location.href);url.searchParams.set("lang",next);
 history.replaceState(history.state,"",url);
 window.SAEBASE_ATLAS_SET_NODE_LANGUAGE?.(locale);
 translate();
}
function initialize(){
 const switcher=document.getElementById("atlasLanguageSwitch");
 if(!switcher)return;
 switcher.addEventListener("click",e=>{const b=e.target.closest("button[data-lang]");if(b)setLanguage(b.dataset.lang)});
 document.querySelectorAll("a[href]").forEach(a=>{
  if(!/^(?:core|knowledge|code|environment|context)\.html(?:[?#]|$)/.test(a.getAttribute("href")||""))return;
  a.addEventListener("click",()=>{try{localStorage.setItem(KEY,locale)}catch{}});
 });
 window.SAEBASE_ATLAS_SET_NODE_LANGUAGE?.(locale);
 translate();
 watcher.observe(document.body,{subtree:true,childList:true,characterData:true});
}
if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",initialize,{once:true});else initialize();
window.SAEBASE_ATLAS_SET_LANGUAGE=setLanguage;
})();
