/* Deterministic public-snapshot navigation: bound real source nodes by
 * directory and callable responsibility. No source implementation is created.
 */
(function () {
  "use strict";
  const LIMIT = 10;
  const LABELS = {
    api:"SAEBase API 与特征接口",sae:"SAE 字典与加载",
    store:"特征数据与分片存储",llm:"语言模型与 SAE 操作",
    nlp:"NLP 样本与 Prompt",
    benchmark:"SAE 指标与可靠性评测",reliability:"统计可靠性",
    integrations:"SAR、RSS 与基准集成",backend:"计算执行后端",
    adapters:"外部基准适配",metrics:"评测指标计算",release:"评测结果发布",
    native:"原生执行与对照",extensions:"扩展评测接口",
    capture:"特征捕获",intervene:"受控特征干预",generate:"生成时特征编辑",
    circuits:"电路追踪与干预",explain:"特征自动解释",train:"SAE 训练",
    models:"模型实现",datasets:"数据集与输入",tasks:"科研任务",
    runtime:"训练与实验产物管理",saebase:"SAEBase 可复用算法"
  };
  const ROLES=[
    ["capture","特征捕获与读取",/captur|activat|feature|encode|decode|token/i],
    ["intervene","特征编辑与生成",/interven|edit|patch|steer|generat|infer|forward/i],
    ["circuit","电路与归因",/circuit|attribut|trac|gradient/i],
    ["explain","特征解释",/explain|interpret|hypothes/i],
    ["metrics","评测与统计",/metric|evaluat|bench|score|loss|rank|reliab|truth/i],
    ["train","训练与优化",/train|optim|schedul|epoch|checkp|learn|backward/i],
    ["storage","数据与保存",/data|prompt|artifact|shard|file|path|save|load|read|write|collect/i],
    ["contract","参数与校验",/valid|check|schema|contract|config|require|policy/i],
    ["runtime","运行与调度",/runtime|runner|execute|run|launch|build|create|setup/i]
  ];
  function sourcePath(n){
    const p=n.path||n.source_path||n.evidence?.find(x=>typeof x==="string"&&x.includes("/"));
    if(p)return String(p).split(":")[0];
    return String(n.id).replace(/^code:(?:module|class|method|function):/,"")
      .split(":")[0].replaceAll(".","/")+".py";
  }
  function prettify(text){
    return String(text||"").replace(/\.py$/,"").replace(/_/g," ")
      .replace(/([a-z0-9])([A-Z])/g,"$1 $2").trim();
  }
  function titleOf(n){
    const raw=n.source_symbol||n.title||sourcePath(n).split("/").pop();
    return prettify(raw).slice(0,65);
  }
  function identify(n){
    const symbol=String(n.source_symbol||n.title||"");
    for(const [key,_label,pattern] of ROLES)
      if(pattern.test(symbol))return key;
    const p=sourcePath(n).split("/").pop();
    for(const [key,_label,pattern] of ROLES)
      if(pattern.test(p))return key;
    return "misc";
  }
  function directoryBuckets(kids,byId){
    const paths=kids.map(id=>sourcePath(byId.get(id)).split("/").slice(0,-1));
    if(paths.some(parts=>!parts.length))return [];
    let prefix=0;
    while(paths.every(parts=>parts.length>prefix &&
          parts[prefix]===paths[0][prefix]))prefix++;
    const groups=new Map();
    for(let i=0;i<kids.length;i++){
      const key=(paths[i][prefix]||"entry").toLowerCase();
      if(!groups.has(key))groups.set(key,[]);
      groups.get(key).push(kids[i]);
    }
    if(groups.size<2||groups.size>LIMIT)return [];
    return [...groups].sort(([a],[b])=>a.localeCompare(b)).map(([key,ids])=>
      [key,LABELS[key]||prettify(key)+" 源码模块",ids]);
  }
  function roleBuckets(kids,byId){
    const groups=new Map();
    for(const id of kids){
      const key=identify(byId.get(id));
      if(!groups.has(key))groups.set(key,[]);
      groups.get(key).push(id);
    }
    if(groups.size<2||groups.size>LIMIT)return [];
    return [...groups].sort(([a],[b])=>a.localeCompare(b)).map(([key,ids])=>
      [key,ROLES.find(row=>row[0]===key)?.[1]||"其他公共源码接口",ids]);
  }
  function sourceBuckets(kids,byId){
    const ordered=[...kids].sort((a,b)=>{
      const na=byId.get(a),nb=byId.get(b);
      return sourcePath(na).localeCompare(sourcePath(nb))||
        (Number(na.line)||0)-(Number(nb.line)||0)||a.localeCompare(b);
    });
    const size=Math.max(LIMIT,Math.ceil(ordered.length/LIMIT));
    const results=[];
    for(let i=0;i<ordered.length;i+=size){
      const ids=ordered.slice(i,i+size);
      const first=byId.get(ids[0]),last=byId.get(ids[ids.length-1]);
      results.push([String(i/size),titleOf(first)+" 至 "+titleOf(last),ids]);
    }
    return results;
  }
  function hashed(value){
    let h=2166136261;
    for(let i=0;i<value.length;i++){
      h^=value.charCodeAt(i);
      h=Math.imul(h,16777619);
    }
    return (h>>>0).toString(16).padStart(8,"0");
  }
  function balance(graph,name="code"){
    if(!graph?.nodes||!graph?.edges)return {inserted:0};
    const byId=new Map(graph.nodes.map(n=>[n.id,n]));
    let inserted=0;
    for(let iteration=0;iteration<1000;iteration++){
      const children=new Map();
      for(const e of graph.edges){
        if(e.type!=="contains")continue;
        if(!children.has(e.source))children.set(e.source,[]);
        children.get(e.source).push(e.target);
      }
      const over=[...children].filter(([,kids])=>kids.length>LIMIT)
        .sort((a,b)=>b[1].length-a[1].length||a[0].localeCompare(b[0]));
      if(!over.length)return {inserted};
      const [parent,kids]=over[0];
      const parts=directoryBuckets(kids,byId).length>1 ?
        directoryBuckets(kids,byId) :
        roleBuckets(kids,byId).length>1 ?
          roleBuckets(kids,byId) : sourceBuckets(kids,byId);
      if(parts.length>LIMIT||parts.length<2)
        throw Error("Cannot balance public "+name+" branch "+parent);
      const moved=new Set(kids);
      graph.edges=graph.edges.filter(e=>e.type!=="contains"||
        !(e.source===parent&&moved.has(e.target)));
      for(const [key,label,ids] of parts){
        const groupId=name+":public-nav:"+hashed(parent+"|"+key+"|"+label);
        if(byId.has(groupId))
          throw Error("Repeated public navigation group "+groupId);
        const evidence=[...new Set(ids.map(id=>sourcePath(byId.get(id))))
          ].filter(Boolean).slice(0,30);
        const entry={
          id:groupId,kind:"group",title:label,
          summary:"按实际源码位置和接口职责整理 "+ids.length+" 个已有节点。",
          description:"依据当前已发布的源文件与方法标识组织导航。",
          status:"implemented",presentation_only:true,evidence,
          context_ids:[],core_ids:[],source_symbol:""
        };
        graph.nodes.push(entry);byId.set(groupId,entry);inserted++;
        graph.edges.push({source:parent,target:groupId,type:"contains"});
        for(const id of ids)
          graph.edges.push({source:groupId,target:id,type:"contains"});
      }
    }
    throw Error("Public Atlas hierarchy convergence limit exceeded");
  }
  window.OPENRESEARCH_ATLAS_BALANCE=balance;
})();
