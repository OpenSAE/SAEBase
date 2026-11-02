/* SAEBase Atlas Node Reader: one source-backed node, two faces for a change. */
(function () {
  "use strict";
  const L = (zh,en) => String(window.SAEBASE_ATLAS_LANGUAGE ||
      new URLSearchParams(window.location.search).get("lang") ||
      document.documentElement.lang || "en").toLowerCase().startsWith("zh") ? zh : en;
  const names = {
    add:["新增节点","Add node"],remove:["删除节点","Remove node"],
    change:["修改节点","Change node"],view:["节点阅读","Node reader"]
  };
  function presentation(entry) {
    const op = ["add","remove","change"].includes(entry.operation) ? entry.operation : "view";
    const cards = op === "change" && entry.before ?
      [{side:"before",data:entry.before},{side:"after",data:entry.after || entry.node}] :
      op === "remove" ? [{side:"before",data:entry.before || entry.node}] :
      [{side:"after",data:entry.after || entry.node}];
    return {operation:op,cards};
  }
  function differs(a,b) { return JSON.stringify(a) !== JSON.stringify(b); }

  function displayValue(value) {
    return Array.isArray(value)?value.join(" · "):
      value && typeof value==="object"?JSON.stringify(value,null,2):
      String(value??"");
  }
  function tokenize(text) {
    return Array.from(String(text).matchAll(/\s+|[\p{Script=Han}]|[\p{L}\p{N}_]+|[^\s]/gu),m=>m[0]);
  }
  // Bounded word/Chinese-character LCS, never interpreting untrusted HTML.
  function diffSegments(oldValue,newValue) {
    const before=tokenize(displayValue(oldValue)),after=tokenize(displayValue(newValue));
    const oldParts=[],newParts=[];
    const put=(target,kind,value)=>{
      if(!value)return;
      const last=target[target.length-1];
      if(last?.kind===kind)last.text+=value;
      else target.push({kind,text:value});
    };
    const n=before.length,m=after.length;
    if(n*m>200000) {
      let start=0,end=0;
      while(start<Math.min(n,m)&&before[start]===after[start])start++;
      while(end<Math.min(n-start,m-start)&&before[n-1-end]===after[m-1-end])end++;
      const prefix=before.slice(0,start).join("");
      const suffix=end?before.slice(n-end).join(""):"";
      put(oldParts,"same",prefix);put(newParts,"same",prefix);
      put(oldParts,"remove",before.slice(start,n-end).join(""));
      put(newParts,"add",after.slice(start,m-end).join(""));
      put(oldParts,"same",suffix);put(newParts,"same",suffix);
      return {before:oldParts,after:newParts};
    }
    const dp=Array.from({length:n+1},()=>new Uint32Array(m+1));
    for(let i=n-1;i>=0;i--)
      for(let j=m-1;j>=0;j--)
        dp[i][j]=before[i]===after[j]?dp[i+1][j+1]+1:
          Math.max(dp[i+1][j],dp[i][j+1]);
    let i=0,j=0;
    while(i<n||j<m){
      if(i<n&&j<m&&before[i]===after[j]){
        put(oldParts,"same",before[i]);put(newParts,"same",after[j]);i++;j++;
      }else if(i<n&&(j===m||dp[i+1][j]>=dp[i][j+1])){
        put(oldParts,"remove",before[i++]);
      }else put(newParts,"add",after[j++]);
    }
    return {before:oldParts,after:newParts};
  }
  function appendDiff(target,oldValue,newValue,side){
    for(const part of diffSegments(oldValue,newValue)[side]){
      target.appendChild(element(part.kind==="remove"?"del":
        part.kind==="add"?"ins":"span","atlas-reader-diff-"+part.kind,part.text));
    }
  }

  function element(tag,className,text) {
    const el = document.createElement(tag);
    if(className)el.className=className;
    if(text !== undefined && text !== null)el.textContent=String(text);
    return el;
  }
  function create({host}) {
    let shell=null,cardWrap=null,current=null,sourceRect=null;
    function animateFromNode(panel,reverse=false) {
      if(!panel?.animate || !sourceRect || !sourceRect.width || !sourceRect.height)return;
      const rect=panel.getBoundingClientRect();
      if(!rect.width || !rect.height)return;
      const from="translate("+(sourceRect.left-rect.left)+"px,"+
        (sourceRect.top-rect.top)+"px) scale("+
        Math.max(.08,Math.min(1,sourceRect.width/rect.width))+","+
        Math.max(.08,Math.min(1,sourceRect.height/rect.height))+")";
      panel.animate(reverse?
        [{opacity:1,transform:"none"},{opacity:.25,transform:from}]:
        [{opacity:.25,transform:from},{opacity:1,transform:"none"}],
        {duration:240,easing:"cubic-bezier(.2,.8,.2,1)"});
    }
    function close() {
      if(!shell)return;
      const old=shell, panel=cardWrap;
      shell=null; cardWrap=null;current=null;
      document.body.classList.remove("atlas-reader-active");
      animateFromNode(panel,true);
      old.remove();
    }
    function section(root,title) {
      const section=element("section","atlas-reader-section");
      section.appendChild(element("h3","atlas-reader-section-title",title));
      root.appendChild(section);
      return section;
    }
    function field(root,label,value,changed=false,comparison=null,side="after") {
      if(value === undefined || value === null || value === "")return;
      if(Array.isArray(value) && !value.length)return;
      const row=element("div","atlas-reader-field"+(changed?" is-changed":""));
      row.appendChild(element("div","atlas-reader-field-label",label));
      const body=element("div","atlas-reader-field-value");
      if(changed)appendDiff(body,side==="before"?value:comparison,
        side==="before"?comparison:value,side);
      else body.textContent=displayValue(value);
      row.appendChild(body);root.appendChild(row);
    }
    function sourceLink(root,name,path,line) {
      const href=current?.sourceHref?.(path,line);
      if(!href)return;
      const a=element("a","atlas-reader-source-link",name);
      a.href=href;a.target="_blank";a.rel="noopener noreferrer";
      root.appendChild(a);
    }
    function relationLink(root,record) {
      const button=element("button","atlas-reader-link");
      button.type="button";
      const label=record.label || record.role || record.field || "";
      button.appendChild(element("span","atlas-reader-link-type",label));
      button.appendChild(element("span","atlas-reader-link-title",
        record.title || record.id));
      button.appendChild(element("span","atlas-reader-link-tree",record.view || ""));
      button.addEventListener("click",()=>{
        current?.onFollow?.(record.id,record.view);
      });
      root.appendChild(button);
    }
    function renderCard(holder,part,other,operation) {
      const data=part.data || {};
      const otherData=other?.data || {};
      const isBefore=part.side==="before";
      const card=element("article","atlas-reader-card is-"+part.side+
        (operation==="remove"?" is-removing":""));
      card.setAttribute("aria-label",isBefore?L("修改前的节点内容","Node before change"):
        L("修改后的节点内容","Node after change"));
      const heading=element("div","atlas-reader-card-head");
      heading.appendChild(element("span","atlas-reader-card-side",
        operation==="change"?(isBefore?L("修改前","Before"):L("修改后","After")):
        operation==="remove"?L("现有节点 · 拟删除","Existing node · proposed removal"):
        operation==="add"?L("拟新增节点","Proposed node"):L("当前节点","Current node")));
      const cardTitle=element("h2","atlas-reader-card-title");
      if(operation==="change"&&differs(data.title,otherData.title))
        appendDiff(cardTitle,isBefore?data.title:otherData.title,
          isBefore?otherData.title:data.title,part.side);
      else cardTitle.textContent=data.title||current.node.title;
      heading.appendChild(cardTitle);
      card.appendChild(heading);
      const scroll=element("div","atlas-reader-scroll");
      scroll.tabIndex=0;
      const changed=key=>operation==="change" && differs(data[key],otherData[key]);
      const readField=(root,label,key)=>field(root,label,data[key],
        changed(key),otherData[key],part.side);
      const explanatory=section(scroll,L("节点说明","Explanation"));
      readField(explanatory,L("摘要","Summary"),"summary");
      readField(explanatory,L("完整说明","Description"),"description");
      const extra=data.reader_explanation;
      if(extra && extra!==data.description && extra!==data.summary)
        readField(explanatory,L("详细解释","Detailed explanation"),"reader_explanation");
      readField(explanatory,L("研究意义","Why this matters"),"why_it_matters");
      readField(explanatory,L("下一步","Next step"),"next");
      readField(explanatory,L("别名","Aliases"),"aliases");
      const hasAttributes=Array.isArray(data.attributes) && data.attributes.length;
      if(data.signature || data.source_symbol || data.path || hasAttributes ||
         data.facts || data.technical_description){
        const attributes=section(scroll,L("属性与实现","Attributes and implementation"));
        readField(attributes,L("源码标识","Source symbol"),"source_symbol");
        readField(attributes,L("函数签名","Signature"),"signature");
        if(data.path){
          field(attributes,L("源码路径","Source path"),data.path+
            (data.line?":"+data.line:""),changed("path"),
            (otherData.path||"")+(otherData.line?":"+otherData.line:""),part.side);
          sourceLink(attributes,L("查看真实源码","View source"),data.path,data.line);
        }
        if(data.technical_description)
          readField(attributes,L("技术说明","Technical details"),"technical_description");
        if(data.facts){
          for(const [key,value] of Object.entries(data.facts))
            field(attributes,key,value,
              operation==="change"&&differs(value,otherData.facts?.[key]),
              otherData.facts?.[key],part.side);
        }
        if(hasAttributes){
          const content=element("div","atlas-reader-attributes");
          current.renderAttributes?.(content,data);
          if(changed("attributes")){
            content.classList.add("is-changed");
            field(attributes,L("属性差异","Attribute differences"),
              data.attributes,true,otherData.attributes,part.side);
          }
          attributes.appendChild(content);
        }
      }
      if(current.contextTerms?.length || current.relations?.length){
        const relations=section(scroll,L("直接关联","Direct relations"));
        if(current.contextTerms?.length){
          relations.appendChild(element("h4","atlas-reader-subheading",L("Context 定义","Context terms")));
          for(const term of current.contextTerms){
            relationLink(relations,{id:term.id,view:"context",title:term.title,
              label:L("术语","Term")});
            if(term.description)
              relations.appendChild(element("p","atlas-reader-term-definition",term.description));
          }
        }
        if(current.relations?.length){
          relations.appendChild(element("h4","atlas-reader-subheading",L("真实关系","Recorded relationships")));
          for(const record of current.relations)relationLink(relations,record);
        }
      }
      if(data.foundation || data.evidence?.length){
        const evidence=section(scroll,L("来源与依据","Sources and evidence"));
        if(data.foundation){
          readField(evidence,L("依据","Foundation"),"foundation");
          sourceLink(evidence,L("查看依据","Open foundation"),data.foundation);
        }
        for(const source of data.evidence || []){
          field(evidence,L("来源","Evidence"),source,changed("evidence"),
            otherData.evidence,part.side);
          if(typeof source==="string" && source.startsWith("https://")){
            const a=element("a","atlas-reader-source-link",L("查看来源","Open source"));
            a.href=source;a.target="_blank";a.rel="noopener noreferrer";
            evidence.appendChild(a);
          }else if(typeof source==="string"){
            const [path,line]=source.split(":");
            sourceLink(evidence,L("查看来源","Open source"),path,Number(line)||null);
          }
        }
      }
      if(operation==="remove"){
        const warning=section(scroll,L("删除影响","Removal effect"));
        warning.appendChild(element("p","atlas-reader-removal-note",
          L("本节点拟从 Atlas 中删除。本页保留其原有说明及证据，方便核对删除范围；仅批准审阅并不会删除源码。",
            "This node is proposed for removal. Its prior content and evidence remain visible for review; approving does not delete source files.")));
      }
      card.appendChild(scroll);
      holder.appendChild(card);
    }
    // Inside the card, the wheel only scrolls content (even at the end).
    function onWheel(event) {
      event.stopPropagation();
      const scroll=event.target.closest?.(".atlas-reader-scroll");
      if(scroll)return;
      const first=cardWrap?.querySelector(".atlas-reader-scroll");
      if(first){event.preventDefault();first.scrollTop+=event.deltaY;}
    }
    function open(entry) {
      if(!entry?.node)return;
      const same=!!current && current.view===entry.view &&
        current.node.id===entry.node.id && current.operation===entry.operation;
      current=entry;
      sourceRect=entry.nodeRect || sourceRect;
      if(!shell){
        shell=element("div","atlas-reader-shell"+(entry.review?" is-review":""));
        shell.setAttribute("role","region");
        shell.setAttribute("aria-label",L("Atlas 节点阅读卡","Atlas node reader"));
        const panel=element("div","atlas-reader-panel");
        shell.appendChild(panel);cardWrap=panel;
        host.appendChild(shell);
        document.body.classList.add("atlas-reader-active");
        cardWrap.addEventListener("wheel",onWheel,{passive:false});
        shell.addEventListener("keydown",event=>{
          if(event.key==="Escape"){
            event.preventDefault();event.stopPropagation();close();
          }
        });
      }else{
        shell.classList.toggle("is-review",!!entry.review);
        cardWrap.replaceChildren();
      }
      const model=presentation(entry);
      shell.dataset.operation=model.operation;
      shell.dataset.reviewState=entry.reviewState || "";
      const head=element("header","atlas-reader-head");
      const title=element("div","atlas-reader-heading");
      title.appendChild(element("div","atlas-reader-eyebrow",
        (entry.view||"Atlas").toUpperCase()+" / "+L(...names[model.operation])));
      title.appendChild(element("div","atlas-reader-current-title",
        entry.node.title || entry.node.id));
      head.appendChild(title);
      const controls=element("div","atlas-reader-controls");
      const dismiss=element("button","atlas-reader-close",L("缩回 Graph","Back to graph"));
      dismiss.type="button";
      dismiss.addEventListener("click",close);
      controls.appendChild(dismiss);
      head.appendChild(controls);cardWrap.appendChild(head);
      const pages=element("div","atlas-reader-pages"+
        (model.cards.length===2?" is-compare":""));
      for(let i=0;i<model.cards.length;i++)
        renderCard(pages,model.cards[i],model.cards[1-i],model.operation);
      cardWrap.appendChild(pages);
      if(entry.proposal){
        const meta=element("footer","atlas-reader-proposal");
        field(meta,L("修改理由","Reason"),entry.proposal.reason);
        field(meta,L("风险","Risk"),entry.proposal.risk);
        field(meta,L("验收条件","Checks"),entry.proposal.checks);
        field(meta,L("涉及文件","Files"),entry.proposal.files);
        cardWrap.appendChild(meta);
      }
      if(!same && entry.nodeRect)animateFromNode(cardWrap);
      if(!same)cardWrap.querySelector(".atlas-reader-scroll")?.scrollTo?.(0,0);
    }
    function isOpen(){return !!shell;}
    return {open,close,isOpen};
  }
  window.SAEBASE_ATLAS_NODE_READER={create,presentation,differs,diffSegments};
})();
