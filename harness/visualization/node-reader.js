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
  function element(tag,className,text) {
    const el = document.createElement(tag);
    if(className)el.className=className;
    if(text !== undefined && text !== null)el.textContent=String(text);
    return el;
  }
  function create({host}) {
    let shell=null,cardWrap=null,current=null,sourceRect=null,threshold=0;
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
      shell=null; cardWrap=null;current=null;threshold=0;
      animateFromNode(panel,true);
      old.remove();
    }
    function section(root,title) {
      const section=element("section","atlas-reader-section");
      section.appendChild(element("h3","atlas-reader-section-title",title));
      root.appendChild(section);
      return section;
    }
    function field(root,label,value,changed=false) {
      if(value === undefined || value === null || value === "")return;
      if(Array.isArray(value) && !value.length)return;
      const row=element("div","atlas-reader-field"+(changed?" is-changed":""));
      row.appendChild(element("div","atlas-reader-field-label",label));
      row.appendChild(element("div","atlas-reader-field-value",
        Array.isArray(value)?value.join(" · "):typeof value==="object"?
          JSON.stringify(value,null,2):value));
      root.appendChild(row);
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
      heading.appendChild(element("h2","atlas-reader-card-title",data.title || current.node.title));
      card.appendChild(heading);
      const scroll=element("div","atlas-reader-scroll");
      scroll.tabIndex=0;
      const changed=key=>operation==="change" && differs(data[key],otherData[key]);
      const explanatory=section(scroll,L("节点说明","Explanation"));
      field(explanatory,L("摘要","Summary"),data.summary,changed("summary"));
      field(explanatory,L("完整说明","Description"),data.description,changed("description"));
      const extra=data.reader_explanation;
      if(extra && extra!==data.description && extra!==data.summary)
        field(explanatory,L("详细解释","Detailed explanation"),extra,changed("reader_explanation"));
      field(explanatory,L("研究意义","Why this matters"),data.why_it_matters,changed("why_it_matters"));
      field(explanatory,L("下一步","Next step"),data.next,changed("next"));
      field(explanatory,L("别名","Aliases"),data.aliases,changed("aliases"));
      const hasAttributes=Array.isArray(data.attributes) && data.attributes.length;
      if(data.signature || data.source_symbol || data.path || hasAttributes ||
         data.facts || data.technical_description){
        const attributes=section(scroll,L("属性与实现","Attributes and implementation"));
        field(attributes,L("源码标识","Source symbol"),data.source_symbol,changed("source_symbol"));
        field(attributes,L("函数签名","Signature"),data.signature,changed("signature"));
        if(data.path){
          field(attributes,L("源码路径","Source path"),data.path+
            (data.line?":"+data.line:""),changed("path"));
          sourceLink(attributes,L("查看真实源码","View source"),data.path,data.line);
        }
        if(data.technical_description)
          field(attributes,L("技术说明","Technical details"),
            data.technical_description,changed("technical_description"));
        if(data.facts){
          for(const [key,value] of Object.entries(data.facts))
            field(attributes,key,value,changed("facts"));
        }
        if(hasAttributes){
          const content=element("div","atlas-reader-attributes");
          current.renderAttributes?.(content,data);
          if(changed("attributes"))content.classList.add("is-changed");
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
          field(evidence,L("依据","Foundation"),data.foundation,changed("foundation"));
          sourceLink(evidence,L("查看依据","Open foundation"),data.foundation);
        }
        for(const source of data.evidence || []){
          field(evidence,L("来源","Evidence"),source,changed("evidence"));
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
    function onWheel(event) {
      if(event.ctrlKey || event.deltaY<=0){threshold=0;return;}
      const scroller=event.target.closest?.(".atlas-reader-scroll");
      if(scroller){
        const end=scroller.scrollTop+scroller.clientHeight>=scroller.scrollHeight-3;
        if(!end){threshold=0;return;}
        threshold+=event.deltaY;
        if(threshold<140)return;
      }
      event.preventDefault();
      event.stopPropagation();
      close();
    }
    function open(entry) {
      if(!entry?.node)return;
      const same=!!current && current.view===entry.view &&
        current.node.id===entry.node.id && current.operation===entry.operation;
      current=entry;
      sourceRect=entry.nodeRect || sourceRect;
      threshold=0;
      if(!shell){
        shell=element("div","atlas-reader-shell"+(entry.review?" is-review":""));
        shell.setAttribute("role","region");
        shell.setAttribute("aria-label",L("Atlas 节点阅读卡","Atlas node reader"));
        const panel=element("div","atlas-reader-panel");
        shell.appendChild(panel);cardWrap=panel;
        host.appendChild(shell);
        shell.addEventListener("wheel",onWheel,{passive:false});
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
      const head=element("header","atlas-reader-head");
      const title=element("div","atlas-reader-heading");
      title.appendChild(element("div","atlas-reader-eyebrow",
        (entry.view||"Atlas").toUpperCase()+" / "+L(...names[model.operation])));
      title.appendChild(element("div","atlas-reader-current-title",
        entry.node.title || entry.node.id));
      head.appendChild(title);
      const controls=element("div","atlas-reader-controls");
      controls.appendChild(element("span","atlas-reader-hint",
        L("正文滚动阅读 · 向外滚轮缩回 Graph","Scroll to read · wheel out to return to graph")));
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
  window.SAEBASE_ATLAS_NODE_READER={create,presentation,differs};
})();
