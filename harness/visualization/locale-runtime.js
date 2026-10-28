/* Apply only SAEBase translations; OpenResearch is a separate project. */
(function () {
  "use strict";
  const projection = window.SAEBASE_ATLAS_LOCALES;
  const graphs = window.OPENRESEARCH_ATLAS;
  if (!projection || projection.project !== "SAEBase" ||
      projection.schema !== "saebase.atlas.locale.v2" ||
      !graphs || !String(graphs.context?.project || "").startsWith("SAEBase")) {
    throw new Error("SAEBase Atlas locale and graph project identities do not agree");
  }

  // Previous static snapshots can retain locale records for removed batch
  // nodes. Only real, currently displayed Atlas IDs are part of a locale.
  for (const [view, graph] of Object.entries(graphs)) {
    const visible = new Set((graph.nodes || []).map(node => node.id));
    const records = projection.nodes?.[view] || {};
    for (const nodeId of Object.keys(records)) {
      if (!visible.has(nodeId)) delete records[nodeId];
    }
  }

  const original = new WeakMap();
  const CJK = /[\u3400-\u9fff]/;
  const skip = new Set(["code", "signature", "annotation", "assignment",
    "source", "source_symbol", "symbol_id", "path", "line", "expressions",
    "default", "file", "source_path", "function_id", "id"]);
  const labels = Object.assign({}, projection.ui, {
    "对应源码": "Associated source",
    "成员属性": "Member attributes",
    "方法": "Method", "类": "Class", "函数": "Function",
    "参数与作用": "Parameters and roles",
    "源代码表达式": "Source expression",
    "返回表达式": "Return expression",
    "实际代码主要部分": "Relevant implementation excerpt",
    "查看完整函数源码": "View complete function source",
    "代码接口": "Code interface",
    "输入 x": "Input x",
    "输出": "Output",
    "默认值": "Default value",
    "保存的字段": "Stored field",
    "真实模型": "Actual model",
    "科研术语定义与源码依据": "Research terminology and source evidence",
    "SAEBase Python 实现树": "SAEBase Python implementation tree",
    "模块职责": "Module purpose", "完整模块": "Complete module source",
    "类定义源码": "Class definition source",
    "当前任务配置": "Current task configuration",
    "配置文件来源": "Configuration file source",
    "由源码推导": "Inferred from source",
    "设备名称": "Device name", "驱动版本": "Driver version",
    "命令标识": "Command identifier", "Python 版本": "Python version",
    "网络条件": "Network status", "沙箱条件": "Sandbox environment",
    "待采集": "Not measured", "待现场采集": "Not measured on a live machine"
  });
  window.SAEBASE_ATLAS_UI = labels;

  function clone(value) {
    return value == null ? value : JSON.parse(JSON.stringify(value));
  }
  function sourceName(node) {
    return node.source_symbol || node.path || node.id;
  }
  function translateString(value, key, parent, node, en) {
    if (!CJK.test(value)) return value;
    if (key === "annotation" && value === "由源码推导") return "Inferred from source";
    if (skip.has(key)) return value;
    if (labels[value.trim()]) return labels[value.trim()];
    const methodCount = value.match(/^类中全部方法 · (\d+) 个$/);
    if (methodCount) return "Class methods (" + methodCount[1] + ")";
    const name = sourceName(node);
    if (key === "title" || key === "label") {
      const linked = parent?.symbol_id && projection.nodes?.code?.[parent.symbol_id];
      if (linked?.title) return linked.title;
      if (parent?.interface?.name) return "Interface: " + parent.interface.name;
      if (parent?.name) return parent.name;
      if (parent?.path) return parent.path;
      return en.title;
    }
    if (key === "purpose" || key === "description" || key === "summary" ||
        key === "reader_explanation" || key === "technical_description") {
      return en.description;
    }
    if (key === "value" && node.id.startsWith("term.")) {
      if (parent?.label === "为什么重要") return en.why_it_matters || en.description;
      if (parent?.label === "其他名称")
        return (en.aliases || []).join(", ") || en.title;
    }
    if (key === "meaning") {
      const field = parent?.name || "";
      const owner = node?.id === "g.sae.sar_adapter" || /(?:^|:)evaluate_sar$/.test(node?.id || "") ? "evaluate_sar" : (node?.source_symbol?.split(".").pop() || "");
      const context = projection.parameter_context?.[owner];
      if (field && context?.[field]) return context[field];
      if (projection.meaning_translations?.[value]) return projection.meaning_translations[value];
      if (field && projection.parameter_meanings[field])
        return projection.parameter_meanings[field];

      // Translate the exact structure of legacy Chinese AST descriptions.
      // These templates describe types/fields, not verified runtime behavior.
      const typedInput = value.match(/^(.+) 类型的输入，参与当前函数的计算。$/);
      if (typedInput) return "Input argument annotated with type " + typedInput[1] + ".";
      const typedReturn = value.match(/^函数返回的 (.+) 类型结果。$/);
      if (typedReturn) return "Result returned by the function, annotated as " + typedReturn[1] + ".";
      const storedField = value.match(/^(.+) 是这个组件保存的内部属性，用于其方法读取初始化状态或计算参数。$/);
      if (storedField) return "Stored instance field " + storedField[1] + " accessed by this class's methods.";
      const literalMeanings = {
        "当前过程生成的文字说明或标识。": "Textual description or identifier produced by this operation.",
        "本函数通过状态更新、训练执行或写入操作完成工作。": "This callable updates state, executes a training step, or writes an output.",
        "传入当前函数的具名数据，具体使用位置见源码。": "Named input supplied by the caller.",
        "由代码中的 return 表达式产生的结果。": "Value computed by the function's return expression.",
        "调用科研任务运行流程后的返回结果。": "Result returned by the registered research task execution.",
      };
      if (literalMeanings[value]) return literalMeanings[value];

      // Do not fabricate semantics from an annotation or a source identifier.
      // Unreviewed meanings are absent, while source signatures stay visible.
      return "";
    }
    if (key === "value") return en.summary;
    if (key === "name" && parent?.path) return parent.path;
    // Unrecognized free-form prose requires an explicit reviewed translation.
    return "";
  }
  function translateNested(value, key, parent, node, en) {
    if (typeof value === "string")
      return translateString(value, key, parent, node, en);
    if (Array.isArray(value))
      return value.map(item => translateNested(item, key, parent, node, en));
    if (value && typeof value === "object") {
      const out = {};
      for (const [childKey, item] of Object.entries(value)) {
        const localizedKey = key === "facts" ? (labels[childKey] || childKey) : childKey;
        out[localizedKey] = translateNested(item, childKey, value, node, en);
      }
      return out;
    }
    return value;
  }
  function snapshot(node) {
    if (original.has(node)) return original.get(node);
    const keys = ["title", "summary", "description", "reader_explanation",
      "technical_description", "why_it_matters", "aliases", "attributes", "facts"];
    const state = {};
    for (const key of keys) {
      if (Object.prototype.hasOwnProperty.call(node, key))
        state[key] = clone(node[key]);
    }
    original.set(node, state);
    return state;
  }
  function apply(locale) {
    if (locale !== "en" && locale !== "zh") return;
    for (const [view, graph] of Object.entries(graphs)) {
      if (!graph?.nodes || !projection.nodes[view])
        throw new Error("Missing SAEBase locale Tree " + view);
      for (const node of graph.nodes) {
        const saved = snapshot(node);
        if (locale === "zh") {
          for (const [key, value] of Object.entries(saved)) node[key] = clone(value);
          continue;
        }
        const en = projection.nodes[view][node.id];
        if (!en || !en.title || !en.summary || !en.description)
          throw new Error("Untranslated SAEBase node " + view + "/" + node.id);
        for (const key of ["title", "summary", "description"]) node[key] = en[key];
        node.reader_explanation = en.reader_explanation || en.description;
        if (saved.technical_description)
          node.technical_description = translateNested(
            saved.technical_description, "technical_description", null, node, en);
        if (Object.prototype.hasOwnProperty.call(saved, "why_it_matters"))
          node.why_it_matters = en.why_it_matters || en.description;
        if (Object.prototype.hasOwnProperty.call(saved, "aliases"))
          node.aliases = en.aliases || saved.aliases.filter(x => !CJK.test(x));
        if (Object.prototype.hasOwnProperty.call(saved, "attributes"))
          node.attributes = translateNested(saved.attributes, "attributes", null, node, en);
        if (Object.prototype.hasOwnProperty.call(saved, "facts"))
          node.facts = translateNested(saved.facts, "facts", null, node, en);
        if (CJK.test(node.title) || CJK.test(node.summary) || CJK.test(node.description)
            || /[\ufffd]/.test(node.title))
          throw new Error("Invalid English SAEBase node " + node.id);
      }
    }
    window.dispatchEvent(new Event("saebase-atlas-locale"));
  }
  window.SAEBASE_ATLAS_SET_NODE_LANGUAGE = apply;
})();
