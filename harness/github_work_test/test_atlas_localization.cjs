"use strict";

// Pure Node regression. Source graphs and the deployed locale script share
// immutable IDs; Chinese / English rendering does not mutate graph identity.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const ROOT = path.resolve(__dirname, "../..");
const read = rel => JSON.parse(fs.readFileSync(path.join(ROOT, rel), "utf8"));
const glossary = read("doc/knowledge/glossary.json");
const vocabulary = read("doc/knowledge/atlas_terms.json");
const core = read("harness/core/project_graph.json");
const knowledge = read("doc/knowledge/context_graph.json");
const terms = [...glossary.terms, ...vocabulary.terms];
const categories = [...glossary.categories, ...vocabulary.categories];
const context = {
  nodes: [
    {
      id: "context:root",
      title: "科研术语定义与源码依据",
      summary: "这里集中解释 Atlas 中使用的科研与工程术语，并给出可核对的资料来源。",
      description: "这里集中解释 Atlas 中使用的科研与工程术语，并给出可核对的资料来源."
    },
    ...categories.map(category => ({
      id: "context:category:" + category.id,
      title: category.title, summary: category.description,
      description: category.description
    })),
    ...terms.map(term => ({
      id: term.id,
      title: term.term,
      summary: term.definition,
      description: term.definition,
      why_it_matters: term.why_it_matters,
      aliases: term.aliases,
      attributes: [
        {type: "text", label: "为什么重要", value: term.why_it_matters},
        {type: "text", label: "其他名称", value: term.aliases.join("、")}
      ]
    }))
  ]
};
const CJK = /[\u3400-\u9fff]/;
const graphs = {
  context,
  core: {nodes: core.nodes},
  knowledge: {nodes: knowledge.nodes},
  code: {nodes: [{
    id: "code:function:saebase.api.features:capture_features",
    title: "提取语言模型的 SAE 特征",
    summary: "从指定位置提取 SAE 响应。",
    source_symbol: "capture_features",
    path: "saebase/api/features.py"
  }]},
  environment: {nodes: [{
    id: "env:gpu",
    title: "设备与执行信息",
    description: "该节点描述实际测量的 GPU 资源。"
  }]}
};
const window = {OPENRESEARCH_ATLAS: graphs};
vm.runInNewContext(
  fs.readFileSync(path.join(ROOT, "harness/visualization/locale-data.js"), "utf8"),
  {window}, {filename: "locale-data.js"}
);
const locales = window.OPENRESEARCH_ATLAS_I18N;
assert(locales && locales.en, "Missing Atlas English locale overlay");

for (const term of terms) {
  assert(locales.terms[term.id], "Missing English Context entry: " + term.id);
}
for (const [view, graph] of Object.entries(graphs)) {
  for (const node of graph.nodes) {
    for (const key of ["title", "summary", "description", "why_it_matters"]) {
      const raw = node[key];
      if (!raw || !CJK.test(raw)) continue;
      const translated = locales.en[raw.trim()];
      assert.equal(typeof translated, "string",
        "Missing " + view + " " + node.id + "." + key);
      assert(translated.length > 10 || key === "title",
        "Empty translation: " + view + " " + node.id + "." + key);
      assert(!CJK.test(translated),
        "English translation contains Chinese: " + view + " " + node.id + "." + key);
    }
    for (const item of node.attributes || []) {
      for (const key of ["label", "value"]) {
        if (typeof item[key] === "string" && CJK.test(item[key])) {
          assert(locales.en[item[key].trim()],
            "Missing attribute " + node.id + "." + key);
        }
      }
    }
  }
}

assert(Object.keys(locales.docs).length >= 13, "Missing English Markdown document coverage");
for (const [source, target] of Object.entries(locales.docs)) {
  assert(fs.existsSync(path.join(ROOT, source)), "Missing source Markdown: " + source);
  assert(fs.existsSync(path.join(ROOT, target)), "Missing English Markdown: " + target);
  assert(target.endsWith(".en.md"), "English document is not a paired locale: " + target);
  const content = fs.readFileSync(path.join(ROOT, target), "utf8");
  assert(content.length >= 200, "Empty English document: " + target);
  assert(!CJK.test(content), "English document contains Chinese prose: " + target);
}
// Every Markdown source directly referenced by an Atlas authority graph has
// an English version; existing English documents need no duplicate file.
const linkedMarkdown = new Set();
function scanReferences(value) {
  if (typeof value === "string") {
    if (/^(?:[\w.-]+\/)*[\w.-]+\.md$/.test(value)) linkedMarkdown.add(value);
    return;
  }
  if (Array.isArray(value)) value.forEach(scanReferences);
  else if (value && typeof value === "object") Object.values(value).forEach(scanReferences);
}
for (const authority of [core, knowledge, glossary, vocabulary]) scanReferences(authority);
for (const relative of linkedMarkdown) {
  const sourceFile = path.join(ROOT, relative);
  assert(fs.existsSync(sourceFile), "Broken Atlas Markdown reference: " + relative);
  // Root README.md is already English; its language selector legitimately
  // contains the self-identifying word "中文".
  if (relative === "README.md") {
    assert(fs.readFileSync(sourceFile, "utf8").includes("Research Infrastructure"));
    continue;
  }
  if (CJK.test(fs.readFileSync(sourceFile, "utf8"))) {
    assert(locales.docs[relative],
      "Linked Chinese Markdown has no English variant: " + relative);
  }
}
for (const view of ["core", "knowledge", "code", "environment", "context"]) {
  const html = fs.readFileSync(
    path.join(ROOT, "harness/visualization", view + ".html"), "utf8"
  );
  assert(html.includes('src="./locale-data.js'),
    "Tree has no locale overlay: " + view);
  assert(html.indexOf('src="./locale-data.js') < html.indexOf('src="./i18n.js'),
    "Incorrect language-script order: " + view);
}
console.log("Atlas English/Chinese source mapping and Markdown documents: OK");
