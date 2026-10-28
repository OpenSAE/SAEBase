"use strict";

// Project-isolated SAEBase acceptance: all five trees, reviewed Context,
// real source IDs, referenced English documents, and UTF-8-safe titles.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const ROOT = path.resolve(__dirname, "../..");
const read = file => fs.readFileSync(path.join(ROOT, file), "utf8");
const load = file => JSON.parse(read(file));
const source = read("harness/visualization/data.js");
const prefix = "window.OPENRESEARCH_ATLAS = ";
assert(source.includes(prefix), "Missing SAEBase-generated graph");
const graph = JSON.parse(source.split(prefix)[1].trim().replace(/;\s*$/, ""));
const window = {};
vm.runInNewContext(read("harness/visualization/locale-data.js"), {window});
const locales = window.SAEBASE_ATLAS_LOCALES;
assert(locales && locales.project === "SAEBase", "Foreign or missing project locale");
assert.equal(locales.schema, "saebase.atlas.locale.v2");
assert(graph.context.project.startsWith("SAEBase"), "Foreign Context tree");
assert.deepEqual(
  Object.keys(locales.nodes).sort(),
  ["core", "knowledge", "code", "environment", "context"].sort()
);

const cjk = /[\u3400-\u9fff]/;
const bad = /[\ufffd]/;
let translated = 0;
for (const [view, tree] of Object.entries(graph)) {
  const ids = new Set();
  const mapped = locales.nodes[view];
  assert.equal(Object.keys(mapped).length, tree.nodes.length,
    "Locale data must cover every node of " + view);
  for (const node of tree.nodes) {
    assert(!ids.has(node.id), "Repeated source ID " + node.id);
    ids.add(node.id);
    const fields = mapped[node.id];
    assert(fields, "Missing English translation " + view + ":" + node.id);
    for (const field of ["title", "summary", "description"]) {
      const value = fields[field];
      assert(typeof value === "string" && value.trim(), node.id + "." + field);
      assert(!cjk.test(value) && !bad.test(value), "Invalid UTF-8 title or prose: " + node.id + "." + field);
    }
    if (view === "code" && ["module", "function", "class", "method"].includes(node.kind)) {
      assert(!fields.title.includes(" · "), "Code title must be a readable action, not concatenated tokens: " + node.id);
      assert(!/ in component$|^Source detail for /.test(fields.title),
        "Code title must describe the real implementation: " + node.id);
      assert(node.path && fields.description.includes(node.path),
        "Code explanation must cite actual source: " + node.id);
    }
    translated++;
  }
  for (const edge of tree.edges) {
    assert(ids.has(edge.source) && ids.has(edge.target), "Broken " + view + " edge");
  }
}

// Source-backed sibling nodes must have distinct natural-language names.
// Equal titles in distant packages are valid, but sibling headings cannot
// force users to inspect hashed IDs to tell research operations apart.
const byParent = new Map();
for (const edge of graph.code.edges) {
  if (edge.type !== "contains") continue;
  if (!byParent.has(edge.source)) byParent.set(edge.source, []);
  byParent.get(edge.source).push(edge.target);
}
for (const [parent, childIds] of byParent) {
  const titles = new Map();
  for (const id of childIds) {
    const title = locales.nodes.code[id]?.title;
    if (!title) continue;
    if (!titles.has(title)) titles.set(title, []);
    titles.get(title).push(id);
  }
  for (const [title, ids] of titles) {
    assert.equal(ids.length, 1,
      "Ambiguous sibling Code titles under " + parent + ": " + title +
      " [" + ids.join(", ") + "]");
  }
}

const glossary = load("doc/knowledge/glossary.json");
const technical = load("doc/knowledge/atlas_terms.json");
const scientific = load("doc/knowledge/i18n/context.en.json");
for (const term of [...glossary.terms, ...technical.terms]) {
  const en = scientific.nodes[term.id];
  assert(en, "Untranslated authoritative scientific term: " + term.id);
  assert(en.why_it_matters && !cjk.test(en.why_it_matters), "Missing Context rationale: " + term.id);
  assert(locales.nodes.context[term.id].title === en.title, "Translation detached from authority: " + term.id);
}
// Derive the required document set from the actual rendered node graph.
// A hand-picked allowlist cannot silently omit a newly linked Chinese page.
const referencedDocs = new Set();
function collectDocuments(value) {
  if (typeof value === "string") {
    const candidate = value.split("#")[0].split("?")[0];
    if (/^(?:[A-Za-z0-9_.-]+\/)*[A-Za-z0-9_.-]+\.md$/.test(candidate))
      referencedDocs.add(candidate);
  } else if (Array.isArray(value)) {
    for (const item of value) collectDocuments(item);
  } else if (value && typeof value === "object") {
    for (const item of Object.values(value)) collectDocuments(item);
  }
}
for (const tree of Object.values(graph)) for (const node of tree.nodes)
  collectDocuments(node);
for (const original of referencedDocs) {
  assert(fs.existsSync(path.join(ROOT, original)),
    "Atlas references a missing Markdown document: " + original);
  // The root README is authored in English and retains a self-identifying
  // Chinese language selector. It is already the required English version.
  if (original === "README.md" || original.endsWith(".en.md")) continue;
  const prose = read(original);
  if (cjk.test(prose)) {
    const english = locales.docs[original];
    assert(english && fs.existsSync(path.join(ROOT, english)),
      "Tree-linked Chinese Markdown lacks English counterpart: " + original);
  }
}

for (const [original, english] of Object.entries(locales.docs)) {
  assert(fs.existsSync(path.join(ROOT, original)), "Missing Markdown authority: " + original);
  assert(fs.existsSync(path.join(ROOT, english)), "Missing English document: " + english);
  assert(english.endsWith(".en.md"), "English document must be distinct: " + english);
  assert(!cjk.test(read(english)), "Untranslated English Markdown prose: " + english);
}
for (const view of Object.keys(graph)) {
  const html = read("harness/visualization/" + view + ".html");
  assert(html.includes("SAEBase Atlas"), "Wrong project page branding: " + view);
  for (const script of ["data.js", "locale-data.js", "locale-runtime.js", "i18n.js"]) {
    assert(html.includes("./" + script), "Missing runtime " + script + " in " + view);
  }
  assert(html.indexOf("locale-data.js") < html.indexOf("locale-runtime.js"),
    "Locale must load before application in " + view);
  assert(html.indexOf("locale-runtime.js") < html.indexOf("i18n.js"),
    "Locale runtime must load before switch in " + view);
  assert(!html.includes("opensae/openresearch/atlas"), "SAEBase page points at OpenResearch");
}
assert(!read("harness/visualization/locale-runtime.js").includes("eval("));

 // Meaningful bilingual coverage must include nested interface semantics.
 // A nonempty English title alone is not an adequate translation gate.
 const runtime = read("harness/visualization/locale-runtime.js");
 const switcher = read("harness/visualization/i18n.js");
 assert(!switcher.includes('return "English translation unavailable"'),
   "Never render missing translations as reader-facing prose");
 assert(!runtime.includes("its behavior is defined in the source"),
   "Unreviewed parameters must not receive invented English explanations");
 assert(!runtime.includes("Source-linked " + "key.replace"),
   "Unreviewed nested fields must not receive generic English explanations");
 const sar = locales.parameter_context?.evaluate_sar;
 for (const key of ["p", "readout_position", "verify_replay", "identity"]) {
   assert(typeof sar?.[key] === "string" && sar[key].length > 35,
     "Missing source-reviewed SAR parameter meaning: " + key);
 }
 for (const entry of Object.values(locales.nodes.code)) {
   assert(!/^Browse \d+ source-backed child interfaces\.$/.test(entry.summary),
     "Code group descriptions must communicate purpose, not child counts");
 }

console.log("SAEBase-only bilingual Atlas accepted:", translated, "nodes, 5 trees");
