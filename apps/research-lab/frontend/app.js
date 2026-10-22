/* OpenSAE Research Lab: dependency-free, accessible, read-only UI.
   Every displayed measurement originates from a JSON API with provenance.
   No demo text, guessed feature labels or fabricated experimental numbers.
*/
"use strict";
(function () {
  const state = { sites: [], datasets: [], assets: [], assetsLoaded: false, current: "overview", resultOffset: 0, resultTotal: 0 };
  const byId = function (id) { return document.getElementById(id); };
  const create = function (tag, cls, text) {
    const element = document.createElement(tag);
    if (cls) element.className = cls;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  };
  const append = function (parent, tag, cls, text) {
    const child = create(tag, cls, text);
    parent.appendChild(child);
    return child;
  };
  const clear = function (node) { while (node.firstChild) node.removeChild(node.firstChild); };
  const value = function (id) { return byId(id).value.trim(); };
  const safeNumber = function (number) {
    return typeof number === "number" && Number.isFinite(number)
      ? new Intl.NumberFormat("en-US", { maximumFractionDigits: 5 }).format(number)
      : "Not defined";
  };
  async function get(path) {
    const response = await fetch(path, { headers: { Accept: "application/json" }, cache: "no-store" });
    let data;
    try { data = await response.json(); } catch (error) { throw new Error("Malformed API response"); }
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Request failed");
    return data;
  }
  function empty(target, title, detail) {
    clear(target);
    const box = append(target, "div", "empty");
    append(box, "strong", "", title);
    append(box, "span", "", detail);
  }
  function errorMessage(target, error) {
    empty(target, "Unable to load verified data", error.message || String(error));
  }
  function externalLink(parent, text, url) {
    // Do not trust external URLs stored in downloadable result manifests.
    const link = append(parent, "a", "", text);
    if (typeof url !== "string" || !url.startsWith("https://github.com/OpenSAE/")) {
      link.textContent = "Source link unavailable";
      link.removeAttribute("href");
      return link;
    }
    link.href = url;
    link.rel = "noopener noreferrer";
    link.target = "_blank";
    return link;
  }
  function activate() {
    let view = window.location.hash.slice(1).split("?")[0] || "overview";
    if (!byId(view) || !byId(view).classList.contains("view")) view = "overview";
    state.current = view;
    document.querySelectorAll(".view").forEach(function (item) { item.hidden = item.id !== view; });
    document.querySelectorAll("[data-nav]").forEach(function (item) {
      const selected = item.dataset.nav === view;
      item.classList.toggle("current", selected);
      if (selected) item.setAttribute("aria-current", "page");
      else item.removeAttribute("aria-current");
    });
    document.title = (view === "overview" ? "OpenSAE" : view[0].toUpperCase() + view.slice(1) + " · OpenSAE") + " Research Lab";
    if (view === "results") loadResults(true);
    if (view === "explore") { loadAssets(); loadFeatures(); }
    if (view === "analyze") loadAnalyses();
    if (view === "datasets") loadDatasets();
    if (view === "reports") loadReports();
    window.scrollTo({ top: 0, behavior: "instant" });
  }
  async function bootstrap() {
    const status = byId("connection-text");
    try {
      const [summary, sites, datasets] = await Promise.all([
        get("/api/summary"), get("/api/sites?limit=100"), get("/api/datasets")
      ]);
      state.sites = sites.items;
      state.datasets = datasets.sources;
      byId("stat-sites").textContent = summary.sites_verified;
      byId("stat-metrics").textContent = summary.historical_metrics;
      byId("stat-datasets").textContent = summary.candidate_datasets;
      byId("stat-prompts").textContent = summary.curated_prompts;
      status.textContent = "Verified historical evidence available · Live GPU compute offline";
      document.querySelector(".status-dot").classList.add("active");
      const selector = byId("intervention-site");
      state.sites.forEach(function (site) {
        const option = append(selector, "option", "", site.model + " / L" + site.layer + " / " + site.sae);
        option.value = site.site_id;
      });
    } catch (error) {
      status.textContent = "Evidence service unavailable: " + error.message;
    }
    activate();
  }
  async function loadResults(restart) {
    const target = byId("results-table");
    if (restart) { state.resultOffset = 0; clear(target); }
    byId("results-count").textContent = "Loading verified source records…";
    const query = new URLSearchParams({
      model: value("results-model"),
      metric: value("results-metric"),
      limit: "70", offset: String(state.resultOffset)
    });
    try {
      const response = await get("/api/results?" + query.toString());
      if (restart) clear(target);
      state.resultTotal = response.total;
      response.items.forEach(function (row) {
        const tr = append(target, "tr");
        const id = append(tr, "td");
        append(id, "span", "major", row.model);
        append(id, "span", "minor", row.sae);
        append(tr, "td", "", "L" + row.layer);
        append(tr, "td", "", row.metric);
        append(tr, "td", "", row.condition || "—");
        append(tr, "td", "number", safeNumber(row.value));
        const cell = append(tr, "td");
        externalLink(cell, "Original ↗", row.evidence_url);
      });
      state.resultOffset += response.items.length;
      byId("results-more").hidden = state.resultOffset >= state.resultTotal;
      byId("results-count").textContent = response.total + " measured metric rows · historical protocol";
      if (!response.total) {
        const tr = append(target, "tr");
        append(tr, "td", "", "No matching measured records.").colSpan = 6;
      }
    } catch (error) {
      const tr = append(target, "tr");
      append(tr, "td", "", "Source unavailable: " + error.message).colSpan = 6;
      byId("results-count").textContent = "Evidence unavailable";
    }
  }
  function renderAssets() {
    const target = byId("asset-table");
    clear(target);
    const selected = value("asset-model");
    const items = state.assets.filter(function (item) {
      return !selected || item.model_key === selected;
    });
    items.forEach(function (asset) {
      const tr = append(target, "tr");
      append(tr, "td", "", asset.model_key);
      append(tr, "td", "", "L" + asset.layer + " · " + asset.hook);
      append(tr, "td", "", asset.sae_id);
      append(tr, "td", "", asset.width || "Unspecified");
      append(tr, "td", "", asset.l0_variant || "Unspecified");
      const cell = append(tr, "td");
      if (typeof asset.repo_id === "string" &&
          asset.repo_id.startsWith("google/gemma-scope-2-") &&
          /^[a-z0-9-]+$/.test(asset.repo_id.slice("google/gemma-scope-2-".length))) {
        const link = append(cell, "a", "", "Official manifest ↗");
        link.href = "https://huggingface.co/" + asset.repo_id;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
      } else {
        append(cell, "span", "", "Source unavailable");
      }
    });
    if (!items.length) {
      const tr = append(target, "tr");
      append(tr, "td", "", "No registered assets for this selection.").colSpan = 6;
    }
    byId("asset-count").textContent = items.length +
      " registered resource sites · weights are not loaded by this page";
  }
  async function loadAssets() {
    if (state.assetsLoaded) { renderAssets(); return; }
    const target = byId("asset-table");
    try {
      const response = await get("/api/sae-assets");
      if (response.mode !== "registered_metadata_only" || !Array.isArray(response.items)) {
        throw new Error("Unexpected asset registry response");
      }
      state.assets = response.items;
      state.assetsLoaded = true;
      const modelNames = Array.from(new Set(state.assets.map(function (a) { return a.model_key; }))).sort();
      const selector = byId("asset-model");
      modelNames.forEach(function (key) {
        const option = append(selector, "option", "", key);
        option.value = key;
      });
      renderAssets();
    } catch (error) {
      clear(target);
      const tr = append(target, "tr");
      append(tr, "td", "", "Asset registry unavailable: " + error.message).colSpan = 6;
    }
  }
  async function loadFeatures() {
    const target = byId("feature-list");
    empty(target, "Loading observations…", "");
    const query = new URLSearchParams({
      q: value("feature-query"), model: value("feature-model"), limit: "48"
    });
    try {
      const data = await get("/api/features?" + query.toString());
      clear(target);
      if (!data.total) {
        empty(target, "Feature measurements not yet released",
          "The index builder is ready for genuine checkpoint activations. No invented feature cards are shown before collection.");
        byId("feature-detail").hidden = true;
        return;
      }
      data.items.forEach(function (feature) {
        const button = append(target, "button", "feature-card");
        button.type = "button";
        append(button, "span", "kicker", feature.model + " · L" + feature.layer);
        append(button, "h3", "", "Feature " + feature.feature_id);
        append(button, "p", "", feature.label || "No verified natural-language explanation");
        append(button, "span", "count", feature.unique_prompt_groups + " observed prompt groups");
        button.addEventListener("click", function () { showFeature(feature); });
      });
    } catch (error) { errorMessage(target, error); }
  }
  function showFeature(feature) {
    const panel = byId("feature-detail");
    panel.hidden = false;
    clear(panel);
    append(panel, "span", "kicker", feature.site_id);
    append(panel, "h2", "", "Feature " + feature.feature_id);
    append(panel, "p", "quiet", "Exact model and SAE identities are recorded in the verified site matrix.");
    append(panel, "p", "", "Observed groups: " + feature.unique_prompt_groups + " · Nonzero observations: " + feature.observations + " · Maximum recorded activation: " + safeNumber(feature.max_activation));
    append(panel, "h3", "", "Source-approved activation examples");
    if (!feature.examples || !feature.examples.length) {
      append(panel, "p", "", "No examples have been approved for public text display.");
    } else feature.examples.forEach(function (example) {
      const sample = append(panel, "div", "example");
      append(sample, "strong", "", "Token " + example.token_position + " · " + example.token_text + " · activation " + safeNumber(example.activation));
      append(sample, "p", "longtext", example.text_excerpt || "Text not included in public record.");
    });
    panel.scrollIntoView({ behavior:"smooth",block:"nearest" });
  }
  async function loadDatasets() {
    const target = byId("dataset-list");
    empty(target, "Loading dataset registry…", "");
    try {
      const catalog = await get("/api/datasets");
      state.datasets = catalog.sources;
      clear(target);
      state.datasets.forEach(function (dataset, idx) {
        const item = append(target, "button", "dataset-card");
        item.type = "button";
        append(item, "span", "kicker", String(idx+1).padStart(2,"0") + " / 10 · " + dataset.kind);
        append(item, "h3", "", dataset.label);
        append(item, "p", "", dataset.repo + " · " + dataset.split);
        const flag = append(item, "span", "status-pill" + (dataset.status === "released" ? " ready" : ""), dataset.imported_count + " / " + dataset.target + " released");
        flag.title = dataset.status === "released" ? "Revisioned, reviewed release" : "Awaiting exact-source sampling and license review";
        item.addEventListener("click", function () { showDataset(dataset); });
      });
    } catch (error) { errorMessage(target, error); }
  }
  async function showDataset(dataset) {
    const panel = byId("dataset-detail");
    panel.hidden = false;
    clear(panel);
    append(panel, "span", "kicker", dataset.id);
    append(panel, "h2", "", dataset.label);
    append(panel, "p", "", "Original source: " + dataset.repo + " · Config: " + (dataset.config || "default") + " · Source split: " + dataset.split);
    append(panel, "p", "", "Target: 50 source-group independent prompts (20 discovery, 30 evaluation). Released: " + dataset.imported_count + ".");
    if (!dataset.imported_count) {
      append(panel, "p", "quiet", "No prompt text is displayed before source revision, sample review, and redistribution rights are verified.");
    } else {
      try {
        const data = await get("/api/prompts?dataset_id=" + encodeURIComponent(dataset.id) + "&limit=10");
        data.items.forEach(function (row) {
          const record = append(panel, "div", "example");
          append(record, "strong", "", row.sample_id + " · " + row.usage_split);
          append(record, "p", "longtext", row.prompt);
        });
        if (data.total > 10) append(panel, "p", "quiet", "Previewing 10 of " + data.total + " prompts. Full release is versioned and checksummed.");
      } catch (error) { append(panel, "p", "", error.message); }
    }
    panel.scrollIntoView({ behavior:"smooth",block:"nearest" });
  }
  async function loadAnalyses() {
    const target = byId("analysis-list");
    target.replaceChildren(create("div", "record", "Loading analysis releases…"));
    try {
      const result = await get("/api/analyses");
      clear(target);
      byId("analysis-count").textContent = result.total + " released analyses";
      if (!result.total) {
        empty(target, "No analyses released for the new corpus",
          "Original SAR, RSS and Localization research methods remain separate. New real outputs will appear here after measurements and protocol checks.");
        return;
      }
      result.items.forEach(function (record) {
        const row = append(target, "div", "record");
        append(row, "h3", "", (record.kind || "analysis").toUpperCase() + " · " + (record.site_id || "specified experiment"));
        append(row, "p", "", record.description || record.protocol_id || "View protocol and source before interpretation.");
        append(row, "span", "status-pill", record.status || "status not supplied");
      });
    } catch (error) { errorMessage(target, error); }
  }
  async function loadReports() {
    const target = byId("reports-list");
    target.replaceChildren(create("div","record","Checking approved publications…"));
    try {
      const response = await get("/api/reports");
      clear(target);
      if (!response.total) {
        empty(target, "No approved public reports yet",
          "The immutable publisher is ready; private runs or unreviewed text are never published automatically.");
        return;
      }
      response.items.forEach(function (record) {
        const item = append(target, "div", "record");
        append(item, "h3", "", record.title);
        append(item, "p", "", record.summary);
        const button = append(item, "button", "btn tiny", "Open verified report");
        button.type = "button";
        button.addEventListener("click", function () { showReport(record.report_id); });
      });
    } catch (error) { errorMessage(target, error); }
  }
  async function showReport(reportId) {
    const panel=byId("report-detail");
    panel.hidden=false;
    clear(panel);
    try {
      const report=await get("/api/reports/"+encodeURIComponent(reportId));
      append(panel,"span","kicker",report.kind+" · "+report.source_type);
      append(panel,"h2","",report.title);
      append(panel,"p","",report.summary || "Experiment result");
      append(panel,"p","quiet","Protocol "+report.protocol_id+" · Report identity "+report.report_id);
      (report.results || []).forEach(function (r) {
        const row=append(panel,"div","example");
        append(row,"strong","",r.metric+" · "+r.status);
        append(row,"p","",r.value == null ? "Not measured" : safeNumber(r.value));
      });
      const link=append(panel,"a","","Open machine-readable report ↗");
      link.href="/api/reports/"+report.report_id;
      link.target="_blank";
      link.rel="noopener noreferrer";
    } catch (error) { append(panel,"p","",error.message); }
    panel.scrollIntoView({behavior:"smooth",block:"nearest"});
  }
  async function onPreflight(event) {
    event.preventDefault();
    const result = byId("preflight-result");
    result.className = "result-message";
    result.textContent = "Checking configuration…";
    const featuresValue = value("intervention-features");
    const features = featuresValue ? featuresValue.split(",").map(function (v) { return Number(v.trim()); }) : [];
    if (features.some(function (v) { return !Number.isInteger(v); })) {
      result.textContent="Invalid feature IDs. Use comma-separated nonnegative integers.";
      result.className="result-message error"; return;
    }
    const request = {
      site_id:value("intervention-site"), text:value("intervention-text"),
      operation:value("intervention-operation"), strength:Number(value("intervention-strength")),
      feature_ids:features,max_new_tokens:Number(value("intervention-tokens"))
    };
    try {
      const response=await fetch("/api/jobs/preflight",{
        method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify(request)
      });
      const data=await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Invalid settings");
      result.textContent="Configuration valid for "+data.site_id+".\n"+data.selected_features+
        " selected features · "+data.operation+".\nNo model executed; GPU worker and authorization still required.";
      result.className="result-message ok";
    } catch (error) { result.textContent=error.message; result.className="result-message error"; }
  }
  byId("results-refresh").addEventListener("click",function(){loadResults(true);});
  byId("results-more").addEventListener("click",function(){loadResults(false);});
  byId("results-model").addEventListener("keydown",function(e){if(e.key==="Enter")loadResults(true);});
  byId("feature-search").addEventListener("click",loadFeatures);
  byId("asset-refresh").addEventListener("click",renderAssets);
  byId("asset-model").addEventListener("change",renderAssets);
  byId("feature-query").addEventListener("keydown",function(e){if(e.key==="Enter")loadFeatures();});
  byId("intervention-form").addEventListener("submit",onPreflight);
  window.addEventListener("hashchange",activate);
  bootstrap();
})();
