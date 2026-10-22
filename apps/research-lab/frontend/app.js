/* OpenSAE Research Lab: dependency-free, accessible, read-only UI.
   Every displayed measurement originates from a JSON API with provenance.
   No demo text, guessed feature labels or fabricated experimental numbers.
*/
"use strict";
(function () {
  const state = { sites: [], datasets: [], assets: [], assetsLoaded: false, tracesLoaded: false, activeTrace: null, searchMode: "explanation", current: "overview", resultOffset: 0, resultTotal: 0,
    circuitsLoaded: false, circuitData: null, circuitSelected: null, circuitPins: new Set(), circuitNotes: {}, circuitGroups: [], graphScale: 1 };
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
    if (view === "explore") {
      loadAssets(); loadFeatures(); loadRecordedTraces();
      const params = new URLSearchParams(window.location.hash.split("?")[1] || "");
      const site = params.get("site");
      const index = params.get("feature");
      if (site && index && /^\\d+$/.test(index)) loadFeature(site, Number(index));
    }
    if (view === "analyze") loadAnalyses();
    if (view === "datasets") loadDatasets();
    if (view === "reports") loadReports();
    if (view === "benchmarks") loadBenchmarks();
    if (view === "training") loadTraining();
    if (view === "circuits") loadCircuitCatalog();
    if (view === "jobs") loadComputeRoutes();
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
      append(tr, "td", "", (asset.layers && asset.layers.length > 1 ? "L" + asset.layers.join(",") :
        asset.layer < 0 ? "Cross-layer" : "L" + asset.layer) + " · " + asset.hook);
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
  async function postExplore(request) {
    const response = await fetch("/api/explore/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      cache: "no-store", body: JSON.stringify(request)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Search failed");
    return data;
  }
  function setSearchMode(mode) {
    state.searchMode = mode;
    document.querySelectorAll("[data-explore-mode]").forEach(function (tab) {
      const selected = tab.dataset.exploreMode === mode;
      tab.classList.toggle("selected", selected);
      tab.setAttribute("aria-pressed", String(selected));
    });
    ["explanation", "inference", "token"].forEach(function (key) {
      byId("explore-mode-" + key).hidden = key !== mode;
    });
    clear(byId("explore-search-results"));
  }
  function featureLink(parent, title, siteId, featureId) {
    const link = append(parent, "a", "", title);
    link.href = "#explore?site=" + encodeURIComponent(siteId) + "&feature=" + featureId;
    link.addEventListener("click", function () {
      if (window.location.hash === link.getAttribute("href")) loadFeature(siteId, featureId);
    });
    return link;
  }
  function showRankedFeatures(parent, items, siteId) {
    if (!items.length) {
      empty(parent, "No observed active features", "This recorded token has no nonzero SAE codes.");
      return;
    }
    const list = append(parent, "div", "ranked-features");
    items.forEach(function (row, rank) {
      const card = append(list, "div", "ranked-row");
      append(card, "span", "token-rank", String(rank + 1));
      featureLink(card, "Feature " + row.feature_id, siteId, row.feature_id);
      append(card, "span", "number", safeNumber(row.activation));
      append(card, "span", "quiet", "Token " + row.position);
    });
  }
  async function searchExplanations() {
    const target = byId("explore-search-results");
    empty(target, "Searching published explanations…", "");
    try {
      const response = await postExplore({
        mode: "explanation", query: value("explanation-query"), limit: 40
      });
      clear(target);
      if (!response.total) {
        empty(target, "No reviewed explanations match", "There are no published descriptions for this query.");
        return;
      }
      const list = append(target, "div", "explanation-results");
      response.items.forEach(function (entry) {
        const item = append(list, "div", "record");
        featureLink(item, entry.site_id + " · Feature " + entry.feature_id, entry.site_id, entry.feature_id);
        append(item, "p", "", entry.description);
        append(item, "span", "quiet", entry.origin + " · " + entry.status +
          " · " + entry.author_or_model + (entry.heldout_score == null ?
            "" : " · Held-out score " + safeNumber(entry.heldout_score)));
      });
    } catch (error) { errorMessage(target, error); }
  }
  async function searchRecordedInference() {
    const target = byId("explore-search-results");
    empty(target, "Looking for an exact recorded input…", "");
    try {
      const response = await postExplore({
        mode: "inference", query: value("inference-prompt"), limit: 10
      });
      clear(target);
      response.items.forEach(function (record) {
        const part = append(target, "div", "record");
        append(part, "h3", "", record.site_id + " · recorded computation");
        showRankedFeatures(part, record.max_features, record.site_id);
      });
    } catch (error) { errorMessage(target, error); }
  }
  async function loadRecordedTraces() {
    if (state.tracesLoaded) return;
    const strip = byId("token-strip");
    empty(strip, "Checking saved prompts…", "");
    try {
      const collection = await get("/api/explore/traces");
      const selector = byId("trace-select");
      collection.items.forEach(function (trace) {
        const option = append(selector, "option", "",
          trace.site_id + " · " + trace.sample_id + " · " + trace.token_count + " tokens");
        option.value = trace.trace_id;
      });
      state.tracesLoaded = true;
      if (!collection.items.length) {
        empty(strip, "No public recorded token traces", "A local GPU or CPU worker can export real traces for review before publication.");
      } else clear(strip);
    } catch (error) { errorMessage(strip, error); }
  }
  async function selectRecordedTrace() {
    const id = value("trace-select");
    const strip = byId("token-strip");
    if (!id) { empty(strip, "Select a recorded prompt", "No computation is started."); return; }
    try {
      const trace = await get("/api/explore/traces/" + encodeURIComponent(id));
      state.activeTrace = trace;
      clear(strip);
      trace.tokens.forEach(function (token) {
        const button = append(strip, "button", "token-button", token.text || "(empty token)");
        button.type = "button";
        button.title = "Token " + token.position + " · id " + token.token_id;
        button.addEventListener("click", function () { showTokenFeatures(trace, token.position); });
      });
      showTokenFeatures(trace, trace.tokens.length - 1);
    } catch (error) { errorMessage(strip, error); }
  }
  async function showTokenFeatures(trace, position) {
    const target = byId("explore-search-results");
    clear(target);
    try {
      const response = await postExplore({
        mode: "token", trace_id: trace.trace_id, position: position, limit: 30
      });
      const token = trace.tokens[position];
      append(target, "h3", "", "Token " + position + " · " + (token.text || "(empty token)"));
      append(target, "p", "quiet", "Recorded nonzero SAE codes · " + trace.site_id +
        " · " + trace.prompt_group_id + " · not live inference");
      showRankedFeatures(target, response.items, response.site_id);
      byId("token-strip").querySelectorAll("button").forEach(function (button, i) {
        button.classList.toggle("selected", i === position);
      });
    } catch (error) { errorMessage(target, error); }
  }
  async function loadFeatures() {
    const target = byId("feature-list");
    empty(target, "Loading published feature cards…", "");
    const query = new URLSearchParams({
      q: value("feature-query"), model: value("feature-model"), limit: "48"
    });
    try {
      const data = await get("/api/features?" + query.toString());
      clear(target);
      if (!data.total) {
        empty(target, "Feature measurements not yet published",
          "The SQLite/index pipeline accepts real verified observations. No synthetic card is displayed as a measurement.");
        return;
      }
      data.items.forEach(function (feature) {
        const card = append(target, "div", "feature-card");
        append(card, "span", "kicker", feature.model + " · L" + feature.layer);
        const title = append(card, "h3");
        featureLink(title, "Feature " + feature.feature_id, feature.site_id, feature.feature_id);
        append(card, "p", "", feature.label || "No approved explanation");
        append(card, "span", "count", feature.unique_prompt_groups + " recorded prompt groups");
      });
    } catch (error) { errorMessage(target, error); }
  }
  async function loadFeature(siteId, featureId) {
    const panel = byId("feature-detail");
    panel.hidden = false;
    clear(panel);
    append(panel, "p", "quiet", "Loading source-bound feature…");
    try {
      const data = await get("/api/explore/features/" + encodeURIComponent(siteId) + "/" + featureId);
      clear(panel);
      append(panel, "span", "kicker", data.site.model + " · L" + data.site.layer + " · " + data.site.sae);
      append(panel, "h2", "", "Feature " + data.feature_id);
      const nav = append(panel, "div", "feature-navigation");
      if (data.previous_feature_id != null) featureLink(nav, "← Previous", siteId, data.previous_feature_id);
      if (data.next_feature_id != null) featureLink(nav, "Next →", siteId, data.next_feature_id);
      const intervene = append(nav, "button", "btn tiny", "Design an intervention ↗");
      intervene.type = "button";
      intervene.addEventListener("click", function () {
        const selector = byId("intervention-site");
        if ([...selector.options].some(function (option) { return option.value === siteId; })) {
          selector.value = siteId;
        }
        byId("intervention-features").value = String(featureId);
        byId("intervention-operation").value = "ablate";
        window.location.hash = "#intervene";
      });
      const download = append(nav, "a", "", "Export offline HTML ↗");
      download.href = "/api/explore/features/" + encodeURIComponent(siteId) + "/" +
        featureId + "/export.html";
      download.setAttribute("download", "opensae-feature-" + featureId + ".html");
      append(panel, "p", "quiet", "Exact site " + data.site.site_id +
        " · " + data.site.hook + " · evidence is recorded, not live");
      append(panel, "p", "", "Observed feature occurrences: " + data.observations +
        " · independent prompt groups: " + data.unique_prompt_groups +
        " · highest recorded activation: " + safeNumber(data.max_activation));
      const labels = append(panel, "div", "feature-explanations");
      append(labels, "h3", "", "Published interpretations");
      if (!(data.explanations || []).length) {
        append(labels, "p", "quiet", "No approved explanation. Feature activations are observations, not semantic labels.");
      } else (data.explanations || []).forEach(function (entry) {
        const item = append(labels, "div", "example");
        append(item, "p", "", entry.description);
        append(item, "span", "quiet", entry.origin + " · " + entry.method +
          " · " + entry.status + " · " + entry.author_or_model);
        if (entry.heldout_score != null)
          append(item, "p", "quiet", "Held-out activation prediction score " +
            safeNumber(entry.heldout_score) + " · not ground-truth semantic accuracy");
      });
      append(panel, "h3", "", "Highest reviewed activation contexts");
      const examples = data.examples || [];
      if (!examples.length) append(panel, "p", "quiet", "No text approved for public display.");
      examples.forEach(function (ex) {
        const item = append(panel, "div", "example");
        append(item, "strong", "", ex.token_text + " · activation " + safeNumber(ex.activation));
        append(item, "p", "longtext", ex.text_excerpt || "Original context not publicly available.");
      });
      if (examples.length) append(panel, "p", "quiet",
        "The displayed values are selected reviewed examples, not a corpus-wide activation histogram or density estimate.");
      const corr = append(panel, "div", "feature-correlations");
      append(corr, "h3", "", "Correlations across recorded prompt groups");
      append(corr, "p", "quiet", "Loading the actual complete-code group comparison…");
      try {
        const scores = await get("/api/explore/features/" + encodeURIComponent(siteId) +
          "/" + featureId + "/correlations");
        clear(corr);
        append(corr, "h3", "", "Correlations across recorded prompt groups");
        append(corr, "p", "quiet", scores.note || "Complete sparse-code source");
        (scores.items || []).forEach(function (value) {
          const row = append(corr, "div", "correlation-row");
          featureLink(row, "Feature " + value.feature_id, siteId, value.feature_id);
          append(row, "span", "number", "Pearson r = " + safeNumber(value.pearson_r));
        });
      } catch (error) { append(corr, "p", "quiet", "Correlation unavailable: " + error.message); }
    } catch (error) { errorMessage(panel, error); }
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
  function downloadJson(filename, data) {
    const bytes = new Blob([JSON.stringify(data, null, 2) + "\n"], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(bytes);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
  }

  async function loadBenchmarks() {
    const target = byId("benchmark-list");
    empty(target, "Loading native comparisons…", "");
    try {
      const res = await get("/api/wp12/benchmarks");
      byId("benchmark-count").textContent = res.total + " reviewed native comparisons";
      clear(target);
      if (!res.total) { empty(target, "No native benchmark comparisons published",
        "Run native SAEBench in an approved worker; no benchmark result is fabricated here."); return; }
      res.items.forEach(function (item) {
        const row = append(target, "div", "record");
        append(row, "strong", "", item.metric || "Native metric");
        append(row, "p", "quiet", item.points + " matched SAE points · exact source review");
        const button = append(row, "button", "btn tiny", "View comparison");
        button.type = "button";
        button.addEventListener("click", function () { showBenchmark(item.id); });
      });
    } catch (error) { errorMessage(target, error); }
  }
  async function showBenchmark(id) {
    const panel = byId("benchmark-detail");
    panel.hidden = false;
    empty(panel, "Loading source results…", "");
    try {
      const record = await get("/api/wp12/benchmarks/" + id);
      if (record.comparable !== true || !Array.isArray(record.points) || record.points.length < 2) {
        throw new Error("Source comparison is not independently matched");
      }
      clear(panel);
      append(panel, "h2", "", record.metric);
      append(panel, "p", "quiet", "Matched source: " + record.scope.hook +
        " · dataset SHA256 " + record.scope.dataset_sha256.slice(0, 16) + "…");
      const table = append(panel, "table");
      const head = append(table, "thead");
      const tr = append(head, "tr");
      ["SAE", "L0", "Measured metric"].forEach(function (name) { append(tr, "th", "", name); });
      const tbody = append(table, "tbody");
      record.points.forEach(function (p) {
        const row = append(tbody, "tr");
        append(row, "td", "", p.sae);
        append(row, "td", "", safeNumber(p.l0));
        append(row, "td", "", safeNumber(p.metric_value));
      });
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 640 280");
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label", "Native SAE measurement by original L0 sparsity");
      svg.classList.add("native-metric-chart");
      panel.appendChild(svg);
      const xs = record.points.map(function (p) { return p.l0; });
      const ys = record.points.map(function (p) { return p.metric_value; });
      const xmin = Math.min(...xs), xmax = Math.max(...xs);
      const ymin = Math.min(...ys), ymax = Math.max(...ys);
      const posX = function (v) { return 45 + (v - xmin) / (xmax - xmin || 1) * 535; };
      const posY = function (v) { return 220 - (v - ymin) / (ymax - ymin || 1) * 180; };
      function el(kind, attrs, title) {
        const node = document.createElementNS("http://www.w3.org/2000/svg", kind);
        Object.keys(attrs).forEach(function (key) { node.setAttribute(key, String(attrs[key])); });
        if (title) {
          const t = document.createElementNS("http://www.w3.org/2000/svg", "title");
          t.textContent = title;
          node.appendChild(t);
        }
        svg.appendChild(node);
      }
      el("path", {d:"M45 32V220H580", stroke:"#7d8ba0",fill:"none"});
      record.points.forEach(function (p) {
        el("circle", {cx:posX(p.l0),cy:posY(p.metric_value),r:6,
          fill:"#2563aa"}, p.sae + ": L0 " + p.l0 + ", " + p.metric_value);
      });
      append(panel, "p", "quiet", record.uncertainty ||
        "Uncertainty cannot be inferred without source-level independent samples.");
    } catch (error) { errorMessage(panel, error); }
  }

  async function loadTraining() {
    const target = byId("training-list");
    empty(target, "Loading training records…", "");
    try {
      const res = await get("/api/wp12/training");
      if (!res.total) {
        empty(target, "No native training runs released",
          "SAELens and Sparsify workers can train privately; checkpoint and source approval come before publication.");
        return;
      }
      clear(target);
      res.items.forEach(function (item) {
        const row = append(target, "div", "record");
        append(row, "strong", "", item.backend || "Native trainer");
        append(row, "p", "quiet", item.id.slice(0, 20) + "… · approved checkpoint source");
      });
    } catch (error) { errorMessage(target, error); }
  }
  function onTrainingExport(event) {
    event.preventDefault();
    const revision = /^[0-9a-f]{40}$/;
    if (!revision.test(value("training-model-revision")) ||
        !revision.test(value("training-dataset-revision"))) {
      return window.alert("Both source revisions must be exact 40-character lowercase SHAs.");
    }
    const steps = Number(value("training-steps"));
    const width = Number(value("training-width"));
    const latents = Number(value("training-latents"));
    const k = Number(value("training-k"));
    if (!Number.isInteger(steps) || steps < 1 || steps > 200 ||
        !Number.isInteger(width) || width < 1 ||
        !Number.isInteger(latents) || latents < 1 ||
        !Number.isInteger(k) || k < 1 || k > latents) {
      return window.alert("Invalid width, sparsity or C1 training budget.");
    }
    const manifest = {
      schema: "opensae.wp12.p5.training_plan_draft.v1",
      status: "user_config_only_not_executed",
      backend: value("training-backend"),
      model_path: value("training-model"),
      dataset_path: value("training-dataset"),
      model_revision: value("training-model-revision"),
      dataset_revision: value("training-dataset-revision"),
      hookpoint: value("training-hook"),
      d_in: width, d_sae: latents, k: k, steps: steps,
      architecture: "topk",
      max_examples: Math.min(10000, steps * 16),
      batch_size: 16, ctx_len: 128,
      run_name: "wp12-local-pilot", device: "cuda",
      distributed_gpus: 1, resume_checkpoint: null
    };
    if (!manifest.model_path || !manifest.dataset_path || !manifest.hookpoint) {
      return window.alert("Model, dataset and hookpoint are required.");
    }
    downloadJson("opensae-wp12-training-plan.json", manifest);
  }

  function svgNode(tag, attrs) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs || {}).forEach(function (entry) {
      node.setAttribute(entry[0], String(entry[1]));
    });
    return node;
  }
  function checkGraph(graph) {
    if (!graph || graph.schema !== "opensae.wp12.p6.graph.v1" ||
        !Array.isArray(graph.nodes) || !Array.isArray(graph.edges) ||
        !graph.nodes.length || graph.nodes.length > 250 || graph.edges.length > 750 ||
        !graph.provenance || !/^[a-f0-9]{64}$/.test(graph.provenance.source_sha256)) {
      throw new Error("Expected a bounded, source-identified attribution graph");
    }
    const ids = new Set();
    graph.nodes.forEach(function (n) {
      if (typeof n.id !== "string" || !["feature", "error", "token", "logit"].includes(n.kind) ||
          n.id.length > 120 || ids.has(n.id)) throw new Error("Invalid or duplicate graph node");
      ids.add(n.id);
    });
    if (!graph.nodes.some(function (n) {return n.kind === "error";}) ||
        !graph.nodes.some(function (n) {return n.kind === "logit";})) {
      throw new Error("Native error and logit nodes required");
    }
    graph.edges.forEach(function (e) {
      if (!ids.has(e.source) || !ids.has(e.target) ||
          typeof e.weight !== "number" || !Number.isFinite(e.weight)) {
        throw new Error("Graph edge identity or attribution is invalid");
      }
    });
    return graph;
  }
  async function loadCircuitCatalog() {
    if (state.circuitsLoaded) return;
    try {
      const res = await get("/api/wp12/circuits");
      const selector = byId("circuit-select");
      res.items.forEach(function (item) {
        const option = append(selector, "option", "",
          item.nodes + " nodes · " + item.edges + " edges · " + item.id.slice(0, 8));
        option.value = item.id;
      });
      state.circuitsLoaded = true;
      byId("circuit-status").textContent = res.total ?
        res.total + " reviewed native graphs available" :
        "No native graphs approved for publication. You can inspect a typed local JSON file.";
    } catch (error) { byId("circuit-status").textContent = error.message; }
  }
  async function openCircuit() {
    const file = byId("circuit-upload").files[0];
    try {
      const graph = file ?
        JSON.parse(await file.text()) :
        await get("/api/wp12/circuits/" + encodeURIComponent(value("circuit-select")));
      state.circuitData = checkGraph(graph);
      state.circuitNotes = graph.annotations && typeof graph.annotations === "object" ?
        graph.annotations.notes || {} : {};
      state.circuitPins = new Set(graph.annotations?.pinned || []);
      state.circuitGroups = graph.annotations?.groups || [];
      state.circuitSelected = null;
      state.graphScale = 1;
      byId("circuit-status").textContent = graph.nodes.length + " actual source nodes · " +
        graph.edges.length + " signed edges · " + graph.provenance.source_sha256.slice(0, 14) +
        "… · annotations are local only";
      renderCircuit();
    } catch (error) { byId("circuit-status").textContent = "Cannot open circuit: " + error.message; }
  }
  function renderCircuit() {
    const graph = state.circuitData, svg = byId("circuit-graph");
    svg.replaceChildren();
    if (!graph) return;
    const layers = [...new Set(graph.nodes.map(function (n) {
      return n.kind === "token" ? "input" : n.kind === "logit" ? "output" :
        "layer" + (n.layer == null ? "?" : n.layer);
    }))].sort(function (a,b) {
      if (a === "input") return -1; if (b === "input") return 1;
      if (a === "output") return 1; if (b === "output") return -1;
      return Number(a.slice(5)) - Number(b.slice(5));
    });
    const groups = new Map(layers.map(function (l) { return [l, []]; }));
    graph.nodes.forEach(function (n) {
      const layer = n.kind === "token" ? "input" :
        n.kind === "logit" ? "output" : "layer" + (n.layer == null ? "?" : n.layer);
      groups.get(layer).push(n);
    });
    const xy = new Map();
    layers.forEach(function (l, col) {
      const members = groups.get(l);
      members.forEach(function (n, index) {
        xy.set(n.id, {x:52 + col * 780 / Math.max(1, layers.length-1),
          y:42 + (index + 1) * 440 / (members.length + 1)});
      });
    });
    const threshold = Number(value("circuit-threshold")) || 0;
    graph.edges.forEach(function (edge) {
      if (Math.abs(edge.weight) < threshold) return;
      const a = xy.get(edge.source), b = xy.get(edge.target);
      const line = svgNode("line", {x1:a.x,y1:a.y,x2:b.x,y2:b.y,
        stroke:edge.weight >= 0 ? "#4c7eb0" : "#c26b62",
        "stroke-width":Math.min(4,1 + Math.log1p(Math.abs(edge.weight))),
        opacity:0.55});
      const label = svgNode("title");
      label.textContent = edge.source + " → " + edge.target + " · " + edge.weight;
      line.appendChild(label); svg.appendChild(line);
    });
    graph.nodes.forEach(function (node) {
      const loc = xy.get(node.id);
      const circle = svgNode("circle", {cx:loc.x,cy:loc.y,
        r:node.id === state.circuitSelected ? 11 : 7,
        fill:node.kind === "error" ? "#cf9a4b" :
          node.kind === "logit" ? "#4a9d78" :
          node.kind === "token" ? "#8e9dad" : "#427cbe",
        stroke:state.circuitPins.has(node.id) ? "#17263d" : "white",
        "stroke-width":state.circuitPins.has(node.id) ? 3 : 1,
        tabindex:0, role:"button"});
      const title = svgNode("title");
      title.textContent = node.id + " · " + node.kind + " · " + (node.label || "");
      circle.appendChild(title);
      circle.addEventListener("click", function () { selectGraphNode(node.id); });
      circle.addEventListener("keydown", function (event) {
        if (event.key === "Enter") selectGraphNode(node.id);
      });
      svg.appendChild(circle);
    });
    svg.setAttribute("viewBox", "0 0 " + (900 / state.graphScale) +
      " " + (520 / state.graphScale));
  }
  function selectGraphNode(id) {
    const node = state.circuitData?.nodes.find(function (n) { return n.id === id; });
    if (!node) return;
    state.circuitSelected = id;
    const target = byId("circuit-inspector");
    const old = target.querySelector(".graph-selected");
    if (old) old.remove();
    const part = create("div", "graph-selected");
    append(part, "strong", "", node.label || node.id);
    append(part, "p", "quiet", "ID " + node.id + " · " + node.kind +
      " · layer " + (node.layer == null ? "N/A" : node.layer));
    target.prepend(part);
    byId("circuit-note").value = state.circuitNotes[id] || "";
    renderCircuit();
  }
  function updateGraphNotes() {
    if (!state.circuitSelected) return;
    state.circuitNotes[state.circuitSelected] = value("circuit-note").slice(0, 500);
    byId("circuit-status").textContent = "Annotation saved locally; export JSON to keep it.";
  }
  function toggleGraphPin() {
    if (!state.circuitSelected) return;
    if (state.circuitPins.has(state.circuitSelected)) state.circuitPins.delete(state.circuitSelected);
    else state.circuitPins.add(state.circuitSelected);
    renderCircuit();
  }
  function groupGraphPins() {
    const pins = [...state.circuitPins];
    if (pins.length < 2) {
      byId("circuit-status").textContent = "Pin at least two nodes to make an annotation group.";
      return;
    }
    state.circuitGroups.push({name:"Group " + (state.circuitGroups.length+1),members:pins});
    byId("circuit-status").textContent = pins.length + " original graph nodes grouped for annotation.";
  }
  function exportCircuit() {
    if (!state.circuitData) return;
    downloadJson("opensaebase-annotated-circuit.json", {
      ...state.circuitData, annotations:{
        notes:state.circuitNotes, pinned:[...state.circuitPins], groups:state.circuitGroups
      }
    });
  }
  async function loadComputeRoutes() {
    try {
      const status = await get("/api/wp12-compute");
      const target = byId("compute-routes");
      clear(target);
      Object.keys(status).forEach(function (key) {
        const row = append(target, "div", "record");
        append(row, "strong", "", key.replaceAll("_", " "));
        append(row, "p", "quiet", String(status[key]));
      });
    } catch (error) { errorMessage(byId("compute-routes"), error); }
  }

  byId("results-refresh").addEventListener("click",function(){loadResults(true);});
  byId("results-more").addEventListener("click",function(){loadResults(false);});
  byId("results-model").addEventListener("keydown",function(e){if(e.key==="Enter")loadResults(true);});
  byId("feature-search").addEventListener("click",loadFeatures);
  byId("asset-refresh").addEventListener("click",renderAssets);
  byId("asset-model").addEventListener("change",renderAssets);
  byId("feature-query").addEventListener("keydown",function(e){if(e.key==="Enter")loadFeatures();});
  document.querySelectorAll("[data-explore-mode]").forEach(function (button) {
    button.addEventListener("click", function () { setSearchMode(button.dataset.exploreMode); });
  });
  byId("explanation-search").addEventListener("click",searchExplanations);
  byId("explanation-query").addEventListener("keydown",function(e){if(e.key==="Enter")searchExplanations();});
  byId("inference-search").addEventListener("click",searchRecordedInference);
  byId("trace-select").addEventListener("change",selectRecordedTrace);
  byId("training-form").addEventListener("submit",onTrainingExport);
  byId("circuit-open").addEventListener("click",openCircuit);
  byId("circuit-threshold").addEventListener("change",renderCircuit);
  byId("circuit-zoom-in").addEventListener("click",function(){
    state.graphScale=Math.min(3,state.graphScale*1.2);renderCircuit();
  });
  byId("circuit-zoom-out").addEventListener("click",function(){
    state.graphScale=Math.max(.6,state.graphScale/1.2);renderCircuit();
  });
  byId("circuit-export").addEventListener("click",exportCircuit);
  byId("circuit-save-note").addEventListener("click",updateGraphNotes);
  byId("circuit-pin").addEventListener("click",toggleGraphPin);
  byId("circuit-group").addEventListener("click",groupGraphPins);
  byId("intervention-form").addEventListener("submit",onPreflight);
  window.addEventListener("hashchange",activate);
  bootstrap();
})();
