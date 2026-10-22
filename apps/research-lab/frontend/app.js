/* OpenSAE Research Lab: dependency-free, accessible, read-only UI.
   Every displayed measurement originates from a JSON API with provenance.
   No demo text, guessed feature labels or fabricated experimental numbers.
*/
"use strict";
(function () {
  const state = { sites: [], datasets: [], assets: [], assetsLoaded: false, tracesLoaded: false, activeTrace: null, searchMode: "explanation", current: "overview", resultOffset: 0, resultTotal: 0 };
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
  byId("intervention-form").addEventListener("submit",onPreflight);
  window.addEventListener("hashchange",activate);
  bootstrap();
})();
