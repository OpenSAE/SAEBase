/* Compatibility cleanup for old SAEBase Atlas snapshots.
 * Navigation batches are UI state, never scientific/source nodes.
 * Do not create pseudo-components or infer Context from node counts.
 */
(function () {
  "use strict";
  const synthetic = node =>
    /:(?:public-nav|nav):/.test(String(node?.id || "")) ||
    /^(?:按源码职责浏览相关操作|分组浏览相关接口与方法)（第\d+组）$/.test(
      String(node?.title || ""));

  function removeFakeNavigation(graph) {
    if (!graph?.nodes || !graph?.edges) return { removed: 0 };
    const removed = new Set(graph.nodes.filter(synthetic).map(node => node.id));
    if (!removed.size) return { removed: 0 };
    const parents = new Map();
    for (const edge of graph.edges) {
      if (edge.type === "contains" && !parents.has(edge.target))
        parents.set(edge.target, edge.source);
    }
    function nearestRealParent(id) {
      let parent = parents.get(id), visited = new Set([id]);
      while (removed.has(parent)) {
        if (visited.has(parent)) throw Error("Cyclic legacy Atlas navigation");
        visited.add(parent);
        parent = parents.get(parent);
      }
      return parent;
    }
    const known = new Set(graph.nodes.filter(n => !removed.has(n.id)).map(n => n.id));
    const kept = [];
    const seen = new Set();
    function keep(edge) {
      if (!known.has(edge.source) || !known.has(edge.target) ||
          edge.source === edge.target) return;
      const key = edge.type + "|" + edge.source + "|" + edge.target;
      if (seen.has(key)) return;
      seen.add(key);
      kept.push(edge);
    }
    for (const edge of graph.edges) {
      if (removed.has(edge.target)) continue;
      if (edge.type === "contains" && removed.has(edge.source)) {
        const parent = nearestRealParent(edge.target);
        if (parent) keep({ source: parent, target: edge.target, type: "contains" });
      } else if (!removed.has(edge.source)) {
        keep(edge);
      }
    }
    graph.nodes = graph.nodes.filter(node => !removed.has(node.id));
    graph.edges = kept;
    return { removed: removed.size };
  }

  window.SAEBASE_ATLAS_REMOVE_FAKE_NAVIGATION = removeFakeNavigation;
})();
