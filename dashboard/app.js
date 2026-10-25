/**
 * Sentinel Graph | Forensic AML Network Analytics Dashboard
 * IBM HI-Small AML Dataset — Real Transaction Data
 * Authors: Aakash R (23MIA1011), Gowthamkrishna R (23MIA1087)
 */

// Utility: Escape HTML special characters to prevent XSS in rendered content
function escapeHtml(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

// Global State
const state = {
  overview: null,
  graphData: null,
  cycles: [],
  leads: [],
  metrics: null,
  ablation: [],
  shap: null,
  selectedAccount: null,
  simulation: null,
  svg: null,
  g: null,
  activeTab: 'graphView',
  currentSar: null
};

const CYPHER_TEMPLATES = {
  cycle: `MATCH path = (a:Account)-[:TRANSFER*3..5]->(a)
WITH a, path, relationships(path) AS txs
RETURN a.id AS start_account,
       [n IN nodes(path) | n.id] AS account_chain,
       length(path) AS cycle_length,
       reduce(total = 0.0, t IN txs | total + t.amount) AS total_volume
ORDER BY total_volume DESC LIMIT 10;`,

  smurf: `MATCH (mule:Account)-[r:TRANSFER]->(collector:Account)
WHERE r.amount >= 7500.0 AND r.amount < 10000.0
WITH collector, count(DISTINCT mule) AS mule_count,
     sum(r.amount) AS total_collected
WHERE mule_count >= 3
RETURN collector.id AS aggregator,
       mule_count,
       total_collected
ORDER BY total_collected DESC;`,

  pagerank: `CALL gds.pageRank.stream('amlNetwork', { dampingFactor: 0.85 })
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).id AS account, score AS pagerank
ORDER BY pagerank DESC LIMIT 15;`,

  betweenness: `CALL gds.betweenness.stream('amlNetwork')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).id AS account, score AS betweenness
ORDER BY betweenness DESC LIMIT 15;`,

  louvain: `CALL gds.louvain.stream('amlNetwork')
YIELD nodeId, communityId
RETURN communityId, count(*) AS accounts_in_community
ORDER BY accounts_in_community DESC;`
};

// Plain-language meaning of every model feature (see ml_pipeline/feature_fusion.py)
const FEATURE_INFO = {
  amount: ['Transaction', 'Amount paid, in the transaction\'s own currency'],
  log_amount: ['Transaction', 'log(1 + amount); softens the effect of very large amounts'],
  hour_of_day: ['Transaction', 'Hour the transfer was made (0-23)'],
  is_wire: ['Transaction', '1 if paid by wire transfer'],
  is_ach: ['Transaction', '1 if paid by ACH (automated bank transfer)'],
  is_cash: ['Transaction', '1 if paid in cash'],
  is_cheque: ['Transaction', '1 if paid by cheque'],
  is_credit_card: ['Transaction', '1 if paid by credit card'],
  is_reinvestment: ['Transaction', '1 if the payment is a reinvestment'],
  is_bitcoin: ['Transaction', '1 if paid in Bitcoin'],
  cross_currency: ['Transaction', '1 if the paid and received currencies differ'],
  cross_bank: ['Transaction', '1 if sender and receiver use different banks'],
  self_transfer: ['Transaction', '1 if the account pays itself'],
  src_pagerank: ['Graph', 'Sender\'s PageRank: how much money importance flows into the sender'],
  dst_pagerank: ['Graph', 'Receiver\'s PageRank: how much money importance flows into the receiver'],
  pagerank_ratio: ['Graph', 'Sender PageRank divided by receiver PageRank; high when money moves from a hub to a minor account'],
  src_betweenness: ['Graph', 'Sender\'s betweenness: how often it bridges other accounts (money-mule signal)'],
  dst_betweenness: ['Graph', 'Receiver\'s betweenness: how often it bridges other accounts'],
  src_in_degree: ['Graph', 'Number of different accounts that send money to the sender'],
  src_out_degree: ['Graph', 'Number of different accounts the sender pays'],
  dst_in_degree: ['Graph', 'Number of different accounts that pay the receiver (fan-in / collector signal)'],
  dst_out_degree: ['Graph', 'Number of different accounts the receiver pays'],
  src_flow_asymmetry: ['Graph', 'Sender\'s (out - in) / total volume; near 0 means money passes straight through'],
  dst_flow_asymmetry: ['Graph', 'Receiver\'s (out - in) / total volume; near 0 means money passes straight through'],
  src_in_cycle: ['Graph', '1 if the sender is in a detected 3-5 hop ring'],
  dst_in_cycle: ['Graph', '1 if the receiver is in a detected 3-5 hop ring'],
  src_cycle_count: ['Graph', 'Number of rings the sender belongs to'],
  dst_cycle_count: ['Graph', 'Number of rings the receiver belongs to'],
  src_velocity_per_day: ['Graph', 'Sender\'s transactions per active day (bursts of activity)'],
  same_community: ['Graph', '1 if sender and receiver are in the same Louvain cluster'],
};

function featureInfo(name) {
  if (FEATURE_INFO[name]) return FEATURE_INFO[name];
  if (name.startsWith('cur_')) {
    const cur = name.slice(4).split('_').map(w => w[0].toUpperCase() + w.slice(1)).join(' ');
    return ['Transaction', `1 if the payment currency is ${cur}`];
  }
  return ['Other', ''];
}

function renderFeatureGlossary() {
  const container = document.getElementById('featureGlossary');
  if (!container) return;
  const names = [
    'amount', 'log_amount', 'hour_of_day',
    'is_wire', 'is_ach', 'is_cash', 'is_cheque', 'is_credit_card', 'is_reinvestment', 'is_bitcoin',
    'cur_*', 'cross_currency', 'cross_bank', 'self_transfer',
    'src_pagerank', 'dst_pagerank', 'pagerank_ratio', 'src_betweenness', 'dst_betweenness',
    'src_in_degree', 'src_out_degree', 'dst_in_degree', 'dst_out_degree',
    'src_flow_asymmetry', 'dst_flow_asymmetry', 'src_in_cycle', 'dst_in_cycle',
    'src_cycle_count', 'dst_cycle_count', 'src_velocity_per_day', 'same_community'
  ];
  const row = n => {
    const [group, text] = n === 'cur_*'
      ? ['Transaction', '15 flags, one per payment currency (US Dollar, Euro, Yuan, Rupee, Bitcoin, ...)']
      : featureInfo(n);
    return `<div class="glossary-row"><code>${n}</code><span>${text}</span></div>`;
  };
  const group = g => names.filter(n => (n === 'cur_*' ? 'Transaction' : featureInfo(n)[0]) === g).map(row).join('');
  container.innerHTML = `
    <div class="glossary-group"><h4>Transaction features (28)</h4>${group('Transaction')}</div>
    <div class="glossary-group"><h4>Graph features (17) <span class="badge gds-badge">GRAPH</span></h4>${group('Graph')}</div>
  `;
}

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', async () => {
  setupTabs();
  setupCypherConsole();
  setupEventListeners();
  showLoadingOverlay(true);

  try {
    await loadOverview();
    await Promise.all([
      loadGraphData(),
      loadCycles(),
      loadLeads(),
      loadMetrics(),
      loadAblation(),
      loadShap()
    ]);
  } catch (err) {
    console.error('Initialization error:', err);
  } finally {
    showLoadingOverlay(false);
  }
});

function showLoadingOverlay(show) {
  const overlay = document.getElementById('loadingOverlay');
  if (overlay) {
    overlay.style.display = show ? 'flex' : 'none';
  }
}

// Tab Switcher
function setupTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      tabs.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const target = btn.getAttribute('data-tab');
      state.activeTab = target;
      const targetEl = document.getElementById(target);
      if (targetEl) targetEl.classList.add('active');

      if (target === 'graphView' && state.simulation) {
        state.simulation.alpha(0.3).restart();
      }
    });
  });
}

// 1. Fetch Overview & Metrics
async function loadOverview() {
  try {
    const res = await fetch('/api/overview');
    const data = await res.json();
    state.overview = data;

    animateCounter('kpiAccounts', data.total_accounts || 0);
    animateCounter('kpiTransfers', data.total_transactions || 0);
    animateCounter('kpiLaunderingVolume', data.laundering_transactions || 0);
    document.getElementById('kpiImbalance').textContent = `${data.imbalance_ratio_pct}% of sample transactions`;
    animateCounter('kpiRings', data.detected_laundering_rings || 0);
    const lift = data.pr_auc_lift_pct || 0;
    document.getElementById('kpiPRAUC').textContent = `${lift >= 0 ? '+' : ''}${lift}%`;
    document.getElementById('kpiPRAUCSub').textContent = data.evaluation_scope || 'Graph GB vs tabular baseline';
    document.getElementById('kpiBankCount').textContent = `Active graph nodes across ${data.unique_banks || 0} banks`;

    const statusEl = document.getElementById('engineStatusText');
    const dotEl = document.getElementById('engineDot');

    if (data.neo4j_live_connected) {
      statusEl.textContent = data.gds_available ? 'Neo4j 5 Bolt + GDS Active' : 'Neo4j 5 Bolt Connected';
      dotEl.className = 'status-dot pulse-green';
    } else {
      statusEl.textContent = 'In-memory graph engine';
      dotEl.className = 'status-dot pulse-cyan';
    }

    // Update dataset badge
    const badge = document.getElementById('badgeEngine');
    if (badge) badge.textContent = `IBM HI-Small • ${(data.total_transactions || 0).toLocaleString('en-US')} TX`;
  } catch (err) {
    console.error('Failed to load overview:', err);
  }
}

function animateCounter(elementId, targetValue) {
  const el = document.getElementById(elementId);
  if (!el) return;
  const duration = 800;
  const startTime = performance.now();
  const startVal = 0;

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(startVal + (targetValue - startVal) * eased);
    el.textContent = current.toLocaleString('en-US');
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

// 2. Fetch Graph Data & Render D3 Force Simulation
async function loadGraphData() {
  try {
    const res = await fetch('/api/graph?max_nodes=200');
    const data = await res.json();
    state.graphData = data;
    renderForceGraph(data);
  } catch (err) {
    console.error('Failed to load graph data:', err);
  }
}

function renderForceGraph(data) {
  const container = document.getElementById('graphSvgContainer');
  container.innerHTML = '';

  const width = container.clientWidth || 900;
  const height = container.clientHeight || 650;

  const svg = d3.select(container)
    .append('svg')
    .attr('width', '100%')
    .attr('height', '100%')
    .attr('viewBox', [0, 0, width, height]);

  state.svg = svg;

  const g = svg.append('g');
  state.g = g;

  const zoom = d3.zoom()
    .scaleExtent([0.15, 5])
    .on('zoom', (event) => g.attr('transform', event.transform));

  svg.call(zoom);

  const defs = svg.append('defs');

  // Glow filter for cycle nodes
  const filter = defs.append('filter').attr('id', 'glow');
  filter.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'coloredBlur');
  const feMerge = filter.append('feMerge');
  feMerge.append('feMergeNode').attr('in', 'coloredBlur');
  feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

  defs.append('marker')
    .attr('id', 'arrow-normal')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 22)
    .attr('refY', 0)
    .attr('markerWidth', 5)
    .attr('markerHeight', 5)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', 'rgba(0, 0, 0, 0.1)');

  defs.append('marker')
    .attr('id', 'arrow-danger')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 24)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', '#dc2626');

  const simulation = d3.forceSimulation(data.nodes)
    .force('link', d3.forceLink(data.links).id(d => d.id).distance(55))
    .force('charge', d3.forceManyBody().strength(-120))
    .force('center', d3.forceCenter(width / 2, height / 2))
    // Pull disconnected fragments toward the middle so nothing drifts out of view
    .force('x', d3.forceX(width / 2).strength(0.06))
    .force('y', d3.forceY(height / 2).strength(0.06))
    .force('collision', d3.forceCollide().radius(16));

  state.simulation = simulation;

  const link = g.append('g')
    .attr('class', 'links')
    .selectAll('line')
    .data(data.links)
    .join('line')
    .attr('stroke', d => d.isLaundering === 1 ? '#dc2626' : 'rgba(0, 0, 0, 0.25)')
    .attr('stroke-width', d => d.isLaundering === 1 ? 2.5 : 0.8)
    .attr('stroke-opacity', d => d.isLaundering === 1 ? 0.9 : 0.35)
    .attr('marker-end', d => d.isLaundering === 1 ? 'url(#arrow-danger)' : 'url(#arrow-normal)');

  const node = g.append('g')
    .attr('class', 'nodes')
    .selectAll('g')
    .data(data.nodes)
    .join('g')
    .call(d3.drag()
      .on('start', dragstarted)
      .on('drag', dragged)
      .on('end', dragended)
    )
    .on('click', (event, d) => inspectAccount(d.id))
    .style('cursor', 'pointer');

  node.append('circle')
    .attr('r', d => {
      const base = 5;
      if (d.inCycle) return base + 4;
      if (d.betweenness > 0.02) return base + 5;
      return base + Math.min(8, d.pagerank * 200);
    })
    .attr('fill', d => {
      if (d.inCycle) return '#dc2626';
      if (d.betweenness > 0.02) return '#d97706';
      return '#1d4ed8';
    })
    .attr('stroke', '#ffffff')
    .attr('stroke-width', 1.5);

  // Only show labels for notable nodes to avoid clutter
  node.filter(d => d.inCycle || d.betweenness > 0.02 || d.pagerank > 0.005)
    .append('text')
    .attr('x', 12)
    .attr('y', 4)
    .text(d => d.id.length > 10 ? d.id.slice(-6) : d.id)
    .attr('font-size', '8px')
    .attr('font-family', 'JetBrains Mono')
    .attr('fill', '#374151')
    .style('pointer-events', 'none');

  // Tooltip on hover
  node.append('title')
    .text(d => `${d.id}\nBank: ${d.bank}\nPageRank: ${d.pagerank?.toFixed(5)}\nBetweenness: ${d.betweenness?.toFixed(5)}\nIn Cycle: ${d.inCycle ? 'YES' : 'No'}`);

  const renderPositions = () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  };
  simulation.on('tick', renderPositions);

  // Settle the layout up front so the graph appears stable, then zoom so every node is visible
  simulation.stop();
  for (let i = 0; i < 300; i++) simulation.tick();
  renderPositions();

  const xs = data.nodes.map(d => d.x), ys = data.nodes.map(d => d.y);
  if (xs.length) {
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const pad = 40;
    const scale = Math.min(2, 0.95 / Math.max((x1 - x0 + pad) / width, (y1 - y0 + pad) / height));
    const tx = width / 2 - scale * (x0 + x1) / 2;
    const ty = height / 2 - scale * (y0 + y1) / 2;
    svg.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
  }

  function dragstarted(event) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    event.subject.fx = event.subject.x;
    event.subject.fy = event.subject.y;
  }
  function dragged(event) {
    event.subject.fx = event.x;
    event.subject.fy = event.y;
  }
  function dragended(event) {
    if (!event.active) simulation.alphaTarget(0);
    event.subject.fx = null;
    event.subject.fy = null;
  }
}

// 3. Load Laundering Cycles
async function loadCycles() {
  try {
    const res = await fetch('/api/cycles');
    const data = await res.json();
    state.cycles = data.cycles || [];

    document.getElementById('tabCycleCount').textContent = state.cycles.length;

    let totalVol = 0;
    state.cycles.forEach(c => totalVol += (c.total_volume || 0));
    document.getElementById('totalRingVolume').textContent = totalVol.toLocaleString('en-US', {maximumFractionDigits: 0});
    const confirmedEl = document.getElementById('ringsConfirmed');
    if (confirmedEl) confirmedEl.textContent = `${data.confirmed_count || 0} of ${state.cycles.length} rings contain labelled laundering transfers (labels shown for validation only)`;

    renderCyclesGrid(state.cycles);
  } catch (err) {
    console.error('Failed to load cycles:', err);
  }
}

function renderCyclesGrid(cycles) {
  const container = document.getElementById('ringsListContainer');
  if (!cycles.length) {
    container.innerHTML = `
      <div class="empty-state">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
        </svg>
        <p>No circular layering rings detected in the current sample. Run <code>python3 -m ml_pipeline.build_artifacts</code> to build a ring-preserving sample.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = cycles.slice(0, 30).map((c, idx) => {
    const pathHtml = c.accounts.map((acc, i) => {
      const shortAcc = acc.length > 10 ? '...' + acc.slice(-7) : acc;
      return `<span class="path-acc">${shortAcc}</span>${i < c.accounts.length - 1 ? '<span class="path-arrow">&rarr;</span>' : '<span class="path-arrow">&#8635;</span>'}`;
    }).join(' ');

    const flagged = c.confirmed_laundering ? '<span class="badge danger">CONFIRMED</span>' : '<span class="badge">UNLABELLED</span>';

    return `
      <div class="ring-card ${c.confirmed_laundering ? 'ring-flagged' : ''}" style="animation-delay: ${idx * 0.05}s;">
        <div class="ring-header">
          <span class="ring-id">${c.cycle_id}</span>
          <div style="display:flex; gap:0.4rem; align-items:center;">
            <span class="badge warning">${c.length}-Hop Ring</span>
            ${flagged}
          </div>
        </div>
        <div class="ring-path-visual">${pathHtml}</div>
        <div class="ring-meta">
          <span>Volume: <strong>${c.total_volume.toLocaleString('en-US', {maximumFractionDigits: 0})}</strong></span>
          <span>Labelled hops: <strong>${c.laundering_hops || 0}/${c.length}</strong></span>
        </div>
        <div class="ring-actions">
          <button class="btn btn-secondary" onclick="inspectAccount('${c.accounts[0]}')">Inspect Origin</button>
        </div>
      </div>
    `;
  }).join('');
}

// 4. Load Prioritized Leads (Triage Table)
async function loadLeads() {
  try {
    const res = await fetch('/api/leads?limit=60');
    const data = await res.json();
    state.leads = data.leads || [];
    renderLeadsTable(state.leads);
  } catch (err) {
    console.error('Failed to load leads:', err);
  }
}

function renderLeadsTable(leads) {
  const tbody = document.getElementById('leadsTableBody');
  if (!leads.length) {
    tbody.innerHTML = '<tr><td colspan="10" style="text-align:center; padding: 2rem; color:var(--text-dim);">No leads found.</td></tr>';
    return;
  }

  tbody.innerHTML = leads.map((l, idx) => {
    const score = l.risk_score || 0.0;
    let scoreClass = 'score-low';
    if (score >= 0.8) scoreClass = 'score-critical';
    else if (score >= 0.5) scoreClass = 'score-high';
    else if (score >= 0.3) scoreClass = 'score-med';

    // Field names from fused DataFrame
    const fromAcc = l.account_from || l.Account || '-';
    const toAcc = l.account_to || l['Account.1'] || '-';
    const fromBank = l.from_bank || l['From Bank'] || '-';
    const toBank = l.to_bank || l['To Bank'] || '-';
    const typology = l.typology || l.Typology || 'NORMAL';

    const graphFlags = [];
    if (l.src_in_cycle || l.dst_in_cycle) graphFlags.push('<span class="badge danger">In Cycle</span>');
    if ((l.src_betweenness || 0) > 0.02) graphFlags.push('<span class="badge warning">Bridge</span>');
    if (l.same_community) graphFlags.push('<span class="badge">Same Cluster</span>');

    const shortFrom = fromAcc.length > 10 ? '...' + fromAcc.slice(-7) : fromAcc;
    const shortTo = toAcc.length > 10 ? '...' + toAcc.slice(-7) : toAcc;

    const paymentFmt = l.is_wire ? 'Wire' : l.is_ach ? 'ACH' : l.is_cash ? 'Cash'
      : l.is_cheque ? 'Cheque' : l.is_credit_card ? 'Credit Card'
      : l.is_reinvestment ? 'Reinvestment' : l.is_bitcoin ? 'Bitcoin' : 'Other';

    return `
      <tr style="animation-delay: ${idx * 0.02}s;">
        <td><span class="score-badge ${scoreClass}">${(score * 100).toFixed(1)}%</span></td>
        <td class="mono-cell"><a href="javascript:void(0)" onclick="inspectAccount('${fromAcc}')" class="acc-link">${shortFrom}</a></td>
        <td>${fromBank}</td>
        <td class="mono-cell"><a href="javascript:void(0)" onclick="inspectAccount('${toAcc}')" class="acc-link">${shortTo}</a></td>
        <td>${toBank}</td>
        <td class="mono-cell"><strong>${Number(l.amount || 0).toLocaleString('en-US', {maximumFractionDigits: 0})}</strong> <span style="color:var(--text-muted); font-size:0.7rem;">${escapeHtml(l.currency || '')}</span></td>
        <td>${paymentFmt}</td>
        <td>${graphFlags.join(' ') || '<span style="color:var(--text-dim);">—</span>'}</td>
        <td><span class="badge ${typology !== 'NORMAL' ? 'danger' : ''}">${typology}</span></td>
        <td>
          <button class="btn-action" title="Generate SAR" onclick="viewSARReport('${fromAcc}')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>
          </button>
        </td>
      </tr>
    `;
  }).join('');
}

// 5. Load Model Metrics & Precision-Recall Curves
async function loadMetrics() {
  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    state.metrics = data.metrics;

    const tab = state.metrics.tabular_baseline || {};
    const graph = state.metrics.graph_gradient_boosting || {};
    setEvaluationNotes(data.evaluation_scope, state.metrics);

    document.getElementById('tabRoc').textContent = tab.roc_auc || '-';
    document.getElementById('tabPr').textContent = tab.pr_auc || '-';
    const pk = (m, k) => (m.precision_at_k || {})[k] !== undefined ? m.precision_at_k[k].toFixed(2) : '-';
    document.getElementById('tabPrec').textContent = pk(tab, 'P@50');
    document.getElementById('tabRec').textContent = pk(tab, 'P@100');

    document.getElementById('graphRoc').textContent = graph.roc_auc || '-';
    document.getElementById('graphPr').textContent = graph.pr_auc || '-';
    document.getElementById('graphPrec').textContent = pk(graph, 'P@50');
    document.getElementById('graphRec').textContent = pk(graph, 'P@100');

    renderConfusionMatrix(graph.confusion_matrix || {});
    renderPrecisionAtK(graph.precision_at_k || {});
    renderPrecisionRecallCurves(tab.pr_curve || [], graph.pr_curve || []);
    renderFeatureImportance(state.metrics.feature_importance || []);
    renderFeatureGlossary();
  } catch (err) {
    console.error('Failed to load metrics:', err);
  }
}

// Shared caption explaining which data the evaluation tabs were computed on
function setEvaluationNotes(scope, metrics) {
  const split = (metrics || {}).split || {};
  const testRows = (metrics || {}).test_sample_size;
  const parts = [scope || 'Temporal split'];
  if (split.test_start) parts.push(`test period from ${split.test_start}`);
  if (testRows) parts.push(`${testRows.toLocaleString('en-US')} test transactions`);
  if (split.test_positive_rate !== undefined) parts.push(`${(split.test_positive_rate * 100).toFixed(3)}% laundering`);
  document.querySelectorAll('.evaluation-note:not(#ringsConfirmed)').forEach(el => { el.textContent = parts.join(' · '); });
}

function renderConfusionMatrix(cm) {
  const container = document.getElementById('cmContainer');
  container.innerHTML = `
    <table class="cm-table">
      <thead>
        <tr>
          <th></th>
          <th>Actual Positive</th>
          <th>Actual Negative</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Pred. Positive</strong></td>
          <td class="cm-tp"><div class="cm-cell-val">${cm.true_positives || 0}</div><span>True Positive</span></td>
          <td class="cm-fp"><div class="cm-cell-val">${cm.false_positives || 0}</div><span>False Positive</span></td>
        </tr>
        <tr>
          <td><strong>Pred. Negative</strong></td>
          <td class="cm-fn"><div class="cm-cell-val">${cm.false_negatives || 0}</div><span>False Negative</span></td>
          <td class="cm-tn"><div class="cm-cell-val">${cm.true_negatives || 0}</div><span>True Negative</span></td>
        </tr>
      </tbody>
    </table>
  `;
}

function renderPrecisionAtK(pAtK) {
  const container = document.getElementById('precisionAtKContainer');
  const items = Object.entries(pAtK).map(([k, v]) => `
    <div class="patk-card">
      <div class="patk-label">${k}</div>
      <div class="patk-value">${(v * 100).toFixed(1)}%</div>
    </div>
  `).join('');

  container.innerHTML = `
    <div class="patk-header">PRECISION @ TOP AUDIT RANKS</div>
    <div class="patk-grid">${items}</div>
  `;
}

function renderPrecisionRecallCurves(tabCurve, graphCurve) {
  const container = document.getElementById('prCurveContainer');
  container.innerHTML = '';

  const width = container.clientWidth || 400;
  const height = container.clientHeight || 260;
  const margin = { top: 20, right: 30, bottom: 40, left: 50 };

  const svg = d3.select(container).append('svg')
    .attr('width', width)
    .attr('height', height);

  const x = d3.scaleLinear().domain([0, 1]).range([margin.left, width - margin.right]);
  const y = d3.scaleLinear().domain([0, 1]).range([height - margin.bottom, margin.top]);

  // Grid lines
  svg.append('g')
    .attr('class', 'grid')
    .attr('transform', `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(5).tickSize(-(width - margin.left - margin.right)).tickFormat(''))
    .selectAll('line').attr('stroke', 'rgba(0,0,0,0.05)');

  svg.append('g')
    .attr('transform', `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(5))
      .attr('color', '#6b7280');

  svg.append('g')
    .attr('transform', `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(5))
      .attr('color', '#6b7280');

  svg.append('text')
    .attr('x', width / 2).attr('y', height - 4)
    .attr('text-anchor', 'middle').attr('fill', '#374151').attr('font-size', '10px')
    .text('Recall');

  svg.append('text')
    .attr('transform', 'rotate(-90)')
    .attr('x', -height / 2).attr('y', 14)
    .attr('text-anchor', 'middle').attr('fill', '#374151').attr('font-size', '10px')
    .text('Precision');

  const line = d3.line().x(d => x(d.x)).y(d => y(d.y));

  if (tabCurve.length > 0) {
    svg.append('path')
      .datum(tabCurve)
      .attr('fill', 'none')
      .attr('stroke', '#9ca3af')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '6,4')
      .attr('d', line);
  }

  if (graphCurve.length > 0) {
    // Area under graph curve
    const area = d3.area().x(d => x(d.x)).y0(height - margin.bottom).y1(d => y(d.y));
    svg.append('path')
      .datum(graphCurve)
      .attr('fill', 'rgba(29, 78, 216, 0.08)')
      .attr('d', area);

    svg.append('path')
      .datum(graphCurve)
      .attr('fill', 'none')
      .attr('stroke', '#1d4ed8')
      .attr('stroke-width', 2.5)
      .attr('d', line);
  }

  const legend = svg.append('g').attr('transform', `translate(${width - 170}, ${margin.top + 5})`);
  legend.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 0).attr('y2', 0).attr('stroke', '#1d4ed8').attr('stroke-width', 2.5);
  legend.append('text').attr('x', 24).attr('y', 4).attr('font-size', '9px').attr('fill', '#1d4ed8').text('Graph-Augmented GB');
  legend.append('line').attr('x1', 0).attr('x2', 18).attr('y1', 16).attr('y2', 16).attr('stroke', '#9ca3af').attr('stroke-width', 2).attr('stroke-dasharray', '4,3');
  legend.append('text').attr('x', 24).attr('y', 20).attr('font-size', '9px').attr('fill', '#374151').text('Tabular Baseline');
}

// 6. Load Ablation Study
async function loadAblation() {
  try {
    const res = await fetch('/api/ablation');
    const data = await res.json();
    state.ablation = data.stages || [];
    renderAblationTable(state.ablation);
    renderAblationChart(state.ablation);
  } catch (err) {
    console.error('Failed to load ablation study:', err);
  }
}

function renderAblationTable(stages) {
  const tbody = document.getElementById('ablationTableBody');
  if (!stages.length) return;

  tbody.innerHTML = stages.map((s, i) => `
    <tr style="animation-delay: ${i * 0.05}s;">
      <td class="mono-cell" style="white-space:nowrap;"><strong>Stage ${s.stage_num}</strong></td>
      <td style="color:var(--accent-cyan); font-weight:600;">${s.stage}</td>
      <td style="color:var(--text-muted); font-size:0.78rem;">${s.description}</td>
      <td class="mono-cell">${s.feature_count}</td>
      <td class="mono-cell" style="font-weight:700; color:#111827;">${s.pr_auc.toFixed(4)}</td>
      <td class="mono-cell">${s.roc_auc.toFixed(4)}</td>
      <td>
        <span class="badge ${i === 0 ? '' : s.incremental_pr_auc_lift >= 0 ? 'success' : 'danger'}">
          ${i === 0 ? 'Baseline' : `${s.incremental_pr_auc_lift >= 0 ? '+' : ''}${s.incremental_pr_auc_lift.toFixed(4)}`}
        </span>
      </td>
    </tr>
  `).join('');
}

function renderAblationChart(stages) {
  const container = document.getElementById('ablationChartContainer');
  container.innerHTML = '';

  const width = container.clientWidth || 700;
  const height = 280;
  const margin = { top: 25, right: 30, bottom: 60, left: 55 };

  const svg = d3.select(container).append('svg')
    .attr('width', width)
    .attr('height', height);

  const x = d3.scaleBand()
    .domain(stages.map(s => s.stage))
    .range([margin.left, width - margin.right])
    .padding(0.3);

  const maxPr = Math.max(...stages.map(s => s.pr_auc), 0.01);
  const y = d3.scaleLinear()
    .domain([0, Math.min(1.05, maxPr * 1.3)])
    .range([height - margin.bottom, margin.top]);

  svg.append('g')
    .attr('transform', `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x))
    .selectAll('text')
    .attr('color', '#374151').attr('font-size', '8px')
    .attr('transform', 'rotate(-12)').style('text-anchor', 'end');

  svg.append('g')
    .attr('transform', `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(5))
      .attr('color', '#6b7280');

  // Gradient
  const defs = svg.append('defs');
  const grad = defs.append('linearGradient').attr('id', 'barGradChampion').attr('x1', '0').attr('y1', '0').attr('x2', '0').attr('y2', '1');
  grad.append('stop').attr('offset', '0%').attr('stop-color', '#475569');
  grad.append('stop').attr('offset', '100%').attr('stop-color', '#334155');

  svg.selectAll('.bar')
    .data(stages)
    .join('rect')
    .attr('x', d => x(d.stage))
    .attr('y', height - margin.bottom)
    .attr('width', x.bandwidth())
    .attr('height', 0)
    .attr('fill', (d, i) => i === stages.length - 1 ? 'url(#barGradChampion)' : (i === 0 ? '#e5e7eb' : '#3b82f6'))
    .attr('rx', 4)
    .transition()
    .duration(600)
    .delay((d, i) => i * 100)
    .attr('y', d => y(d.pr_auc))
    .attr('height', d => (height - margin.bottom) - y(d.pr_auc));

  // Value labels (after animation)
  setTimeout(() => {
    svg.selectAll('.val-label')
      .data(stages)
      .join('text')
      .attr('x', d => x(d.stage) + x.bandwidth() / 2)
      .attr('y', d => y(d.pr_auc) - 8)
      .attr('text-anchor', 'middle')
      .attr('fill', '#111827')
      .attr('font-size', '10px')
      .attr('font-family', 'JetBrains Mono')
      .text(d => d.pr_auc.toFixed(3));
  }, 800);
}

// 7. Load SHAP Explainability
async function loadShap() {
  try {
    const res = await fetch('/api/shap');
    const data = await res.json();
    state.shap = data.shap || {};

    const container = document.getElementById('shapInsightsContainer');
    const ranking = state.shap.ranking || [];

    container.innerHTML = `
      <div class="shap-highlight-card">
        <div class="shap-highlight-icon">⚡</div>
        <div>
          <h4>Top Risk Driver: <span style="color:var(--accent-cyan);">${escapeHtml(state.shap.top_driving_feature || '-')}</span></h4>
          <p>${state.shap.status === 'computed' ? 'Mean absolute SHAP values from TreeExplainer on the graph-augmented Random Forest' : 'Approximated from Random Forest feature importances (SHAP not installed)'}.
          ${graphShare(ranking)}</p>
        </div>
      </div>

      <div class="shap-ranking-list">
        <div class="shap-ranking-header">Mean |SHAP| ranking</div>
        ${ranking.slice(0, 8).map((r, i) => `
          <div class="shap-rank-item">
            <span class="shap-rank-num">#${i+1}</span>
            <span class="shap-rank-name">${r.feature} ${/^(src_|dst_)|same_community|pagerank_ratio/.test(r.feature) ? '<span class="badge gds-badge">GRAPH</span>' : ''}
              <span class="feature-desc">${escapeHtml(featureInfo(r.feature)[1])}</span></span>
            <span class="shap-rank-value">${r.mean_shap_value !== undefined ? r.mean_shap_value.toFixed(4) : (r.importance * 100).toFixed(1) + '%'}</span>
          </div>
        `).join('')}
      </div>
    `;
  } catch (err) {
    console.error('Failed to load SHAP:', err);
  }
}

// Share of the top-8 attribution that comes from graph (src_/dst_/community/pagerank) features
function graphShare(ranking) {
  const top = ranking.slice(0, 8);
  const total = top.reduce((a, r) => a + (r.mean_shap_value || 0), 0);
  if (!total) return '';
  const isGraph = f => /^(src_|dst_)|same_community|pagerank_ratio/.test(f);
  const graph = top.filter(r => isGraph(r.feature)).reduce((a, r) => a + r.mean_shap_value, 0);
  return `Graph features account for <strong>${(graph / total * 100).toFixed(0)}%</strong> of the attribution among the top 8 features.`;
}

function renderFeatureImportance(features) {
  const container = document.getElementById('featureImportanceList');
  if (!features.length) return;

  const maxImp = Math.max(...features.map(f => f.importance));

  container.innerHTML = features.slice(0, 10).map((f, i) => {
    const pct = ((f.importance / maxImp) * 100).toFixed(1);
    return `
      <div class="f-bar-row" style="animation-delay: ${i * 0.04}s;">
        <div class="f-bar-labels">
          <span>${f.feature} ${f.is_graph_feature ? '<span class="badge gds-badge">GRAPH</span>' : ''}
            <span class="feature-desc">${escapeHtml(featureInfo(f.feature)[1])}</span></span>
          <span class="mono-cell">${(f.importance * 100).toFixed(2)}%</span>
        </div>
        <div class="f-bar-track">
          <div class="f-bar-fill ${f.is_graph_feature ? '' : 'tabular'}" style="width: ${pct}%"></div>
        </div>
      </div>
    `;
  }).join('');
}

// 8. Account Inspection & SAR Viewer
async function inspectAccount(accountId) {
  try {
    const res = await fetch(`/api/account/${accountId}`);
    if (!res.ok) {
      const err = await res.json();
      showToast(err.detail || 'Account not found', 'error');
      return;
    }
    const data = await res.json();
    state.selectedAccount = data;

    const badge = document.getElementById('sideInspectorBadge');
    badge.textContent = accountId.length > 12 ? '...' + accountId.slice(-8) : accountId;
    badge.className = data.in_cycle ? 'badge danger' : 'badge';

    const sideHtml = `
      <div class="inspector-content">
        <div class="inspector-header">
          <h4>${accountId.length > 12 ? '...' + accountId.slice(-10) : accountId}</h4>
          <span class="inspector-bank">Bank: ${escapeHtml(data.bank || 'Unknown')}</span>
        </div>

        <div class="inspector-grid">
          <div class="inspector-metric">
            <span class="inspector-metric-label">PageRank</span>
            <span class="inspector-metric-value">${Number(data.pagerank || 0).toFixed(5)}</span>
          </div>
          <div class="inspector-metric">
            <span class="inspector-metric-label">Betweenness</span>
            <span class="inspector-metric-value">${Number(data.betweenness || 0).toFixed(5)}</span>
          </div>
          <div class="inspector-metric">
            <span class="inspector-metric-label">Total Volume</span>
            <span class="inspector-metric-value">${Number(data.total_volume || 0).toLocaleString('en-US', {maximumFractionDigits: 0})}</span>
          </div>
          <div class="inspector-metric">
            <span class="inspector-metric-label">In / Out Degree</span>
            <span class="inspector-metric-value">${data.in_degree || 0} / ${data.out_degree || 0}</span>
          </div>
          <div class="inspector-metric">
            <span class="inspector-metric-label">Flow Asymmetry</span>
            <span class="inspector-metric-value">${Number(data.flow_asymmetry || 0).toFixed(4)}</span>
          </div>
          <div class="inspector-metric">
            <span class="inspector-metric-label">Cycle Rings</span>
            <span class="inspector-metric-value" title="${escapeHtml(data.participating_rings || '')}" style="color:${data.in_cycle ? 'var(--accent-danger)' : 'var(--text-main)'};">${data.cycle_count ? `${data.cycle_count} ring${data.cycle_count > 1 ? 's' : ''}` : 'None'}</span>
          </div>
        </div>

        <button class="btn btn-primary" style="width:100%; justify-content:center; margin-top:0.75rem;" onclick="viewSARReport('${accountId}')">
          Generate SAR Report
        </button>
      </div>
    `;
    document.getElementById('sideInspectorContent').innerHTML = sideHtml;

    // The inspector lives on the Network Graph tab; bring it into view when opened from another tab
    if (state.activeTab && state.activeTab !== 'graphView') {
      document.querySelector('.tab-btn[data-tab="graphView"]')?.click();
    }
    document.getElementById('sideInspectorContent')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Outline the selected account in the graph if it is drawn there
    if (state.svg) {
      state.svg.selectAll('.nodes circle')
        .attr('stroke', d => d.id === accountId ? '#111827' : '#ffffff')
        .attr('stroke-width', d => d.id === accountId ? 3 : 1.5);
    }
  } catch (err) {
    console.error('Error inspecting account:', err);
    showToast('Failed to load account profile', 'error');
  }
}

async function viewSARReport(accountId) {
  try {
    const res = await fetch(`/api/sar/${accountId}`);
    if (!res.ok) {
      showToast('Could not generate SAR for this account', 'error');
      return;
    }
    const sar = await res.json();
    state.currentSar = sar;

    document.getElementById('sarModalSubtitle').textContent = `Filing ID: ${sar.report_id} | Subject: ${sar.primary_suspect_account}`;

    const indicatorsHtml = sar.suspicion_indicators.map(ind => `
      <div class="sar-indicator">
        <strong>[${ind.code}] ${ind.category}</strong>
        <p>${ind.description}</p>
      </div>
    `).join('');

    const txsHtml = (sar.recent_transactions || []).slice(0, 8).map(t => `
      <tr>
        <td>${t.Timestamp || '-'}</td>
        <td class="mono-cell">${(t.Account || '-')} &rarr; ${(t['Account.1'] || '-')}</td>
        <td><strong>${Number(t['Amount Paid'] || t['Amount Received'] || 0).toLocaleString('en-US', {maximumFractionDigits: 0})}</strong> <span style="color:var(--text-muted); font-size:0.7rem;">${escapeHtml(t['Payment Currency'] || '')}</span></td>
        <td>${t['Payment Format'] || '-'}</td>
        <td><span class="badge ${t['Is Laundering'] ? 'danger' : ''}">${t['Is Laundering'] ? 'Flagged' : 'Normal'}</span></td>
      </tr>
    `).join('');

    document.getElementById('sarModalContent').innerHTML = `
      <div class="sar-kpi-row">
        <div class="sar-kpi">
          <div class="sar-kpi-label">ML RISK SCORE</div>
          <div class="sar-kpi-value danger">${(sar.ml_risk_score * 100).toFixed(1)}%</div>
        </div>
        <div class="sar-kpi">
          <div class="sar-kpi-label">THREAT LEVEL</div>
          <div class="sar-kpi-value">${sar.threat_level}</div>
        </div>
        <div class="sar-kpi">
          <div class="sar-kpi-label">ANALYZED VOLUME</div>
          <div class="sar-kpi-value">${sar.total_volume_analyzed?.toLocaleString('en-US', {maximumFractionDigits: 0}) || '0'}</div>
        </div>
        <div class="sar-kpi">
          <div class="sar-kpi-label">CYCLE RINGS</div>
          <div class="sar-kpi-value cyan">${sar.graph_metrics?.cycle_participation_count || 0}</div>
        </div>
      </div>

      <div class="sar-section">
        <h4>Suspicion Indicators</h4>
        ${indicatorsHtml}
      </div>

      <div class="sar-narrative">
        <h4>Investigative Narrative</h4>
        <p>${sar.forensic_narrative}</p>
      </div>

      <div class="sar-section">
        <h4>Forensic Transaction Ledger</h4>
        <table class="data-table">
          <thead><tr><th>Timestamp</th><th>Transfer Flow</th><th>Amount</th><th>Format</th><th>Flag</th></tr></thead>
          <tbody>${txsHtml || '<tr><td colspan="5">No transactions</td></tr>'}</tbody>
        </table>
      </div>
    `;

    document.getElementById('sarModal').classList.add('open');
  } catch (err) {
    console.error('Error opening SAR:', err);
    showToast('Error generating SAR report', 'error');
  }
}

// 9. Cypher Console
function setupCypherConsole() {
  const select = document.getElementById('cypherTemplateSelect');
  const input = document.getElementById('cypherQueryInput');
  const btn = document.getElementById('btnRunCypher');
  const btnTable = document.getElementById('btnViewCypherTable');
  const btnJson = document.getElementById('btnViewCypherJson');
  const tableContainer = document.getElementById('cypherTableContainer');
  const jsonPre = document.getElementById('cypherJsonResult');

  input.value = CYPHER_TEMPLATES.cycle;

  select?.addEventListener('change', (e) => {
    const val = e.target.value;
    if (CYPHER_TEMPLATES[val]) {
      input.value = CYPHER_TEMPLATES[val];
    }
  });

  btnTable?.addEventListener('click', () => {
    btnTable.classList.add('active');
    btnJson?.classList.remove('active');
    if (tableContainer) tableContainer.style.display = 'block';
    if (jsonPre) jsonPre.style.display = 'none';
  });

  btnJson?.addEventListener('click', () => {
    btnJson.classList.add('active');
    btnTable?.classList.remove('active');
    if (tableContainer) tableContainer.style.display = 'none';
    if (jsonPre) jsonPre.style.display = 'block';
  });

  btn?.addEventListener('click', async () => {
    const query = input.value.trim();
    if (!query) return;

    const statusEl = document.getElementById('cypherResultStatus');
    statusEl.textContent = 'Executing...';
    statusEl.className = 'badge';

    try {
      const res = await fetch('/api/cypher', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      const data = await res.json();
      statusEl.textContent = `${data.count || 0} records — ${data.mode || 'Embedded'}`;
      statusEl.className = data.status === 'success' ? 'badge success' : 'badge danger';
      // Update JSON view
      if (jsonPre) jsonPre.textContent = JSON.stringify(data, null, 2);

      // Render Table view
      if (tableContainer) {
        if (data.records && data.records.length > 0) {
          const firstRec = data.records[0];
          const keys = typeof firstRec === 'object' && firstRec !== null ? Object.keys(firstRec) : ['Value'];
          let ths = keys.map(k => `<th>${escapeHtml(k)}</th>`).join('');
          let trs = data.records.map(rec => {
            let tds = keys.map(k => {
              let val = rec[k];
              if (Array.isArray(val)) val = val.join(' -> ');
              else if (typeof val === 'object' && val !== null) val = JSON.stringify(val);
              return `<td>${escapeHtml(String(val ?? ''))}</td>`;
            }).join('');
            return `<tr>${tds}</tr>`;
          }).join('');

          tableContainer.innerHTML = `
            <table class="data-table" style="width:100%; font-size:0.78rem;">
              <thead><tr>${ths}</tr></thead>
              <tbody>${trs}</tbody>
            </table>
          `;
        } else {
          tableContainer.innerHTML = `<div class="empty-state" style="padding:1.5rem;"><p>${data.message || 'No records returned.'}</p></div>`;
        }
      }
    } catch (err) {
      statusEl.textContent = 'Error';
      statusEl.className = 'badge danger';
      if (jsonPre) jsonPre.textContent = String(err);
      if (tableContainer) tableContainer.innerHTML = `<div class="alert danger">${escapeHtml(String(err))}</div>`;
    }
  });
}

// 10. Event Listeners & SAR Export
function setupEventListeners() {
  document.getElementById('btnCloseModal')?.addEventListener('click', () => {
    document.getElementById('accountModal').classList.remove('open');
  });
  document.getElementById('btnModalClose')?.addEventListener('click', () => {
    document.getElementById('accountModal').classList.remove('open');
  });
  document.getElementById('btnCloseSarModal')?.addEventListener('click', () => {
    document.getElementById('sarModal').classList.remove('open');
  });
  document.getElementById('btnCloseSarFooter')?.addEventListener('click', () => {
    document.getElementById('sarModal').classList.remove('open');
  });

  // Reload IBM Dataset
  document.getElementById('btnReloadData')?.addEventListener('click', async () => {
    const btn = document.getElementById('btnReloadData');
    const overlay = document.getElementById('loadingOverlay');
    btn.disabled = true;
    if (overlay) {
      overlay.querySelector('.spinner-text h3').textContent = 'RELOADING IBM DATASET';
      overlay.querySelector('.spinner-text p').textContent = 'Re-sampling IBM HI-Small data & retraining graph ensemble...';
      overlay.classList.remove('fade-out');
      overlay.style.display = 'flex';
    }
    showToast('Reloading IBM HI-Small AML dataset...', 'info');

    try {
      const res = await fetch('/api/reload-dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_size: 30000 })
      });
      const data = await res.json();
      showToast(data.message || 'Dataset reloaded successfully', 'success');

      // Refresh all state
      await loadOverview();
      await Promise.all([
        loadGraphData(),
        loadCycles(),
        loadLeads(),
        loadMetrics(),
        loadAblation(),
        loadShap()
      ]);
    } catch (err) {
      showToast('Error reloading dataset: ' + err, 'error');
    } finally {
      btn.disabled = false;
      if (overlay) {
        overlay.classList.add('fade-out');
        setTimeout(() => { overlay.style.display = 'none'; }, 500);
      }
    }
  });

  // Sync to Neo4j
  document.getElementById('btnSyncNeo4j')?.addEventListener('click', async () => {
    const btn = document.getElementById('btnSyncNeo4j');
    btn.textContent = 'Syncing...';
    btn.disabled = true;
    try {
      const res = await fetch('/api/neo4j/sync', { method: 'POST' });
      const data = await res.json();
      showToast(data.message || `Synced ${data.synced_rows || 0} rows`, data.status === 'success' ? 'success' : 'warning');
      await loadOverview();
    } catch (err) {
      showToast('Error syncing to Neo4j: ' + err, 'error');
    } finally {
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg> Sync to Neo4j`;
      btn.disabled = false;
    }
  });

  // Download SAR JSON
  document.getElementById('btnDownloadSarJson')?.addEventListener('click', () => {
    if (!state.currentSar) return;
    const blob = new Blob([JSON.stringify(state.currentSar, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${state.currentSar.report_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('SAR dossier downloaded', 'success');
  });

  // Lead search
  document.getElementById('leadSearch')?.addEventListener('input', (e) => {
    const term = e.target.value.toLowerCase();
    const filtered = state.leads.filter(l =>
      (l.account_from && l.account_from.toLowerCase().includes(term)) ||
      (l.account_to && l.account_to.toLowerCase().includes(term)) ||
      (l.typology && l.typology.toLowerCase().includes(term)) ||
      (l.from_bank && l.from_bank.toLowerCase().includes(term))
    );
    renderLeadsTable(filtered);
  });

  // Lead category filter
  document.getElementById('leadFilterCategory')?.addEventListener('change', (e) => {
    const cat = e.target.value;
    let filtered = state.leads;
    if (cat === 'HIGH_RISK') {
      filtered = state.leads.filter(l => l.risk_score >= 0.85);
    } else if (cat === 'CYCLES') {
      filtered = state.leads.filter(l => l.src_in_cycle || l.dst_in_cycle);
    } else if (cat === 'STRUCTURING') {
      filtered = state.leads.filter(l => (l.amount || 0) < 10000);
    }
    renderLeadsTable(filtered);
  });

  document.getElementById('btnExportSAR')?.addEventListener('click', () => {
    if (state.leads.length > 0) {
      const acc = state.leads[0].account_from || state.leads[0].Account;
      if (acc) viewSARReport(acc);
    } else {
      showToast('No accounts available to export', 'warning');
    }
  });

  // Keep the active filter chip in sync with the highlight mode
  document.querySelectorAll('.btn-chip').forEach(chip => chip.addEventListener('click', () => {
    document.querySelectorAll('.btn-chip').forEach(c => c.classList.toggle('active', c === chip));
  }));

  document.getElementById('btnHighlightCycles')?.addEventListener('click', () => {
    if (!state.svg) return;
    state.svg.selectAll('.nodes circle')
      .transition().duration(300)
      .attr('opacity', d => d.inCycle ? 1 : 0.08);
    state.svg.selectAll('.links line')
      .transition().duration(300)
      .attr('opacity', d => d.isLaundering === 1 ? 1 : 0.03);
  });

  document.getElementById('btnHighlightBridges')?.addEventListener('click', () => {
    if (!state.svg) return;
    state.svg.selectAll('.nodes circle')
      .transition().duration(300)
      .attr('opacity', d => d.betweenness > 0.02 ? 1 : 0.08);
    state.svg.selectAll('.links line')
      .transition().duration(300)
      .attr('opacity', 0.15);
  });

  document.getElementById('btnHighlightAll')?.addEventListener('click', () => {
    if (!state.svg) return;
    state.svg.selectAll('.nodes circle')
      .transition().duration(300)
      .attr('opacity', 1);
    state.svg.selectAll('.links line')
      .transition().duration(300)
      .attr('opacity', d => d.isLaundering === 1 ? 0.9 : 0.35);
  });

  document.getElementById('btnResetGraph')?.addEventListener('click', () => {
    if (state.svg) {
      state.svg.transition().duration(500).call(d3.zoom().transform, d3.zoomIdentity);
    }
  });
}

// Toast Notification System
function showToast(message, type = 'info') {
  const existing = document.querySelector('.toast-notification');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = `toast-notification toast-${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);

  requestAnimationFrame(() => toast.classList.add('show'));
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}
