"""Generate the static evaluation site from benchmark data.

Usage
-----
    python generate_site.py

Reads data from the isdin-search-api repo and generates all HTML pages
into the current directory (isdin-search-eval/).
"""

import json
import re
import shutil
from pathlib import Path

# Paths
SITE_DIR = Path(__file__).parent
API_REPO = Path(__file__).parent.parent.parent / 'isdin-search-api'
BENCHMARK_DIR = API_REPO / 'data' / 'benchmark' / 'v1'
EMBEDDINGS_DIR = API_REPO / 'output' / 'embeddings'

# Output dirs
(SITE_DIR / 'benchmark').mkdir(exist_ok=True)
(SITE_DIR / 'experiments').mkdir(exist_ok=True)
(SITE_DIR / 'embeddings').mkdir(exist_ok=True)
(SITE_DIR / 'results').mkdir(exist_ok=True)
(SITE_DIR / 'assets').mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_runs() -> dict[str, list[dict]]:
    """Load pre-normalized runs from data/ directory."""
    runs = {}
    data_dir = SITE_DIR / 'data'
    for index in ('products', 'events', 'cross-index'):
        f = data_dir / f'{index}.json'
        if f.exists():
            runs[index] = json.loads(f.read_text())
            print(f'  {index}: {len(runs[index])} runs')
        else:
            runs[index] = []
            print(f'  {index}: no data file')
    return runs


def normalize_run(data: dict, filename: str, index: str) -> dict | None:
    """Normalize a run into a consistent format."""
    sync_label = data.get('sync_label', '')
    label = data.get('label', filename.replace('.json', ''))
    config = data.get('config', {})
    metrics = data.get('metrics', {})

    # Derive model
    if 'emb1' in sync_label or 'emb1' in filename:
        model = 'gemini-001'
    elif 'emb2' in sync_label or 'emb2' in filename:
        model = 'gemini-2'
    elif 'products-api' in filename:
        model = 'products-api'
    elif 'search-api' in filename:
        model = 'search-api'
    else:
        model = 'gemini-2'

    # Derive multimodal approach
    if 'multimodal-caption' in sync_label:
        multimodal = 'image+caption'
    elif 'multimodal' in sync_label:
        multimodal = 'image'
    elif 'caption-short' in sync_label or 'caption-short' in filename:
        multimodal = 'caption-short'
    elif 'caption' in sync_label:
        multimodal = 'caption-long'
    else:
        multimodal = 'none'

    # Derive popularity normalization
    if 'linear' in sync_label:
        pop_norm = 'linear'
    elif 'log' in sync_label and 'log' not in 'catalog':
        pop_norm = 'log'
    else:
        pop_norm = 'percent_rank'

    # Mode
    mode = config.get('mode', '')
    if not mode:
        if 'semantic' in filename:
            mode = 'semantic'
        elif 'keyword' in filename:
            mode = 'keyword'
        elif 'hybrid' in filename:
            mode = 'hybrid'
        elif 'unified' in filename:
            mode = 'unified'
        elif 'rerank' in filename:
            mode = 'rerank'
        else:
            mode = 'hybrid'

    # Alpha (semantic weight)
    alpha = config.get('semantic_weight')
    if alpha is None:
        sw_match = re.search(r'sw([\d.]+)', filename)
        if sw_match:
            alpha = float(sw_match.group(1))

    # Beta (popularity boost)
    beta = config.get('popularity_boost')
    if beta is None:
        pb_match = re.search(r'pb([\d.]+)', filename)
        if pb_match:
            beta = float(pb_match.group(1))

    # Field boosts
    fb = config.get('field_boosts', '')
    if not fb:
        if 'fb-flat' in filename:
            fb = 'flat'
        elif 'fb-default' in filename:
            fb = 'default'
        elif 'fb-title_dominant' in filename or 'title_dominant' in filename:
            fb = 'title_dominant'
        elif 'fb-no_ingred' in filename:
            fb = 'no_ingredients'
        elif 'fb-boost_ing' in filename:
            fb = 'boost_ingredients'
        elif 'fb-name_dom' in filename:
            fb = 'name_dominant'

    # Rename "default" based on when the run was made
    if fb == 'default':
        # emb2-pctrank, emb2-linear, emb2-log ran with old weighted defaults
        if sync_label in ('emb2-pctrank', 'emb2-linear', 'emb2-log'):
            fb = 'intuition'
        else:
            fb = 'flat'

    # Variant (for cross-index)
    variant = ''
    if index == 'cross-index':
        if 'unified_both' in filename or 'unified_both' in label:
            variant = 'both'
        elif 'unified_products' in filename:
            variant = 'products-only'
        elif 'unified_events' in filename:
            variant = 'events-only'
        elif 'rerank_default' in filename:
            variant = 'default'
        elif 'rerank_pure' in filename:
            variant = 'pure_semantic'
        elif 'rerank_aggressive' in filename:
            variant = 'aggressive_pop'

    # For events v2 runs
    if 'events-v2' in filename:
        if 'flat' in filename:
            fb = 'flat'
        elif 'title_dominant' in filename:
            fb = 'title_dominant'
        elif 'weighted' in filename:
            fb = 'weighted'
        elif 'title3_notype' in filename:
            fb = 'title3_notype'

    return {
        'id': 0,
        'filename': filename,
        'label': label,
        'model': model,
        'multimodal': multimodal,
        'pop_norm': pop_norm,
        'mode': mode,
        'alpha': alpha,
        'beta': beta,
        'field_boosts': fb or '-',
        'variant': variant,
        'metrics': metrics,
        'n_queries': data.get('n_queries', 0),
        'results': data.get('results', []),
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

CSS = '''
:root {
  --bg: #f8fafc; --card: #ffffff; --text: #1e293b; --muted: #64748b;
  --border: #e2e8f0; --accent: #3b82f6; --good: #10b981; --warn: #f59e0b;
  --danger: #ef4444; --hover: #f1f5f9; --selected: #eff6ff;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --mono: 'SF Mono', 'JetBrains Mono', monospace;
  --radius: 12px; --shadow: 0 1px 3px rgba(0,0,0,0.04);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--font); background: var(--bg); color: var(--text); font-size: 13px; line-height: 1.5; }
'''


def generate_index():
    """Generate the main portal page."""
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ISDIN Search Evaluation — TFM</title>
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="layout">
  <nav class="sidebar">
    <div class="logo">
      <h1>ISDIN Search</h1>
      <span class="subtitle">Evaluation Portal</span>
    </div>
    <ul class="nav-sections">
      <li class="nav-section">
        <h3>Benchmark</h3>
        <ul>
          <li><a href="benchmark/products.html" target="content">Products (114 queries)</a></li>
          <li><a href="benchmark/events.html" target="content">Events (22 queries)</a></li>
          <li><a href="benchmark/cross-index.html" target="content">Cross-Index (5 queries)</a></li>
        </ul>
      </li>
      <li class="nav-section">
        <h3>Experiments</h3>
        <ul>
          <li><a href="experiments/runs.html" target="content">All Runs</a></li>
        </ul>
      </li>
      <li class="nav-section">
        <h3>Embedding Space Analysis</h3>
        <ul>
          <li><a href="embeddings/gemini-2-text.html" target="content">Gemini 2 (text)</a></li>
          <li><a href="embeddings/gemini-2-images.html" target="content">Gemini 2 (multimodal)</a></li>
          <li><a href="embeddings/gemini-2-caption.html" target="content">Gemini 2 (caption)</a></li>
          <li><a href="embeddings/gemini-2-images-caption.html" target="content">Gemini 2 (multi+caption)</a></li>
          <li><a href="embeddings/gemini-001-text.html" target="content">Gemini 001 (baseline)</a></li>
          <li><a href="embeddings/gemini-2-text-3072d.html" target="content">Gemini 2 (3072d)</a></li>
          <li><a href="embeddings/titan-v2-text-1024d.html" target="content">Titan v2 (text)</a></li>
          <li><a href="embeddings/titan-v2-caption-1024d.html" target="content">Titan v2 (caption)</a></li>
          <li class="divider"></li>
          <li><a href="embeddings/gender-bias.html" target="content">Gender Bias Analysis</a></li>
        </ul>
      </li>
      <li class="nav-section">
        <h3>Results</h3>
        <ul>
          <li><a href="results/comparison.html" target="content">System Comparison</a></li>
          <li><a href="results/summary.html" target="content">Latency</a></li>
        </ul>
      </li>
    </ul>
    <div class="footer">
      <p>Ignasi Cervero &middot; UPC 2026</p>
      <p class="muted">MSc Data Science — TFM</p>
    </div>
  </nav>
  <main class="content">
    <iframe name="content" id="content-frame" src="results/comparison.html"></iframe>
  </main>
</div>
</body>
</html>'''
    (SITE_DIR / 'index.html').write_text(html)
    print('  index.html')


def generate_benchmark():
    """Copy benchmark viewers."""
    viewers = {
        'products': BENCHMARK_DIR / 'products' / 'ground-truth-viewer.html',
        'events': BENCHMARK_DIR / 'events' / 'ground-truth-viewer.html',
        'cross-index': BENCHMARK_DIR / 'cross-index' / 'ground-truth-viewer.html',
    }
    for name, src in viewers.items():
        if src.exists():
            shutil.copy2(src, SITE_DIR / 'benchmark' / f'{name}.html')
            print(f'  benchmark/{name}.html')


def generate_embeddings():
    """Copy UMAP visualizations."""
    mapping = {
        'gemini-2-text': 'gemini-2-text/umap_es.html',
        'gemini-2-images': 'gemini-2-images/umap_es.html',
        'gemini-2-caption': 'gemini-2-caption/umap_es.html',
        'gemini-2-images-caption': 'gemini-2-images-caption/umap_es.html',
        'gemini-001-text': 'gemini-001-text/umap_es.html',
        'gemini-2-text-3072d': 'gemini-2-text-3072d/umap_es.html',
        'titan-v2-text-1024d': 'titan-v2-text-1024d/umap_es.html',
        'titan-v2-caption-1024d': 'titan-v2-caption-1024d/umap_es.html',
        'gender-bias': 'gender_bias_umap.html',
    }
    for name, rel_path in mapping.items():
        src = EMBEDDINGS_DIR / rel_path
        if src.exists():
            shutil.copy2(src, SITE_DIR / 'embeddings' / f'{name}.html')
            print(f'  embeddings/{name}.html')


def generate_comparison(runs: dict[str, list[dict]]):
    """Generate the 3-system comparison landing page."""
    products = runs.get('products', [])

    # Find the 3 key runs (ordered: baseline first → production last)
    run_map = {}
    for r in products:
        l = r.get('label', '')
        if 'products-api' in l:
            run_map['products-api'] = r
        elif 'hybrid-baseline' in l:
            run_map['baseline'] = r
        elif 'multimodal-caption_products_hybrid_sw0.95_pb0.05' in l:
            run_map['production'] = r

    systems = [
        ('Products API', 'Algolia lexical search', run_map.get('products-api')),
        ('Intuition Baseline', 'gemini-001, α=0.6, β=0.2, weighted boosts', run_map.get('baseline')),
        ('ISDIN Search Service', 'gemini-2, multimodal+caption, α=0.95, β=0.05', run_map.get('production')),
    ]
    systems = [(n, d, r) for n, d, r in systems if r is not None]

    if len(systems) < 3:
        print(f'  WARNING: only found {len(systems)}/3 systems for comparison')

    metrics_keys = ['ndcg@5', 'ndcg@10', 'f1@5', 'f1@10', 'mrr']

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>System Comparison — ISDIN Search Evaluation</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
{CSS}
body {{ padding: 40px; }}
h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 4px; }}
.subtitle {{ font-size: 14px; color: var(--muted); margin-bottom: 32px; }}
.systems {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 32px; }}
.system-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; text-align: center; }}
.system-card h3 {{ font-size: 15px; margin-bottom: 4px; }}
.system-card .desc {{ font-size: 11px; color: var(--muted); margin-bottom: 16px; }}
.system-card .score {{ font-size: 36px; font-weight: 700; color: var(--accent); }}
.system-card .score-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; margin-top: 4px; }}
.chart-container {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; margin-bottom: 24px; }}
.chart-container h3 {{ font-size: 15px; margin-bottom: 16px; }}
.chart-container canvas {{ max-height: 300px; }}
.sub-tabs {{ display: flex; gap: 0; border-bottom: 1px solid var(--border); margin: 24px 0 16px; }}
.sub-tab {{ padding: 10px 20px; cursor: pointer; border-bottom: 2px solid transparent; font-size: 13px; color: var(--muted); font-weight: 500; }}
.sub-tab:hover {{ color: var(--text); }}
.sub-tab.active {{ border-bottom-color: var(--accent); color: var(--accent); }}
.tab-panel {{ margin-bottom: 24px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }}
th {{ text-align: left; padding: 10px 14px; font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; border-bottom: 2px solid var(--border); }}
td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); }}
.metric {{ font-family: var(--mono); font-size: 12px; }}
.improvement {{ color: var(--good); font-size: 11px; font-weight: 600; }}
</style>
</head>
<body>
<h1>System Comparison</h1>
<p class="subtitle">Three stages of the ISDIN product search system, evaluated on 114 queries.</p>

<div class="systems">'''

    for name, desc, run in systems:
        ndcg = run['metrics'].get('ndcg@10', 0)
        html += f'''<div class="system-card">
  <h3>{name}</h3>
  <div class="desc">{desc}</div>
  <div class="score">{ndcg:.3f}</div>
  <div class="score-label">NDCG@10</div>
</div>'''

    html += '''</div>

<div class="chart-container">
  <h3>Metric Comparison</h3>
  <canvas id="comp-chart"></canvas>
</div>

<table>
<thead><tr><th>Metric</th>'''

    for name, _, _ in systems:
        html += f'<th>{name}</th>'
    html += '<th>Improvement</th></tr></thead><tbody>'

    for mk in metrics_keys:
        html += f'<tr><td><strong>{mk.upper()}</strong></td>'
        vals = []
        for _, _, run in systems:
            v = run['metrics'].get(mk, 0)
            vals.append(v)
            html += f'<td class="metric">{v:.4f}</td>'
        if len(vals) >= 2 and vals[0] > 0:
            improvement = (vals[-1] - vals[0]) / vals[0] * 100
            sign = '+' if improvement >= 0 else ''
            html += f'<td class="improvement">{sign}{improvement:.0f}%</td>'
        else:
            html += '<td>-</td>'
        html += '</tr>'

    html += '</tbody></table></div>'

    # Prepare per-query data for JS (merge items from detail file)
    detail_file = SITE_DIR / 'data' / 'products-detail.json'
    detail_data = json.loads(detail_file.read_text()) if detail_file.exists() else {}

    all_per_query = {}
    for name, _, run in systems:
        pq = run.get('per_query', [])
        run_detail = detail_data.get(str(run['id']), {})
        # Merge items into per_query
        for q in pq:
            qid = q.get('query_id')
            if qid and qid in run_detail:
                q['items'] = run_detail[qid]
        all_per_query[name] = pq

    cat_names_map = {
        'A': 'Sinónimo semántico', 'B': 'Problema / skin concern', 'C': 'Ingrediente',
        'D': 'Marca / sublínea', 'E': 'Atributo (SPF, formato)', 'F': 'Broad / intención vaga',
        'G': 'Multilingüe', 'H': 'Prefijo / autocompletado', 'I': 'Typo / error ortográfico',
        'J': 'Zero-result', 'K': 'Parte del cuerpo', 'L': 'Caso de uso / ocasión',
        'M': 'Multi-intent', 'N': 'Nombre completo', 'O': 'Conversacional',
        'P': 'Refill vs completo', 'Q': 'Visual / multimodal',
    }

    # By Category table (static)
    html += '<div id="panel-category" class="tab-panel" style="display:none"><div class="chart-container"><h3>NDCG@10 by Category</h3>'
    html += '<div style="overflow-x:auto"><table><thead><tr><th>Category</th><th>Name</th>'
    for name, _, _ in systems:
        html += f'<th>{name}</th>'
    html += '</tr></thead><tbody>'

    all_cats = set()
    system_cats = []
    for _, _, run in systems:
        cats = {}
        for q in run.get('per_query', []):
            cat = q.get('category', '?')
            all_cats.add(cat)
            if cat not in cats:
                cats[cat] = []
            cats[cat].append(q.get('ndcg@10', 0))
        system_cats.append(cats)

    for cat in sorted(all_cats):
        html += f'<tr style="cursor:pointer" onclick="showByQuery(\'{cat}\')"><td><strong>{cat}</strong></td><td style="font-size:11px;color:var(--muted)">{cat_names_map.get(cat, "")}</td>'
        for cats in system_cats:
            vals = cats.get(cat, [])
            avg = sum(vals) / len(vals) if vals else 0
            html += f'<td class="metric">{avg:.3f}</td>'
        html += '</tr>'
    html += '</tbody></table></div></div></div>'

    # By Query panel (dynamic via JS)
    html += '<div id="panel-query" class="tab-panel" style="display:none"></div>'

    # Query Detail panel (dynamic via JS)
    html += '<div id="panel-detail" class="tab-panel" style="display:none"></div>'

    # Tabs navigation (insert before chart)
    tabs_html = '''<div class="sub-tabs" style="margin:24px 0 16px">
  <div class="sub-tab active" onclick="switchCompTab('overview')">Overview</div>
  <div class="sub-tab" onclick="switchCompTab('category')">By Category</div>
  <div class="sub-tab" onclick="switchCompTab('query')">By Query</div>
  <div class="sub-tab" onclick="switchCompTab('detail')">Query Detail</div>
</div>
<div id="panel-overview">'''

    # Insert tabs before the chart
    html = html.replace('<div class="chart-container">\n  <h3>Metric Comparison</h3>', tabs_html + '<div class="chart-container">\n  <h3>Metric Comparison</h3>')
    # Close overview panel before category
    html = html.replace('<div id="panel-category"', '</div><div id="panel-category"')

    # Embed per-query data
    per_query_js = json.dumps(all_per_query, ensure_ascii=False)
    system_names_js = json.dumps([n for n, _, _ in systems])
    cat_names_js = json.dumps(cat_names_map, ensure_ascii=False)

    # Chart data
    colors = ['#ef4444', '#f59e0b', '#3b82f6']
    html += f'''
<script>
new Chart(document.getElementById('comp-chart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps([mk.upper() for mk in metrics_keys])},
    datasets: ['''

    for i, (name, _, run) in enumerate(systems):
        vals = [run['metrics'].get(mk, 0) for mk in metrics_keys]
        html += f'''{{
      label: '{name}',
      data: {json.dumps(vals)},
      backgroundColor: '{colors[i]}80',
      borderColor: '{colors[i]}',
      borderWidth: 1,
    }},'''

    html += f'''],
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    scales: {{ y: {{ beginAtZero: true, max: 1 }} }},
    plugins: {{ legend: {{ position: 'top' }} }},
  }},
}});

// --- Tab switching and dynamic content ---
const PER_QUERY = {per_query_js};
const SYSTEM_NAMES = {system_names_js};
const CAT_NAMES = {cat_names_js};
const COLORS = ['#ef4444', '#f59e0b', '#3b82f6'];
let currentCatFilter = null;
let currentQueryId = null;

function switchCompTab(tab) {{
  document.querySelectorAll('.sub-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel, #panel-overview').forEach(p => p.style.display = 'none');
  event.target.classList.add('active');
  document.getElementById('panel-' + tab).style.display = 'block';
  if (tab === 'query') renderByQuery();
  if (tab === 'detail') renderQueryDetail();
}}

function showByQuery(cat) {{
  currentCatFilter = cat;
  switchCompTab('query');
  document.querySelectorAll('.sub-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.sub-tab')[2].classList.add('active');
}}

function renderByQuery() {{
  const panel = document.getElementById('panel-query');
  let html = '';
  if (currentCatFilter) {{
    html += `<div style="margin-bottom:12px"><button onclick="currentCatFilter=null;renderByQuery()" style="cursor:pointer;padding:4px 12px;border:1px solid #e2e8f0;border-radius:6px;background:#fff;font-size:12px">\\u2190 All categories</button> <strong>${{currentCatFilter}}: ${{CAT_NAMES[currentCatFilter] || ''}}</strong></div>`;
  }}
  html += '<p style="font-size:11px;color:#64748b;margin-bottom:8px">Click a query to see returned items</p>';
  html += '<div style="overflow-x:auto"><table><thead><tr><th>ID</th><th>Query</th>';
  SYSTEM_NAMES.forEach(n => html += `<th>${{n}}</th>`);
  html += '</tr></thead><tbody>';

  // Collect all query IDs
  const allQids = new Set();
  SYSTEM_NAMES.forEach(n => PER_QUERY[n].forEach(q => {{
    if (!currentCatFilter || q.category === currentCatFilter) allQids.add(q.query_id);
  }}));
  const sortedQids = [...allQids].sort((a, b) => a.localeCompare(b, undefined, {{numeric: true}}));

  sortedQids.forEach(qid => {{
    const qInfo = SYSTEM_NAMES.map(n => PER_QUERY[n].find(q => q.query_id === qid)).find(x => x);
    html += `<tr style="cursor:pointer" onclick="currentQueryId='${{qid}}';switchCompTab('detail');document.querySelectorAll('.sub-tab').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.sub-tab')[3].classList.add('active')">`;
    html += `<td>${{qid}}</td><td>${{qInfo?.query || ''}}</td>`;
    SYSTEM_NAMES.forEach(n => {{
      const q = PER_QUERY[n].find(x => x.query_id === qid);
      const v = q ? (q['ndcg@10'] || 0) : 0;
      html += `<td class="metric">${{v.toFixed(3)}}</td>`;
    }});
    html += '</tr>';
  }});
  html += '</tbody></table></div>';
  panel.innerHTML = html;
}}

function renderQueryDetail() {{
  const panel = document.getElementById('panel-detail');
  const allQids = new Set();
  SYSTEM_NAMES.forEach(n => PER_QUERY[n].forEach(q => allQids.add(q.query_id)));
  const sortedQids = [...allQids].sort((a, b) => a.localeCompare(b, undefined, {{numeric: true}}));

  let html = '<div style="margin-bottom:16px;display:flex;align-items:center;gap:12px">';
  html += '<label style="font-size:11px;font-weight:600;color:#64748b;text-transform:uppercase">Query:</label>';
  html += `<select style="font-size:13px;padding:6px 10px;border:1px solid #e2e8f0;border-radius:6px;min-width:300px" onchange="currentQueryId=this.value;renderQueryDetail()">`;
  if (!currentQueryId) html += '<option value="">-- Select --</option>';
  sortedQids.forEach(qid => {{
    const q = SYSTEM_NAMES.map(n => PER_QUERY[n].find(x => x.query_id === qid)).find(x => x);
    html += `<option value="${{qid}}" ${{qid === currentQueryId ? 'selected' : ''}}>${{qid}}: ${{q?.query || ''}}</option>`;
  }});
  html += '</select>';
  if (currentQueryId) {{
    const idx = sortedQids.indexOf(currentQueryId);
    if (idx > 0) html += `<button onclick="currentQueryId='${{sortedQids[idx-1]}}';renderQueryDetail()" style="cursor:pointer;padding:4px 12px;border:1px solid #e2e8f0;border-radius:6px;background:#fff;font-size:12px">\\u2190 Prev</button>`;
    if (idx < sortedQids.length-1) html += `<button onclick="currentQueryId='${{sortedQids[idx+1]}}';renderQueryDetail()" style="cursor:pointer;padding:4px 12px;border:1px solid #e2e8f0;border-radius:6px;background:#fff;font-size:12px">Next \\u2192</button>`;
  }}
  html += '</div>';

  if (!currentQueryId) {{
    html += '<div style="padding:40px;text-align:center;color:#64748b">Select a query above</div>';
  }} else {{
    const qInfo = SYSTEM_NAMES.map(n => PER_QUERY[n].find(x => x.query_id === currentQueryId)).find(x => x);
    html += `<h3 style="margin-bottom:4px">${{currentQueryId}}: "${{qInfo?.query || ''}}"</h3>`;
    html += `<div style="color:#64748b;font-size:12px;margin-bottom:16px">Category: ${{qInfo?.category || '?'}} — ${{CAT_NAMES[qInfo?.category] || ''}}</div>`;

    // Metrics per system
    html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:20px">';
    SYSTEM_NAMES.forEach((n, i) => {{
      const q = PER_QUERY[n].find(x => x.query_id === currentQueryId);
      const ndcg = q ? (q['ndcg@10'] || 0).toFixed(4) : '-';
      const f1 = q ? (q['f1@10'] || 0).toFixed(4) : '-';
      const mrr_v = q ? (q['mrr'] || 0).toFixed(4) : '-';
      html += `<div style="border:2px solid ${{COLORS[i]}};border-radius:8px;padding:14px;text-align:center">`;
      html += `<div style="font-weight:600;color:${{COLORS[i]}};margin-bottom:8px">${{n}}</div>`;
      html += `<div style="font-size:11px;color:#64748b">NDCG@10=<strong>${{ndcg}}</strong> | F1@10=<strong>${{f1}}</strong> | MRR=<strong>${{mrr_v}}</strong></div>`;
      html += '</div>';
    }});
    html += '</div>';

    // Items per system (from per_query items if available)
    html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px">';
    SYSTEM_NAMES.forEach((n, i) => {{
      const q = PER_QUERY[n].find(x => x.query_id === currentQueryId);
      const items = q?.items || [];
      html += `<div style="border:2px solid ${{COLORS[i]}};border-radius:8px;padding:16px">`;
      html += `<div style="font-weight:600;color:${{COLORS[i]}};margin-bottom:8px">${{n}}</div>`;
      if (items.length) {{
        html += '<table style="width:100%"><thead><tr><th>#</th><th>Name</th><th>Rel</th></tr></thead><tbody>';
        items.slice(0, 10).forEach((item, j) => {{
          const rel = item.relevance || 0;
          const bg = rel >= 3 ? 'background:#dcfce7' : rel >= 2 ? 'background:#fef9c3' : rel >= 1 ? 'background:#fef3c7' : '';
          html += `<tr style="${{bg}}"><td>${{j+1}}</td><td style="font-size:11px">${{item.name || item.id || '-'}}</td><td>${{rel > 0 ? '<strong>'+rel+'</strong>' : '0'}}</td></tr>`;
        }});
        html += '</tbody></table>';
      }} else {{
        html += '<div style="color:#64748b;font-size:12px">No item detail</div>';
      }}
      html += '</div>';
    }});
    html += '</div>';
  }}
  panel.innerHTML = html;
}}
</script>
</body>
</html>'''

    (SITE_DIR / 'results' / 'comparison.html').write_text(html)
    print('  results/comparison.html')


def generate_summary():
    """Copy the summary page (already hand-crafted)."""
    src = SITE_DIR / 'results' / 'summary.html'
    if src.exists():
        print('  results/summary.html (exists)')


def generate_experiments(runs: dict[str, list[dict]]):
    """Generate the experiments page with all runs."""
    # Embed runs with per-query metrics (no items) + detail files are loaded via fetch
    runs_light = {}
    runs_detail = {}

    for index, index_runs in runs.items():
        runs_light[index] = []
        runs_detail[index] = {}
        for run in index_runs:
            light = {k: v for k, v in run.items() if k != 'per_query'}
            runs_light[index].append(light)
            if run.get('per_query'):
                runs_detail[index][run['id']] = run['per_query']

    # Copy detail files to experiments/ for lazy loading
    data_dir = SITE_DIR / 'data'
    for index in ('products', 'events', 'cross-index'):
        detail_src = data_dir / f'{index}-detail.json'
        detail_dst = SITE_DIR / 'experiments' / f'{index}-detail.json'
        if detail_src.exists():
            shutil.copy2(detail_src, detail_dst)

    runs_js = json.dumps(runs_light, ensure_ascii=False)
    detail_js = json.dumps(runs_detail, ensure_ascii=False)

    html = _build_experiments_html(runs_js, detail_js)
    (SITE_DIR / 'experiments' / 'runs.html').write_text(html)
    print(f'  experiments/runs.html ({sum(len(v) for v in runs.values())} runs)')


def _build_experiments_html(runs_js: str, detail_js: str) -> str:
    """Build the experiments HTML page."""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experiment Runs — ISDIN Search Evaluation</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
{CSS}
.header {{ background: #1e293b; color: #f1f5f9; padding: 18px 28px; display: flex; align-items: center; justify-content: space-between; }}
.header h1 {{ font-size: 18px; font-weight: 600; }}
.header .subtitle {{ font-size: 12px; color: #94a3b8; }}
.tabs {{ display: flex; gap: 0; background: var(--card); border-bottom: 1px solid var(--border); padding: 0 28px; }}
.tab {{ padding: 12px 20px; cursor: pointer; border-bottom: 2px solid transparent; font-weight: 500; color: var(--muted); }}
.tab:hover {{ color: var(--text); background: var(--hover); }}
.tab.active {{ border-bottom-color: var(--accent); color: var(--accent); }}
.content-area {{ padding: 24px 28px; max-width: 1800px; margin: 0 auto; }}
.filters {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-bottom: 16px; }}
.filters label {{ font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; }}
.filters select {{ font-size: 12px; padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--card); }}
.selection-bar {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; font-size: 12px; }}
.selection-bar .count {{ color: var(--muted); }}
.btn {{ padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border); background: var(--card); cursor: pointer; font-size: 12px; font-weight: 500; }}
.btn:hover {{ background: var(--hover); }}
.btn-primary {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
.btn-primary:hover {{ background: #2563eb; }}
.btn-primary:disabled {{ opacity: 0.4; cursor: not-allowed; }}
.btn-danger {{ color: var(--danger); border-color: var(--danger); }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th {{ text-align: left; padding: 8px 10px; font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.3px; border-bottom: 2px solid var(--border); cursor: pointer; white-space: nowrap; }}
th:hover {{ color: var(--accent); }}
td {{ padding: 7px 10px; border-bottom: 1px solid var(--border); }}
tr:hover {{ background: var(--hover); }}
tr.selected {{ background: var(--selected); }}
.metric {{ font-family: var(--mono); font-size: 11px; }}
.metric.best {{ color: var(--good); font-weight: 700; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 600; }}
.badge-hybrid {{ background: #dbeafe; color: #1e40af; }}
.badge-semantic {{ background: #d1fae5; color: #065f46; }}
.badge-keyword {{ background: #fef3c7; color: #92400e; }}
.badge-unified {{ background: #ede9fe; color: #5b21b6; }}
.badge-rerank {{ background: #fce7f3; color: #9d174d; }}
.back-btn {{ margin-bottom: 16px; }}
.detail-view {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; }}
.detail-header {{ margin-bottom: 20px; }}
.detail-header h2 {{ font-size: 18px; margin-bottom: 8px; }}
.config-badges {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }}
.config-badge {{ padding: 4px 10px; border-radius: 6px; font-size: 11px; background: var(--hover); border: 1px solid var(--border); }}
.sub-tabs {{ display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 16px; }}
.sub-tab {{ padding: 8px 16px; cursor: pointer; border-bottom: 2px solid transparent; font-size: 12px; color: var(--muted); }}
.sub-tab.active {{ border-bottom-color: var(--accent); color: var(--accent); }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 20px; }}
.stat-card {{ background: var(--hover); border-radius: 8px; padding: 14px; text-align: center; }}
.stat-card .value {{ font-size: 24px; font-weight: 700; color: var(--accent); }}
.stat-card .label {{ font-size: 10px; color: var(--muted); text-transform: uppercase; margin-top: 2px; }}
.compare-config {{ margin-bottom: 20px; overflow-x: auto; }}
.chart-container {{ position: relative; height: 300px; margin: 16px 0; }}
</style>
</head>
<body>
<div class="header">
  <div><h1>Experiment Runs</h1><div class="subtitle">ISDIN Search — Parameter Optimization</div></div>
</div>
<div class="tabs" id="tabs"></div>
<div class="content-area" id="content-area"></div>

<script>
const RUNS = {runs_js};
const DETAIL = {detail_js};
const INDICES = ['products', 'events', 'cross-index'];
let currentIndex = 'products';
let sortCol = 'ndcg@10';
let sortAsc = false;
let selected = new Set();
let viewMode = 'table'; // table | detail | compare
let detailRunId = null;

const FILTERS = {{}};

function init() {{
  renderTabs();
  renderContent();
}}

function renderTabs() {{
  const tabs = document.getElementById('tabs');
  tabs.innerHTML = INDICES.map(idx => {{
    const count = RUNS[idx] ? RUNS[idx].length : 0;
    const active = idx === currentIndex ? 'active' : '';
    return `<div class="tab ${{active}}" onclick="switchIndex('${{idx}}')">${{idx}} (${{count}})</div>`;
  }}).join('');
}}

function switchIndex(idx) {{
  currentIndex = idx;
  selected.clear();
  viewMode = 'table';
  FILTERS[idx] = FILTERS[idx] || {{}};
  renderTabs();
  renderContent();
}}

function renderContent() {{
  const area = document.getElementById('content-area');
  if (viewMode === 'detail') {{ renderDetail(area); return; }}
  if (viewMode === 'compare') {{ renderCompare(area); return; }}
  renderTable(area);
}}

function renderTable(area) {{
  const runs = getFilteredRuns();
  const total = (RUNS[currentIndex] || []).length;
  const cols = getColumns();
  const bestValues = findBestValues(runs, cols);

  let html = renderFilters();
  html += renderSelectionBar();
  html += `<div style="font-size:12px;color:var(--muted);margin-bottom:8px">Showing ${{runs.length}} / ${{total}} runs</div>`;
  html += '<div style="overflow-x:auto"><table><thead><tr>';
  html += '<th><input type="checkbox" onchange="toggleAll(this)"></th>';
  html += '<th onclick="sort(\\'id\\')">#</th>';
  cols.forEach(([key, label]) => {{
    const arrow = sortCol === key ? (sortAsc ? ' \\u2191' : ' \\u2193') : '';
    html += `<th onclick="sort('${{key}}')">${{label}}${{arrow}}</th>`;
  }});
  html += '</tr></thead><tbody>';

  runs.forEach(run => {{
    const sel = selected.has(run.id) ? 'selected' : '';
    html += `<tr class="${{sel}}" onclick="clickRow(event, ${{run.id}})">`;
    html += `<td><input type="checkbox" ${{selected.has(run.id) ? 'checked' : ''}} ${{!selected.has(run.id) && selected.size >= 10 ? 'disabled' : ''}} onchange="toggleSelect(event, ${{run.id}})"></td>`;
    html += `<td>${{run.id}}</td>`;
    cols.forEach(([key]) => {{
      const val = getColValue(run, key);
      const isMetric = ['ndcg@5','ndcg@10','precision@5','precision@10','mrr'].includes(key);
      const isBest = false;
      if (key === 'mode') {{
        html += `<td><span class="badge badge-${{val}}">${{val.toUpperCase()}}</span></td>`;
      }} else if (isMetric) {{
        html += `<td class="metric ${{isBest ? 'best' : ''}}">${{typeof val === 'number' ? val.toFixed(4) : val}}</td>`;
      }} else {{
        html += `<td>${{val ?? '-'}}</td>`;
      }}
    }});
    html += '</tr>';
  }});

  html += '</tbody></table></div>';
  area.innerHTML = html;
}}

function getColumns() {{
  let base;
  if (currentIndex === 'events') {{
    base = [['model', 'Model'], ['alpha', '&alpha;'], ['field_boosts', 'Field Boosts']];
  }} else if (currentIndex === 'cross-index') {{
    base = [['multimodal', 'Multimodal'], ['mode', 'Mode'], ['variant', 'Variant']];
  }} else {{
    base = [
      ['model', 'Model'], ['multimodal', 'Multimodal'], ['pop_norm', 'Pop. Norm.'],
      ['alpha', '&alpha;'], ['beta', '&beta;'], ['field_boosts', 'Field Boosts'],
    ];
  }}
  base.push(['ndcg@5', 'NDCG@5'], ['ndcg@10', 'NDCG@10'], ['f1@5', 'F1@5'], ['f1@10', 'F1@10'], ['mrr', 'MRR']);
  return base;
}}

function getColValue(run, key) {{
  if (key.includes('@') || key === 'mrr') return run.metrics[key] ?? 0;
  if (run.config && run.config[key] !== undefined) return run.config[key];
  return run[key];
}}

function getFilteredRuns() {{
  let runs = RUNS[currentIndex] || [];
  const filters = FILTERS[currentIndex] || {{}};
  Object.entries(filters).forEach(([key, val]) => {{
    if (val && val !== 'All') {{
      runs = runs.filter(r => String(getColValue(r, key)) === val);
    }}
  }});
  runs.sort((a, b) => {{
    const av = getColValue(a, sortCol);
    const bv = getColValue(b, sortCol);
    if (av == null) return 1;
    if (bv == null) return -1;
    return sortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
  }});
  return runs;
}}

function findBestValues(runs, cols) {{
  const best = {{}};
  const metricCols = cols.filter(([k]) => ['ndcg@5','ndcg@10','precision@5','precision@10','mrr'].includes(k));
  metricCols.forEach(([key]) => {{
    const vals = runs.map(r => getColValue(r, key)).filter(v => typeof v === 'number');
    best[key] = vals.length ? Math.max(...vals) : 0;
  }});
  return best;
}}

function renderFilters() {{
  const runs = RUNS[currentIndex] || [];
  const filterKeys = currentIndex === 'events' ? ['model', 'alpha', 'field_boosts'] : ['model', 'multimodal', 'pop_norm', 'field_boosts', 'alpha', 'beta'];
  const labels = {{model:'Model', multimodal:'Multimodal', pop_norm:'Pop. Norm.', field_boosts:'Field Boosts', alpha:'\\u03b1', beta:'\\u03b2'}};
  const filters = FILTERS[currentIndex] || {{}};

  let html = '<div class="filters">';
  filterKeys.forEach(key => {{
    const values = [...new Set(runs.map(r => String(getColValue(r, key))))].sort();
    html += `<label>${{labels[key] || key}}</label><select onchange="setFilter('${{key}}', this.value)">`;
    html += `<option value="All" ${{!filters[key] || filters[key]==='All' ? 'selected' : ''}}>All</option>`;
    values.forEach(v => {{
      html += `<option value="${{v}}" ${{filters[key]===v ? 'selected' : ''}}>${{v}}</option>`;
    }});
    html += '</select>';
  }});
  html += `<button class="btn" onclick="resetFilters()">Reset filters</button>`;
  html += '</div>';
  return html;
}}

function renderSelectionBar() {{
  return `<div class="selection-bar">
    <span class="count">${{selected.size}}/10 selected</span>
    <button class="btn btn-danger" onclick="selected.clear();renderContent()">Deselect all</button>
    <button class="btn btn-primary" ${{selected.size < 2 ? 'disabled' : ''}} onclick="viewMode='compare';renderContent()">Compare selected (${{selected.size}})</button>
  </div>`;
}}

function sort(col) {{
  if (sortCol === col) sortAsc = !sortAsc;
  else {{ sortCol = col; sortAsc = false; }}
  renderContent();
}}

function setFilter(key, val) {{
  if (!FILTERS[currentIndex]) FILTERS[currentIndex] = {{}};
  FILTERS[currentIndex][key] = val;
  renderContent();
}}

function resetFilters() {{
  FILTERS[currentIndex] = {{}};
  renderContent();
}}

function toggleSelect(e, id) {{
  e.stopPropagation();
  if (selected.has(id)) selected.delete(id);
  else if (selected.size < 10) selected.add(id);
  renderContent();
}}

function toggleAll(checkbox) {{
  const runs = getFilteredRuns();
  if (checkbox.checked) {{
    runs.slice(0, 10).forEach(r => selected.add(r.id));
  }} else {{
    selected.clear();
  }}
  renderContent();
}}

function clickRow(e, id) {{
  if (e.target.type === 'checkbox') return;
  detailRunId = id;
  viewMode = 'detail';
  renderContent();
}}

// --- Category names ---
const CAT_NAMES = {{
  'A': 'Sinónimo semántico', 'B': 'Problema / skin concern', 'C': 'Ingrediente',
  'D': 'Marca / sublínea', 'E': 'Atributo (SPF, formato, tamaño)', 'F': 'Broad / intención vaga',
  'G': 'Multilingüe', 'H': 'Prefijo / autocompletado', 'I': 'Typo / error ortográfico',
  'J': 'Zero-result (irrelevante)', 'K': 'Parte del cuerpo', 'L': 'Caso de uso / ocasión',
  'M': 'Multi-intent (combinada)', 'N': 'Nombre completo de producto', 'O': 'Conversacional / natural',
  'P': 'Refill vs producto completo', 'Q': 'Visual / multimodal',
  'R': 'Intención / actividad', 'S': 'Atributo / tipo de evento', 'T': 'Ubicación / formato',
  'U': 'Temática / marca', 'V': 'Dificultad (typos, parcial, multilingüe)',
  'W': 'Cross-index (productos + eventos)',
}};

let detailSubTab = 'overview';
let detailCatFilter = null;
let selectedQueryId = null;

// --- Detail view ---
function renderDetail(area) {{
  const run = (RUNS[currentIndex] || []).find(r => r.id === detailRunId);
  if (!run) {{ viewMode = 'table'; renderContent(); return; }}
  const detail = (DETAIL[currentIndex] || {{}})[run.id] || [];

  let html = `<button class="btn back-btn" onclick="viewMode='table';detailSubTab='overview';detailCatFilter=null;renderContent()">\\u2190 Back to table</button>`;
  html += '<div class="detail-view">';

  // Header with config
  html += `<div class="detail-header"><h2>Run #${{run.id}}</h2>`;
  html += '<div class="config-badges">';
  const c = run.config || {{}};
  ['model','multimodal','pop_norm','alpha','beta','field_boosts','variant'].forEach(k => {{
    const v = c[k];
    if (v && v !== '-' && v !== '') {{
      const label = k === 'alpha' ? '\\u03b1' : k === 'beta' ? '\\u03b2' : k;
      html += `<span class="config-badge">${{label}}: ${{v}}</span>`;
    }}
  }});
  html += '</div></div>';

  // Metrics overview
  html += '<div class="stats-grid">';
  ['ndcg@5','ndcg@10','f1@5','f1@10','mrr'].forEach(k => {{
    const v = run.metrics[k];
    html += `<div class="stat-card"><div class="value">${{v ? v.toFixed(4) : '-'}}</div><div class="label">${{k.toUpperCase()}}</div></div>`;
  }});
  html += '</div>';

  // Sub-tabs
  html += '<div class="sub-tabs">';
  ['overview','by-category','by-query','query-detail'].forEach(t => {{
    const active = detailSubTab === t ? 'active' : '';
    const label = t === 'overview' ? 'Overview' : t === 'by-category' ? 'By Category' : t === 'by-query' ? 'By Query' : 'Query Detail';
    html += `<div class="sub-tab ${{active}}" onclick="detailSubTab='${{t}}';renderContent()">${{label}}</div>`;
  }});
  html += '</div>';

  if (detail.length && detailSubTab === 'overview') {{
    // Chart placeholder
    html += '<div class="chart-container"><canvas id="detail-chart"></canvas></div>';
  }}

  if (detail.length && detailSubTab === 'by-category') {{
    const cats = {{}};
    detail.forEach(r => {{
      const cat = r.category || r.query_id?.[0] || '?';
      if (!cats[cat]) cats[cat] = [];
      cats[cat].push(r);
    }});
    html += '<table><thead><tr><th>Category</th><th>Name</th><th>Queries</th><th>NDCG@5</th><th>NDCG@10</th><th>F1@5</th><th>F1@10</th><th>MRR</th></tr></thead><tbody>';
    Object.keys(cats).sort().forEach(cat => {{
      const items = cats[cat];
      const n = items.length;
      const avg = (k) => (items.reduce((s, r) => s + (r[k] || 0), 0) / n).toFixed(4);
      html += `<tr style="cursor:pointer" onclick="detailSubTab='by-query';detailCatFilter='${{cat}}';renderContent()"><td><strong>${{cat}}</strong></td><td>${{CAT_NAMES[cat] || ''}}</td><td>${{n}}</td><td class="metric">${{avg('ndcg@5')}}</td><td class="metric">${{avg('ndcg@10')}}</td><td class="metric">${{avg('f1@5')}}</td><td class="metric">${{avg('f1@10')}}</td><td class="metric">${{avg('mrr')}}</td></tr>`;
    }});
    html += '</tbody></table>';
  }}

  if (detail.length && detailSubTab === 'by-query') {{
    if (detailCatFilter) {{
      html += `<div style="margin-bottom:12px"><button class="btn" onclick="detailCatFilter=null;renderContent()">\\u2190 All categories</button> <strong>${{detailCatFilter}}: ${{CAT_NAMES[detailCatFilter] || ''}}</strong></div>`;
    }}
    const filtered = detailCatFilter ? detail.filter(r => (r.category || r.query_id?.[0]) === detailCatFilter) : detail;
    const sorted = [...filtered].sort((a, b) => a.query_id.localeCompare(b.query_id, undefined, {{numeric: true}}));
    html += '<p style="font-size:11px;color:var(--muted);margin-bottom:8px">Click a query to see returned items</p>';
    html += '<table><thead><tr><th>ID</th><th>Query</th><th>Cat</th><th>NDCG@5</th><th>NDCG@10</th><th>F1@5</th><th>F1@10</th><th>MRR</th></tr></thead><tbody>';
    sorted.forEach(r => {{
      const m = (k) => (r[k] || 0).toFixed(4);
      html += `<tr style="cursor:pointer" onclick="selectedQueryId='${{r.query_id}}';detailSubTab='query-detail';renderContent()"><td>${{r.query_id}}</td><td>${{r.query || ''}}</td><td>${{r.category || ''}}</td><td class="metric">${{m('ndcg@5')}}</td><td class="metric">${{m('ndcg@10')}}</td><td class="metric">${{m('f1@5')}}</td><td class="metric">${{m('f1@10')}}</td><td class="metric">${{m('mrr')}}</td></tr>`;
    }});
    html += '</tbody></table>';
  }}

  if (detailSubTab === 'query-detail') {{
    // Query selector dropdown
    html += '<div style="margin-bottom:16px;display:flex;align-items:center;gap:12px">';
    html += '<label style="font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase">Query:</label>';
    html += `<select style="font-size:13px;padding:6px 10px;border:1px solid var(--border);border-radius:6px;background:var(--card);min-width:300px" onchange="selectedQueryId=this.value;renderContent()">`;
    if (!selectedQueryId) html += '<option value="">-- Select a query --</option>';
    const allQueries = detail.sort((a, b) => a.query_id.localeCompare(b.query_id));
    allQueries.forEach(q => {{
      const sel = q.query_id === selectedQueryId ? 'selected' : '';
      html += `<option value="${{q.query_id}}" ${{sel}}>${{q.query_id}}: ${{q.query || ''}}</option>`;
    }});
    html += '</select>';
    if (selectedQueryId) {{
      const idx = allQueries.findIndex(q => q.query_id === selectedQueryId);
      const prev = idx > 0 ? allQueries[idx - 1].query_id : null;
      const next = idx < allQueries.length - 1 ? allQueries[idx + 1].query_id : null;
      if (prev) html += `<button class="btn" onclick="selectedQueryId='${{prev}}';renderContent()">\\u2190 Prev</button>`;
      if (next) html += `<button class="btn" onclick="selectedQueryId='${{next}}';renderContent()">Next \\u2192</button>`;
    }}
    html += '</div>';

    if (!selectedQueryId) {{
      html += '<div style="padding:40px;text-align:center;color:var(--muted)">Select a query above or from the "By Query" tab</div>';
    }} else {{
      html += `<div id="query-detail-content"><div style="padding:20px;color:var(--muted)">Loading...</div></div>`;
    }}
  }}

  html += '</div>';
  area.innerHTML = html;

  // Load query detail if needed
  if (detailSubTab === 'query-detail' && selectedQueryId) {{
    loadQueryDetail(selectedQueryId, detailRunId);
  }}

  // Draw overview chart if on overview tab
  if (detailSubTab === 'overview' && detail.length) {{
    const cats = {{}};
    detail.forEach(r => {{
      const cat = r.category || r.query_id?.[0] || '?';
      if (!cats[cat]) cats[cat] = [];
      cats[cat].push(r);
    }});
    const catKeys = Object.keys(cats).sort();
    const ctx = document.getElementById('detail-chart');
    if (ctx) {{
      new Chart(ctx, {{
        type: 'bar',
        data: {{
          labels: catKeys.map(c => c + ' ' + (CAT_NAMES[c] || '').substring(0, 20)),
          datasets: [{{
            label: 'NDCG@10',
            data: catKeys.map(c => cats[c].reduce((s, r) => s + (r['ndcg@10'] || 0), 0) / cats[c].length),
            backgroundColor: '#3b82f680', borderColor: '#3b82f6', borderWidth: 1,
          }}, {{
            label: 'F1@10',
            data: catKeys.map(c => cats[c].reduce((s, r) => s + (r['f1@10'] || 0), 0) / cats[c].length),
            backgroundColor: '#10b98180', borderColor: '#10b981', borderWidth: 1,
          }}],
        }},
        options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ beginAtZero: true, max: 1 }} }} }},
      }});
    }}
  }}
}}

// --- Compare view ---
let compareSubTab = 'overview';

function renderCompare(area) {{
  const runs = (RUNS[currentIndex] || []).filter(r => selected.has(r.id));
  if (runs.length < 2) {{ viewMode = 'table'; renderContent(); return; }}

  let html = `<button class="btn back-btn" onclick="viewMode='table';compareSubTab='overview';renderContent()">\\u2190 Back to table</button>`;

  // Config comparison table
  html += '<div class="compare-config"><table><thead><tr><th>#</th><th>Model</th><th>Multimodal</th><th>Pop. Norm.</th><th>&alpha;</th><th>&beta;</th><th>Field Boosts</th><th>NDCG@10</th><th>F1@10</th><th>MRR</th></tr></thead><tbody>';
  runs.forEach(r => {{
    const c = r.config || {{}};
    html += `<tr><td><strong>#${{r.id}}</strong></td><td>${{c.model||'-'}}</td><td>${{c.multimodal||'-'}}</td><td>${{c.pop_norm||'-'}}</td><td>${{c.alpha ?? '-'}}</td><td>${{c.beta ?? '-'}}</td><td>${{c.field_boosts||'-'}}</td><td class="metric">${{(r.metrics['ndcg@10'] || 0).toFixed(4)}}</td><td class="metric">${{(r.metrics['f1@10'] || 0).toFixed(4)}}</td><td class="metric">${{(r.metrics.mrr || 0).toFixed(4)}}</td></tr>`;
  }});
  html += '</tbody></table></div>';

  // Sub-tabs
  html += '<div class="sub-tabs">';
  ['overview','by-category','by-query','query-detail'].forEach(t => {{
    const active = compareSubTab === t ? 'active' : '';
    const label = t === 'overview' ? 'Overview' : t === 'by-category' ? 'By Category' : t === 'by-query' ? 'By Query' : 'Query Detail';
    html += `<div class="sub-tab ${{active}}" onclick="compareSubTab='${{t}}';renderContent()">${{label}}</div>`;
  }});
  html += '</div>';

  const allDetail = runs.map(r => (DETAIL[currentIndex] || {{}})[r.id] || []);
  const allCats = new Set();
  allDetail.forEach(d => d.forEach(r => allCats.add(r.category || r.query_id?.[0] || '?')));
  const cats = [...allCats].sort();

  if (compareSubTab === 'overview') {{
    // Chart
    html += '<div class="chart-container"><canvas id="compare-chart"></canvas></div>';
  }}

  if (compareSubTab === 'by-category' && cats.length) {{
    html += '<div style="overflow-x:auto"><table><thead><tr><th>Category</th><th>Name</th>';
    runs.forEach(r => html += `<th>#${{r.id}}</th>`);
    html += '</tr></thead><tbody>';
    cats.forEach(cat => {{
      html += `<tr style="cursor:pointer" onclick="compareSubTab='by-query';detailCatFilter='${{cat}}';renderContent()"><td><strong>${{cat}}</strong></td><td>${{CAT_NAMES[cat] || ''}}</td>`;
      runs.forEach((r, ri) => {{
        const items = allDetail[ri].filter(x => (x.category || x.query_id?.[0]) === cat);
        const avg = items.length ? items.reduce((s, x) => s + (x['ndcg@10'] || 0), 0) / items.length : 0;
        const color = avg >= 0.8 ? 'var(--good)' : avg >= 0.5 ? 'var(--warn)' : avg > 0 ? 'var(--danger)' : 'var(--muted)';
        html += `<td class="metric" style="color:${{color}}">${{avg.toFixed(3)}}</td>`;
      }});
      html += '</tr>';
    }});
    html += '</tbody></table></div>';
  }}

  if (compareSubTab === 'by-query') {{
    if (detailCatFilter) {{
      html += `<div style="margin-bottom:12px"><button class="btn" onclick="detailCatFilter=null;compareSubTab='by-category';renderContent()">\\u2190 All categories</button> <strong>${{detailCatFilter}}: ${{CAT_NAMES[detailCatFilter] || ''}}</strong></div>`;
    }}
    // Show per-query comparison table
    const queryIds = new Set();
    allDetail.forEach(d => d.forEach(r => {{
      if (!detailCatFilter || (r.category || r.query_id?.[0]) === detailCatFilter) queryIds.add(r.query_id);
    }}));
    const sortedQids = [...queryIds].sort();

    html += '<div style="overflow-x:auto"><table><thead><tr><th>ID</th><th>Query</th>';
    runs.forEach(r => html += `<th>#${{r.id}} NDCG@10</th>`);
    html += '</tr></thead><tbody>';

    const rows = sortedQids.map(qid => {{
      const vals = runs.map((r, ri) => {{
        const item = allDetail[ri].find(x => x.query_id === qid);
        return item ? (item['ndcg@10'] || 0) : 0;
      }});
      const query = allDetail.flat().find(x => x.query_id === qid);
      return {{ qid, query: query?.query || '', vals }};
    }});
    rows.sort((a, b) => a.qid.localeCompare(b.qid, undefined, {{numeric: true}}));

    html += '<p style="font-size:11px;color:var(--muted);margin-bottom:8px">Click a query to compare returned items across runs</p>';
    rows.forEach(row => {{
      html += `<tr style="cursor:pointer" onclick="selectedQueryId='${{row.qid}}';compareSubTab='query-detail';renderContent()"><td>${{row.qid}}</td><td>${{row.query}}</td>`;
      row.vals.forEach(v => {{
        html += `<td class="metric">${{v.toFixed(3)}}</td>`;
      }});
      html += '</tr>';
    }});
    html += '</tbody></table></div>';
  }}

  if (compareSubTab === 'query-detail') {{
    // Query selector
    const allQids = [...new Set(allDetail.flat().map(r => r.query_id))].sort((a, b) => a.localeCompare(b, undefined, {{numeric: true}}));
    html += '<div style="margin-bottom:16px;display:flex;align-items:center;gap:12px">';
    html += '<label style="font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase">Query:</label>';
    html += `<select style="font-size:13px;padding:6px 10px;border:1px solid var(--border);border-radius:6px;background:var(--card);min-width:300px" onchange="selectedQueryId=this.value;renderContent()">`;
    if (!selectedQueryId) html += '<option value="">-- Select a query --</option>';
    allQids.forEach(qid => {{
      const q = allDetail.flat().find(x => x.query_id === qid);
      const sel = qid === selectedQueryId ? 'selected' : '';
      html += `<option value="${{qid}}" ${{sel}}>${{qid}}: ${{q?.query || ''}}</option>`;
    }});
    html += '</select>';
    if (selectedQueryId) {{
      const idx = allQids.indexOf(selectedQueryId);
      if (idx > 0) html += `<button class="btn" onclick="selectedQueryId='${{allQids[idx-1]}}';renderContent()">\\u2190 Prev</button>`;
      if (idx < allQids.length - 1) html += `<button class="btn" onclick="selectedQueryId='${{allQids[idx+1]}}';renderContent()">Next \\u2192</button>`;
    }}
    html += '</div>';

    if (selectedQueryId) {{
      const qInfo = allDetail.flat().find(x => x.query_id === selectedQueryId);
      html += `<h3 style="margin-bottom:4px">${{selectedQueryId}}: "${{qInfo?.query || ''}}"</h3>`;
      html += `<div style="color:var(--muted);font-size:12px;margin-bottom:16px">Category: ${{qInfo?.category || '?'}} — ${{CAT_NAMES[qInfo?.category] || ''}}</div>`;
      html += `<div id="compare-query-detail"><div style="padding:20px;color:var(--muted)">Loading...</div></div>`;
    }} else {{
      html += '<div style="padding:40px;text-align:center;color:var(--muted)">Select a query above or from the "By Query" tab</div>';
    }}
  }}

  area.innerHTML = html;

  // Draw chart
  const ctx = document.getElementById('compare-chart');
  if (ctx) {{
    const metrics = ['ndcg@5','ndcg@10','f1@5','f1@10','mrr'];
    const colors = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4','#84cc16','#f97316','#6366f1'];
    new Chart(ctx, {{
      type: 'bar',
      data: {{
        labels: metrics.map(m => m.toUpperCase()),
        datasets: runs.map((r, i) => ({{
          label: `#${{r.id}}`,
          data: metrics.map(m => r.metrics[m] || 0),
          backgroundColor: colors[i % colors.length] + '80',
          borderColor: colors[i % colors.length],
          borderWidth: 1,
        }})),
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        scales: {{ y: {{ beginAtZero: true, max: 1 }} }},
        plugins: {{ legend: {{ position: 'top' }} }},
      }},
    }});
  }}

  // Load compare query detail if needed
  if (compareSubTab === 'query-detail' && selectedQueryId) {{
    loadCompareQueryDetail(selectedQueryId, runs);
  }}
}}

async function loadCompareQueryDetail(qid, runs) {{
  const panel = document.getElementById('compare-query-detail');
  if (!panel) return;

  if (!detailCache[currentIndex]) {{
    panel.innerHTML = '<div style="padding:20px;color:var(--muted)">Loading query detail...</div>';
    try {{
      const resp = await fetch(`${{currentIndex}}-detail.json`);
      detailCache[currentIndex] = await resp.json();
    }} catch (e) {{
      panel.innerHTML = '<div style="padding:20px;color:var(--danger)">Could not load query detail. Run from a web server.</div>';
      return;
    }}
  }}

  const colors = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4','#84cc16','#f97316','#6366f1'];
  let html = '<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(350px, 1fr));gap:16px;margin-top:16px">';

  runs.forEach((run, ri) => {{
    const runItems = detailCache[currentIndex][run.id]?.[qid] || [];
    const perQuery = (DETAIL[currentIndex] || {{}})[run.id] || [];
    const qMetrics = perQuery.find(q => q.query_id === qid);

    html += `<div style="border:2px solid ${{colors[ri % colors.length]}};border-radius:8px;padding:16px;background:var(--card)">`;
    html += `<div style="font-weight:600;margin-bottom:8px;color:${{colors[ri % colors.length]}}">Run #${{run.id}}</div>`;
    if (qMetrics) {{
      html += `<div style="font-size:11px;color:var(--muted);margin-bottom:8px">NDCG@10=${{(qMetrics['ndcg@10']||0).toFixed(4)}} | F1@10=${{(qMetrics['f1@10']||0).toFixed(4)}} | MRR=${{(qMetrics.mrr||0).toFixed(4)}}</div>`;
    }}
    html += '<table style="width:100%"><thead><tr><th>#</th><th>Name</th><th>Score</th><th>Rel</th></tr></thead><tbody>';
    runItems.forEach((item, i) => {{
      const rel = item.relevance || 0;
      const bg = rel >= 3 ? 'background:#dcfce7' : rel >= 2 ? 'background:#fef9c3' : rel >= 1 ? 'background:#fef3c7' : '';
      html += `<tr style="${{bg}}"><td>${{i+1}}</td><td style="font-size:11px">${{item.name || item.id}}</td><td class="metric">${{item.score ? item.score.toFixed(3) : '-'}}</td><td>${{rel > 0 ? '<strong>'+rel+'</strong>' : '0'}}</td></tr>`;
    }});
    if (!runItems.length) html += '<tr><td colspan="4" style="color:var(--muted)">No detail available</td></tr>';
    html += '</tbody></table></div>';
  }});

  html += '</div>';
  panel.innerHTML = html;
}}

// --- Query detail (lazy load items) ---
let detailCache = {{}};

async function loadQueryDetail(qid, runId) {{
  const panel = document.getElementById('query-detail-content');
  if (!panel) return;

  // Load detail file if not cached
  if (!detailCache[currentIndex]) {{
    panel.innerHTML = '<div style="padding:20px;color:var(--muted)">Loading query detail...</div>';
    try {{
      const resp = await fetch(`${{currentIndex}}-detail.json`);
      detailCache[currentIndex] = await resp.json();
    }} catch (e) {{
      panel.innerHTML = '<div style="padding:20px;color:var(--danger)">Could not load query detail. Run from a web server.</div>';
      return;
    }}
  }}

  const runDetail = detailCache[currentIndex][runId];
  if (!runDetail || !runDetail[qid]) {{
    panel.innerHTML = '<div style="padding:20px;color:var(--muted)">No item detail available for this query.</div>';
    return;
  }}

  // Get query info from per_query data
  const run = (RUNS[currentIndex] || []).find(r => r.id === runId);
  const perQuery = (DETAIL[currentIndex] || {{}})[runId] || [];
  const qMetrics = perQuery.find(q => q.query_id === qid);

  const items = runDetail[qid];
  let html = '';
  html += `<div style="margin-bottom:16px"><button class="btn" onclick="detailSubTab='by-query';renderContent()">\\u2190 Back to queries</button></div>`;
  html += `<h3 style="margin-bottom:4px">${{qid}}: "${{qMetrics?.query || qid}}"</h3>`;
  html += `<div style="color:var(--muted);font-size:12px;margin-bottom:16px">Category: ${{qMetrics?.category || '?'}} — ${{CAT_NAMES[qMetrics?.category] || ''}}</div>`;

  // Query metrics
  if (qMetrics) {{
    html += '<div class="stats-grid" style="margin-bottom:16px">';
    ['ndcg@5','ndcg@10','f1@5','f1@10','mrr'].forEach(k => {{
      const v = qMetrics[k];
      html += `<div class="stat-card"><div class="value" style="font-size:20px">${{v ? v.toFixed(4) : '-'}}</div><div class="label">${{k.toUpperCase()}}</div></div>`;
    }});
    html += '</div>';
  }}

  // Results table
  html += '<h4 style="margin-bottom:8px">Returned items (top 10)</h4>';
  html += '<table><thead><tr><th>#</th><th>ID</th><th>Name</th><th>Score</th><th>Relevance</th></tr></thead><tbody>';

  items.forEach((item, i) => {{
    const rel = item.relevance || 0;
    const bg = rel >= 3 ? 'background:#dcfce7' : rel >= 2 ? 'background:#fef9c3' : rel >= 1 ? 'background:#fef3c7' : '';
    const relBadge = rel > 0 ? `<span class="badge" style="${{rel >= 3 ? 'background:#c8e6c9;color:#2e7d32' : rel >= 2 ? 'background:#fff9c4;color:#f57f17' : 'background:#ffecb3;color:#e65100'}}">${{rel}}</span>` : '<span style="color:var(--muted)">0</span>';
    html += `<tr style="${{bg}}"><td>${{i + 1}}</td><td style="font-family:var(--mono);font-size:11px">${{item.id}}</td><td>${{item.name || '-'}}</td><td class="metric">${{item.score ? item.score.toFixed(4) : '-'}}</td><td>${{relBadge}}</td></tr>`;
  }});

  html += '</tbody></table>';
  panel.innerHTML = html;
}}

init();
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Generating site...')

    # Load data
    print('Loading runs...')
    runs = load_all_runs()
    for idx, r in runs.items():
        print(f'  {idx}: {len(r)} runs')

    # Generate pages
    print('Generating pages...')
    generate_index()
    generate_benchmark()
    generate_embeddings()
    generate_comparison(runs)
    generate_summary()
    generate_experiments(runs)

    # Copy CSS
    (SITE_DIR / 'assets' / 'style.css').write_text('''
:root {
  --bg: #f8fafc; --sidebar-bg: #1e293b; --sidebar-text: #e2e8f0;
  --sidebar-muted: #94a3b8; --sidebar-hover: #334155; --sidebar-active: #3b82f6;
  --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--font); background: var(--bg); height: 100vh; overflow: hidden; }
.layout { display: flex; height: 100vh; }
.sidebar { width: 280px; min-width: 280px; background: var(--sidebar-bg); color: var(--sidebar-text); display: flex; flex-direction: column; overflow-y: auto; padding: 24px 0; }
.logo { padding: 0 24px 24px; border-bottom: 1px solid rgba(255,255,255,0.08); }
.logo h1 { font-size: 18px; font-weight: 700; }
.logo .subtitle { font-size: 12px; color: var(--sidebar-muted); margin-top: 2px; }
.nav-sections { list-style: none; flex: 1; padding: 16px 0; }
.nav-section { margin-bottom: 8px; }
.nav-section h3 { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--sidebar-muted); padding: 12px 24px 6px; }
.nav-section ul { list-style: none; }
.nav-section ul li a { display: block; padding: 7px 24px 7px 32px; font-size: 13px; color: var(--sidebar-text); text-decoration: none; border-left: 3px solid transparent; }
.nav-section ul li a:hover { background: var(--sidebar-hover); color: #fff; }
.nav-section .divider { height: 1px; background: rgba(255,255,255,0.08); margin: 8px 24px; }
.footer { padding: 16px 24px; border-top: 1px solid rgba(255,255,255,0.08); font-size: 12px; color: var(--sidebar-muted); }
.footer .muted { font-size: 11px; opacity: 0.6; margin-top: 2px; }
.content { flex: 1; overflow: hidden; }
.content iframe { width: 100%; height: 100%; border: none; background: var(--bg); }
''')
    print('  assets/style.css')

    print(f'Done! Open {SITE_DIR}/index.html')


if __name__ == '__main__':
    main()
