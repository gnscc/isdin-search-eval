"""Normalize all benchmark runs into a unified format with per-query metrics.

Reads runs from the isdin-search-api repo, computes NDCG@k, F1@k, MRR
per query using ground truth, and writes normalized JSON files to
normalized_runs/ for the site generator to consume.

Usage
-----
    python normalize_runs.py
"""

import json
import math
from pathlib import Path

API_REPO = Path(__file__).parent.parent.parent / 'isdin-search-api'
BENCHMARK_DIR = API_REPO / 'data' / 'benchmark' / 'v1'
OUTPUT_DIR = Path(__file__).parent / 'data'


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances[:k]))


def ndcg_at_k(relevances: list[int], ideal: list[int], k: int) -> float:
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(sorted(ideal, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(relevances: list[int], k: int) -> float:
    return sum(1 for r in relevances[:k] if r >= 1) / k if k > 0 else 0.0


def recall_at_k(relevances: list[int], total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(1 for r in relevances[:k] if r >= 1) / total_relevant


def f1_at_k(relevances: list[int], total_relevant: int, k: int) -> float:
    p = precision_at_k(relevances, k)
    r = recall_at_k(relevances, total_relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mrr(relevances: list[int]) -> float:
    for i, r in enumerate(relevances):
        if r >= 1:
            return 1.0 / (i + 1)
    return 0.0


def compute_query_metrics(result_ids: list[str], gt_judgments: dict[str, int]) -> dict:
    """Compute all metrics for a single query.

    Matches evaluate_benchmark.py exactly:
    - total_relevant = len(gt_judgments) (all judged items, not just score >= 1)
    - mrr truncated to k
    """
    relevances = [gt_judgments.get(rid, 0) for rid in result_ids]
    ideal = sorted(gt_judgments.values(), reverse=True)
    total_relevant = len(gt_judgments)

    return {
        'ndcg@5': round(ndcg_at_k(relevances, ideal, 5), 4),
        'ndcg@10': round(ndcg_at_k(relevances, ideal, 10), 4),
        'precision@5': round(precision_at_k(relevances, 5), 4),
        'precision@10': round(precision_at_k(relevances, 10), 4),
        'f1@5': round(f1_at_k(relevances, total_relevant, 5), 4),
        'f1@10': round(f1_at_k(relevances, total_relevant, 10), 4),
        'mrr': round(mrr(relevances[:10]), 4),
        'relevances': relevances[:20],
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_ground_truth(index: str) -> dict[str, dict[str, int]]:
    """Load GT as {query_id: {doc_id: score}}."""
    if index == 'events':
        gt_file = BENCHMARK_DIR / 'events' / 'ground-truth-claude-v2.json'
    else:
        gt_file = BENCHMARK_DIR / index / 'ground-truth-claude.json'

    if not gt_file.exists():
        return {}

    data = json.loads(gt_file.read_text())
    return {j['query_id']: j['judgments'] for j in data['judgments']}


def load_queries(index: str) -> dict[str, dict]:
    """Load queries as {query_id: {query, category, notes}}."""
    if index == 'events':
        q_file = BENCHMARK_DIR / 'events' / 'queries-v2.json'
    else:
        q_file = BENCHMARK_DIR / index / 'queries.json'

    if not q_file.exists():
        return {}

    data = json.loads(q_file.read_text())
    return {q['id']: q for q in data['queries']}


def extract_results(run_data: dict) -> dict[str, list[dict]]:
    """Extract {query_id: [{id, name, score}]} from a run."""
    results_by_query = {}
    for r in run_data.get('results', []):
        qid = r.get('query_id')
        if not qid:
            continue
        items = r.get('results', [])
        if items:
            results_by_query[qid] = [
                {
                    'id': str(item.get('product_id', item.get('id', ''))),
                    'name': item.get('name', ''),
                    'score': round(item.get('score') or 0, 4),
                }
                for item in items[:10]
            ]
        else:
            ids = r.get('result_ids', [])
            results_by_query[qid] = [{'id': pid, 'name': '', 'score': 0} for pid in ids[:10]]
    return results_by_query


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

import re  # noqa: E402


def derive_config(data: dict, filename: str) -> dict:
    """Derive configuration variables from run data and filename."""
    # Special cases for baselines
    if 'products-api' in filename:
        return {
            'model': 'products-api',
            'multimodal': '-',
            'pop_norm': '-',
            'mode': 'lexical',
            'alpha': None,
            'beta': None,
            'field_boosts': '-',
            'variant': '',
        }
    if 'search-api-hybrid-baseline' in filename:
        return {
            'model': 'gemini-001',
            'multimodal': 'none',
            'pop_norm': 'percent_rank',
            'mode': 'hybrid',
            'alpha': 0.6,
            'beta': 0.2,
            'field_boosts': 'intuition',
            'variant': '',
        }
    if 'search-api-embedding2' in filename:
        return {
            'model': 'gemini-2',
            'multimodal': 'none',
            'pop_norm': 'percent_rank',
            'mode': 'hybrid',
            'alpha': 0.6,
            'beta': 0.2,
            'field_boosts': 'intuition',
            'variant': '',
        }

    sync_label = data.get('sync_label', '')
    config = data.get('config', {})

    # Model
    if 'emb1' in sync_label or 'emb1' in filename:
        model = 'gemini-001'
    elif 'products-api' in filename:
        model = 'products-api'
    elif 'search-api' in filename:
        model = 'search-api'
    else:
        model = 'gemini-2'

    # Multimodal
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

    # Pop normalization
    if 'linear' in sync_label:
        pop_norm = 'linear'
    elif 'log' in sync_label:
        pop_norm = 'log'
    else:
        pop_norm = 'percent_rank'

    # Mode
    mode = config.get('mode', '')
    if not mode:
        for m in ('semantic', 'keyword', 'hybrid', 'unified', 'rerank'):
            if m in filename:
                mode = m
                break
        if not mode:
            mode = 'hybrid'

    # Alpha
    alpha = config.get('semantic_weight')
    if alpha is None:
        match = re.search(r'sw([\d.]+)', filename)
        if match:
            alpha = float(match.group(1))

    # Beta
    beta = config.get('popularity_boost')
    if beta is None:
        match = re.search(r'pb([\d.]+)', filename)
        if match:
            beta = float(match.group(1))

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
        elif 'flat' in filename:
            fb = 'flat'
        elif 'weighted' in filename:
            fb = 'weighted'
        elif 'title3_notype' in filename:
            fb = 'title3_notype'

    # Rename "default" to intuition/flat based on sync config timing
    if fb == 'default':
        if sync_label in ('emb2-pctrank', 'emb2-linear', 'emb2-log'):
            fb = 'intuition'
        else:
            fb = 'flat'

    # Variant (cross-index)
    variant = ''
    if 'unified_both' in filename:
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

    return {
        'model': model,
        'multimodal': multimodal,
        'pop_norm': pop_norm,
        'mode': mode,
        'alpha': alpha,
        'beta': beta,
        'field_boosts': fb or '-',
        'variant': variant,
    }


def normalize_index(index: str) -> list[dict]:
    """Normalize all runs for an index."""
    runs_dir = BENCHMARK_DIR / index / 'runs'
    if not runs_dir.exists():
        return []

    gt = load_ground_truth(index)
    queries = load_queries(index)

    # Events v2 runs already have correct metrics from evaluate_events_v2.py
    # (binary relevance, group-based). Use them directly instead of recalculating.
    use_precomputed = (index == 'events')

    # For events, only process v2 runs
    if index == 'events':
        patterns = ['events-v2-*.json']
    else:
        patterns = ['*.json']

    normalized = []
    for pattern in patterns:
        for f in sorted(runs_dir.glob(pattern)):
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            # Skip empty
            if data.get('n_queries', 0) == 0 and not data.get('results'):
                continue

            config = derive_config(data, f.name)

            if use_precomputed:
                # Events v2: use pre-computed metrics from evaluate_events_v2.py
                aggregated = data.get('metrics', {})
                if not aggregated:
                    continue

                # Build per-query from results (already has ndcg@10, mrr per query)
                per_query = []
                for r in data.get('results', []):
                    qid = r.get('query_id')
                    if not qid:
                        continue
                    q_info = queries.get(qid, {})
                    items = [
                        {'id': str(item.get('product_id', '')), 'name': item.get('name', ''), 'score': round(item.get('score') or 0, 4), 'relevance': item.get('relevant', 0)}
                        for item in r.get('results', [])[:10]
                    ]
                    per_query.append({
                        'query_id': qid,
                        'query': q_info.get('query', r.get('query', '')),
                        'category': q_info.get('category', r.get('category', qid[0] if qid else '?')),
                        'n_results': r.get('n_results', len(items)),
                        'ndcg@5': r.get('ndcg@5', 0),
                        'ndcg@10': r.get('ndcg@10', 0),
                        'precision@5': r.get('precision@5', 0),
                        'precision@10': r.get('precision@10', 0),
                        'f1@5': 0,
                        'f1@10': 0,
                        'mrr': r.get('mrr', 0),
                        'items': items,
                    })
            else:
                # Products/cross-index: compute from GT
                results_by_query = extract_results(data)

                if not results_by_query:
                    continue

                # Compute per-query metrics (iterate over run results, like evaluate_benchmark)
                per_query = []
                for qid in sorted(results_by_query.keys()):
                    gt_j = gt.get(qid)
                    if gt_j is None:
                        continue

                    items = results_by_query.get(qid, [])
                    result_ids = [item['id'] for item in items]
                    q_info = queries.get(qid, {})
                    metrics = compute_query_metrics(result_ids, gt_j)

                    annotated_items = []
                    for item in items:
                        annotated_items.append({
                            **item,
                            'relevance': gt_j.get(item['id'], 0),
                        })

                    per_query.append({
                        'query_id': qid,
                        'query': q_info.get('query', ''),
                        'category': q_info.get('category', qid[0] if qid else '?'),
                        'n_results': len(result_ids),
                        'items': annotated_items,
                        **metrics,
                    })

                if not per_query:
                    continue

                # Aggregate metrics
                n = len(per_query)
                aggregated = {
                    'ndcg@5': round(sum(q['ndcg@5'] for q in per_query) / n, 4),
                    'ndcg@10': round(sum(q['ndcg@10'] for q in per_query) / n, 4),
                    'precision@5': round(sum(q['precision@5'] for q in per_query) / n, 4),
                    'precision@10': round(sum(q['precision@10'] for q in per_query) / n, 4),
                    'f1@5': round(sum(q['f1@5'] for q in per_query) / n, 4),
                    'f1@10': round(sum(q['f1@10'] for q in per_query) / n, 4),
                    'mrr': round(sum(q['mrr'] for q in per_query) / n, 4),
                }

            # For precomputed (events), don't recalculate aggregated — already set from data['metrics']

            # Skip if all zeros
            if aggregated['ndcg@10'] == 0 and aggregated['mrr'] == 0:
                continue

            # Skip pure semantic mode runs (alpha=1.0 hybrid is equivalent)
            if config['mode'] == 'semantic':
                continue

            # Skip unified variants that aren't "both" (redundant with single-index search)
            if config['mode'] == 'unified' and config['variant'] not in ('both', ''):
                continue

            normalized.append({
                'filename': f.name,
                'label': data.get('label', f.stem),
                'config': config,
                'metrics': aggregated,
                'n_queries': len(per_query),
                'per_query': per_query,
            })

    # Sort by NDCG@10 desc and assign IDs
    normalized.sort(key=lambda r: r['metrics']['ndcg@10'], reverse=True)
    for i, run in enumerate(normalized, 1):
        run['id'] = i

    return normalized


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for index in ('products', 'events', 'cross-index'):
        print(f'Normalizing {index}...')
        runs = normalize_index(index)
        print(f'  {len(runs)} valid runs')

        # Split: light (metrics only) for table + detail (with items) separate
        runs_light = []
        runs_detail = {}
        for run in runs:
            light_per_query = []
            for q in run['per_query']:
                light_per_query.append({k: v for k, v in q.items() if k != 'items'})
            runs_light.append({**run, 'per_query': light_per_query})
            runs_detail[run['id']] = {q['query_id']: q.get('items', []) for q in run['per_query']}

        output_file = OUTPUT_DIR / f'{index}.json'
        output_file.write_text(json.dumps(runs_light, ensure_ascii=False))
        print(f'  Wrote {output_file} ({output_file.stat().st_size / 1024 / 1024:.1f} MB)')

        detail_file = OUTPUT_DIR / f'{index}-detail.json'
        detail_file.write_text(json.dumps(runs_detail, ensure_ascii=False))
        print(f'  Wrote {detail_file} ({detail_file.stat().st_size / 1024 / 1024:.1f} MB)')


if __name__ == '__main__':
    main()
