"""Cross-dataset comparison figure of thesis chapter 7 (fig:trajecto-cross-dataset).

Reads navigation styles, engagement levels, and purposes from the three
Trajecto STaKG instantiations in Fuseki and plots them as shares of each
dataset's trajectories. Writes output/figures/fig-trajecto-cross-dataset.png.

    python notebooks/make_cross_dataset_figure.py
"""
import pathlib
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / 'helpers'))
from fuseki_queries import execute_sparql_query  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / 'output' / 'figures' / 'fig-trajecto-cross-dataset.png'
FUSEKI = 'http://localhost:3030'
# name -> (Fuseki dataset, color); darker steps of the thesis blue/orange/purple, validated for CVD separation
DATASETS = {
    'OSM Rome': ('osm-rome', '#3F6FB0'),
    'GeoLife': ('geolife', '#E08E00'),
    'BIRD': ('bird', '#8C5FA0'),
}
PREFIXES = """
PREFIX trajecto: <http://w3id.org/trajecto#>
PREFIX patterns: <http://w3id.org/trajecto/patterns#>
PREFIX datacron: <http://www.datacron-project.eu/datAcron#>
"""
QUERIES = {
    'nav': 'SELECT ?t ?v WHERE { ?t a datacron:Trajectory ; patterns:exhibitsPattern ?v }',
    'eng': 'SELECT ?t ?v WHERE { ?t a datacron:Trajectory ; trajecto:hasEngagementLevel ?v }',
    'purpose': 'SELECT ?t ?v WHERE { ?t a datacron:Trajectory ; trajecto:hasPurpose ?v }',
}
NAV = ['Targeted', 'Casual', 'Exploratory', 'Immersive', 'Steady']
ENG = ['Low', 'Medium', 'High']
PURPOSE = ['Commercial', 'Cultural', 'Transportation', 'Recreational', 'Other',
           'General', 'ClassicalArts', 'ModernArts', 'ContemporaryArts', '(none)']
OTHER = {'Educational', 'Healthcare', 'Accommodation'}  # each under 1% of GeoLife, absent elsewhere
SUFFIXES = ('Segment', 'Movement', 'Engagement', 'Purpose')


def label(uri):
    # local name without the class suffix; merges purpose# and purpose/skos# duplicates
    name = uri.rsplit('#', 1)[-1]
    for s in SUFFIXES:
        name = name.removesuffix(s)
    return name


def rows(ds, q):
    res = execute_sparql_query(f'{FUSEKI}/{ds}/sparql', PREFIXES + q)
    if res is None:
        sys.exit(f'query failed on {ds}; is Fuseki running?')
    return [{k: v['value'] for k, v in b.items()} for b in res['results']['bindings']]


def shares(ds):
    """Per facet: {label: % of the dataset's trajectories carrying it}."""
    n = int(rows(ds, 'SELECT (COUNT(DISTINCT ?t) AS ?n) WHERE { ?t a datacron:Trajectory }')[0]['n'])
    out = {}
    for facet, q in QUERIES.items():
        trajs = defaultdict(set)
        for r in rows(ds, q):
            lab = label(r['v'])
            trajs['Other' if facet == 'purpose' and lab in OTHER else lab].add(r['t'])
        if facet == 'purpose':
            trajs['(none)'] = set(range(n - len(set().union(*trajs.values()))))
        out[facet] = {k: 100 * len(v) / n for k, v in trajs.items()}
    return n, out


def grouped(ax, cats, data, horizontal=False):
    k = len(data)
    w = 0.8 / k
    for i, (name, (vals, color)) in enumerate(data.items()):
        pos = [j + (i - (k - 1) / 2) * w for j in range(len(cats))]
        v = [vals.get(c, 0) for c in cats]
        bars = (ax.barh if horizontal else ax.bar)(pos, v, w * 0.9, color=color, label=name)
        ax.bar_label(bars, labels=[('<1' if x < 1 else f'{x:.0f}') if x else '' for x in v], padding=1.5, fontsize=6.5, color='#333333')
    ticks = (ax.set_yticks if horizontal else ax.set_xticks)
    ticks(range(len(cats)), cats)
    ax.grid(axis='x' if horizontal else 'y', color='#e6e6e6', linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': '#999999',
                         'xtick.color': '#333333', 'ytick.color': '#333333', 'axes.labelcolor': '#333333'})
    res = {}
    for name, (ds, color) in DATASETS.items():
        n, s = shares(ds)
        res[name] = (n, s, color)
        print(name, ds, n, {f: {k: round(v, 1) for k, v in d.items()} for f, d in s.items()})

    fig = plt.figure(figsize=(5.9, 5.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 3], height_ratios=[1, 1.45], hspace=0.45, wspace=0.22)
    ax_nav, ax_eng, ax_pur = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    legend = {f'{name} (n={n:,})': (None, c) for name, (n, _, c) in res.items()}
    for ax, facet, cats, horiz in ((ax_nav, 'nav', NAV, False), (ax_eng, 'eng', ENG, False),
                                   (ax_pur, 'purpose', PURPOSE, True)):
        grouped(ax, cats, {nm: (s[facet], c) for nm, (_, s, c) in res.items()}, horizontal=horiz)
    ax_nav.set_ylabel('% of trajectories')
    ax_nav.set_title('(a) Navigation style', loc='left', fontsize=9)
    ax_eng.set_title('(b) Engagement level', loc='left', fontsize=9)
    ax_pur.set_title('(c) Purpose', loc='left', fontsize=9)
    ax_pur.set_xlabel('% of trajectories')
    ax_pur.invert_yaxis()
    ax_nav.tick_params(axis='x', labelsize=7.5)
    plt.setp(ax_nav.get_xticklabels(), rotation=20, ha='right', rotation_mode='anchor')
    # BIRD's purposes are era-based and share no class with the urban purposes above the line
    ax_pur.axhline(PURPOSE.index('General') - 0.5, color='#999999', linewidth=0.6, linestyle='--')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in legend.values()]
    fig.legend(handles, legend.keys(), loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
    print('saved', OUT)


if __name__ == '__main__':
    main()
