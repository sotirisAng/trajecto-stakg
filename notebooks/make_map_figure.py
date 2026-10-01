"""Render the recommendation-map figure for the thesis.

Folium writes an interactive HTML map; a thesis needs a PNG. Headless Chrome
renders the HTML, which works because the page is self-contained apart from
its tiles. Two things have to be corrected before the shot is usable:

  * the visualiser's default base layer is CARTO Positron, whose free tiles
    are stamped API KEY REQUIRED across every tile, so the base layer is
    switched to standard OpenStreetMap, and the attribution with it;
  * the default view is a fixed centre and zoom, which on a city-scale corpus
    leaves the markers as specks, so the view is fitted to the POIs shown.

    python make_map_figure.py [--kind profile] [--traj <uri>]

"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

CHROME = os.environ.get('CHROME', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
OSM_TILES = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
FIGS = pathlib.Path(__file__).resolve().parent / 'output' / 'figures'


OSM_ATTR = ('&copy; <a href="https://www.openstreetmap.org/copyright">'
            'OpenStreetMap</a> contributors')


def osm_base(m):
    """Repoint every CARTO tile layer at OpenStreetMap, attribution included.

    Done on the map object rather than on the rendered HTML: the attribution
    string Leaflet prints in the footer is a layer option, so rewriting only
    the tile URLs leaves the figure crediting a provider it no longer uses.
    """
    import folium
    for layer in m._children.values():
        if isinstance(layer, folium.raster_layers.TileLayer) and \
                'cartocdn' in str(layer.tiles):
            layer.tiles = OSM_TILES
            layer.options['attribution'] = OSM_ATTR
    return m


def shoot(m, out: pathlib.Path, size=(1600, 1000)) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    html = osm_base(m).get_root().render().replace('</head>', PRINT_CSS + '</head>')
    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as fh:
        fh.write(html)
        src = fh.name
    subprocess.run(
        [CHROME, '--headless', '--disable-gpu', '--no-sandbox',
         '--force-device-scale-factor=2',
         '--hide-scrollbars', f'--window-size={size[0]},{size[1]}',
         '--virtual-time-budget=60000', f'--screenshot={out}', f'file://{src}'],
        check=True, capture_output=True)
    pathlib.Path(src).unlink()
    # Headless Chrome sometimes leaves the bottom tile row unloaded, which
    # prints as a flat Leaflet-grey band; trim it (captions credit OSM).
    from PIL import Image
    im = Image.open(out).convert('RGB')
    h = im.height
    while h > 1 and all(im.getpixel((x, h - 1)) == (221, 221, 221)
                        for x in range(0, im.width // 2, 50)):
        h -= 1
    if h < im.height:
        im.crop((0, 0, im.width, h)).save(out)
    print(f'wrote {out.name} ({out.stat().st_size // 1024} KB)')


PRINT_CSS = ('<style>.leaflet-control-zoom,.leaflet-control-layers'
             '{display:none}</style>')
PROFILE_TRAJ = 'http://w3id.org/trajecto#trajectory_177704932659'


def nb15():
    """The card and trace queries of notebook 15, run as they are there."""
    nb = json.loads((pathlib.Path(__file__).parent /
                     '15-trajectory-profile-card.ipynb').read_text())
    src = lambda i: ''.join(nb['cells'][i]['source'])
    ns = {}
    for i in (1, 3, 5):
        exec(src(i), ns)
    maps = src(10)
    exec(maps[:maps.find('MAPS_DIR =')], ns)   # defs only, not the demo run
    return ns


def profile_map(traj, nb=None, panel=True):
    """Profile map: the GPS trace, the stops coloured by category, and the
    profile card of notebook 15 beside them, all read from the KG."""
    import folium
    from helpers.map_visualizer import TrajectoMapVisualizer, get_category_color
    from helpers.poi_categories import resolve_category

    nb = nb or nb15()
    card = nb['build_card'](traj)
    path = nb['fetch_trajectory_path'](traj)
    colour = lambda c: get_category_color(resolve_category(c) or c)
    # Repeated interactions with one POI stack on one spot; draw one stop
    # carrying the strongest class the pipeline gave it.
    strength = ['MeaningfulInteraction', 'ImmersiveInteraction',
                'DominantInteraction'].index
    visited = []
    for p in nb['fetch_visited_pois'](traj):
        if visited and visited[-1]['poi'] == p['poi']:
            if strength(p['interaction_type']) > strength(visited[-1]['interaction_type']):
                visited[-1]['interaction_type'] = p['interaction_type']
        else:
            visited.append(p)
    print(f'{card["trajectory"]}: {len(visited)} stops, {len(path)} positions')

    viz = TrajectoMapVisualizer()
    # The palette is keyed on canonical names; the KG holds ontology classes.
    m = viz.create_trajectory_map(
        [{**p, 'category': resolve_category(p['category']) or p['category']}
         for p in visited], show_path=False)
    folium.PolyLine(path, color='#37474F', weight=3, opacity=0.85).add_to(m)
    # Rings mark the interaction class that the pipeline assigned the stop.
    ring = {'DominantInteraction': '#FFB300', 'ImmersiveInteraction': '#1A237E'}
    for p in visited:
        if p['interaction_type'] in ring:
            folium.CircleMarker([p['latitude'], p['longitude']], radius=19,
                                color=ring[p['interaction_type']], weight=4,
                                fill=False).add_to(m)

    d, ic = card['duration'], card['interaction_counts']
    dom = card['dominant_poi']
    rows = [
        ('Duration', f"{d['hours']} h, {d['n_positions']} positions<br>"
                     f"{d['start'][:16]} to {d['end'][11:16]}"),
        ('Temporal context', ', '.join(card['temporal_context'])),
        ('Purpose', ', '.join(card['purpose'])),
        ('Navigation style', ', '.join(card['navigation_style'])),
        ('Engagement', card['engagement']),
        ('Dominant POI', f"{dom['label']} ({dom['category']})" if dom else ''),
        ('Significant POIs', ', '.join(p['label'] for p in card['significant_pois'])),
        ('Category sequence', ' &rarr; '.join(card['category_arc'])),
        ('Interactions', ', '.join(f'{k} {v}' for k, v in ic.items()
                                   if v and k != 'total_distinct')),
        ('Co-visited with', f"{len(card['covisited_trajectories'])} trajectories"
                            ' sharing a POI'),
    ]
    table = ''.join(
        f'<tr><td style="color:#666;padding:4px 12px 4px 0;vertical-align:top;'
        f'white-space:nowrap">{k}</td><td style="padding:4px 0">{v or "-"}</td></tr>'
        for k, v in rows)
    cats = dict.fromkeys(p['category'] for p in visited)
    legend = ''.join(
        f'<span style="display:inline-block;margin:2px 10px 2px 0">'
        f'<span style="display:inline-block;width:11px;height:11px;'
        f'border-radius:50%;background:{colour(c)};vertical-align:-1px"></span> {c}</span>'
        for c in cats)
    legend += ('<br><span style="color:#FFB300;font-weight:bold">&#9711;</span> dominant'
               ' &nbsp; <span style="color:#1A237E;font-weight:bold">&#9711;</span>'
               ' immersive &nbsp; <span style="color:#37474F">&#9472;&#9472;</span>'
               ' GPS trace')
    card_panel = f'''
    <div style="position:fixed;top:16px;right:16px;width:470px;z-index:1000;
                background:rgba(255,255,255,0.96);border-radius:6px;padding:18px 22px;
                box-shadow:0 2px 8px rgba(0,0,0,0.3);font-family:Arial;font-size:14px;
                color:#222">
      <div style="font-size:17px;font-weight:bold;margin-bottom:10px">
        Profile card: {card['trajectory']}</div>
      <table style="border-collapse:collapse;width:100%">{table}</table>
      <div style="border-top:1px solid #ddd;margin-top:12px;padding-top:10px;
                  font-size:13px;line-height:1.6">{legend}</div>
    </div>'''
    if panel:
        m.get_root().html.add_child(folium.Element(card_panel))
    m.fit_bounds(path + [[p['latitude'], p['longitude']] for p in visited],
                 padding_top_left=(40, 40),
                 padding_bottom_right=(540 if panel else 40, 40))
    return osm_base(m), card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kg', default='osm-rome')
    ap.add_argument('--traj', default=None)
    ap.add_argument('--top-k', type=int, default=8)
    ap.add_argument('--kind', choices=('recommendation', 'profile'),
                    default='recommendation')
    a = ap.parse_args()

    traj = a.traj or PROFILE_TRAJ
    if a.kind == 'profile':
        return shoot(profile_map(traj)[0], FIGS / 'fig-trajecto-rome-trajectory.png')

    from helpers.unified_recommender import UnifiedRecommender

    rec = UnifiedRecommender()
    if a.kg not in rec.endpoint:
        raise RuntimeError(f'endpoint {rec.endpoint} != --kg {a.kg}')
    rec.load_all_data(features_path='output/semantic_features/trajectory_features.json')

    shoot(recommendation_map(rec, traj, a.top_k),
          FIGS / 'fig-trajecto-recommendation-map.png')


def recommendation_map(rec, traj, top_k=8):
    """Observed 70% prefix, top-k recommendations, and the explanation of
    each: the leading signal contributions, the endorsing rules, the
    confidence. Uses the evaluation split, so 'visited later' is a hit."""
    import folium
    from helpers.eval_protocols import _split
    from helpers.map_visualizer import TrajectoMapVisualizer
    from helpers.poi_categories import resolve_category
    from helpers.unified_recommender import UNIFIED_WEIGHTS, _RULE_LABELS

    seq, held = _split(rec, traj, 'trajectory_split', 0.7, 1)
    held = set(held)
    recs = [r for r in rec.recommend(traj, observed_sequence=seq, top_k=top_k,
                                     exclude_traj=traj)
            if r.get('latitude') and r.get('longitude')]
    print(f'{traj}: {len(seq)} observed, {len(recs)} recommended')

    visited = []
    for p in seq:
        md = rec.poi_metadata.get(p, {})
        if md.get('lat') and md.get('lon'):
            cat = md.get('category', '')
            visited.append({'poi': p, 'label': md.get('label', ''),
                            'category': resolve_category(cat) or cat,
                            'latitude': md['lat'], 'longitude': md['lon']})
    m = TrajectoMapVisualizer().create_recommendation_map(
        visited, [{**r, 'name': r['label']} for r in recs])
    for i, r in enumerate(recs, 1):
        folium.Marker([r['latitude'], r['longitude']], icon=folium.DivIcon(
            html=f'<div style="font:bold 13px Arial;color:#fff;'
                 f'text-shadow:0 0 3px #1B5E20">R{i}</div>',
            icon_size=(24, 16), icon_anchor=(10, 8))).add_to(m)

    def why(r):
        top = sorted(((UNIFIED_WEIGHTS.get(k, 0) * v, k)
                      for k, v in r['signals'].items()), reverse=True)
        sig = ', '.join(k.replace('_', ' ') for c, k in top[:3] if c > 0)
        rules = ', '.join(f'{x} ({_RULE_LABELS.get(x, x)})'
                          for x in (r['rules'] or {}).get('matched', []))
        return sig + (f'<br><span style="color:#666">rules: {rules}</span>'
                      if rules else '')

    body = ''.join(
        f'<tr style="border-top:1px solid #eee"><td style="padding:5px 8px 5px 0;'
        f'vertical-align:top;font-weight:bold;color:#2E7D32">R{i}</td>'
        f'<td style="padding:5px 8px 5px 0;vertical-align:top">{r["label"]}'
        f'<br><span style="color:#666">{r["category"]}</span>'
        f'{"<br><b style=color:#2E7D32>&#10003; visited later</b>" if r["poi"] in held else ""}</td>'
        f'<td style="padding:5px 8px 5px 0;vertical-align:top">{r["score"]:.2f}'
        f'<br><span style="color:#666">{r["confidence"]}</span></td>'
        f'<td style="padding:5px 0;vertical-align:top">{why(r)}</td></tr>'
        for i, r in enumerate(recs, 1))
    meta = rec.traj_metadata.get(traj, {})
    panel = f"""
    <div style="position:fixed;top:16px;right:16px;width:560px;z-index:1000;
                background:rgba(255,255,255,0.96);border-radius:6px;padding:16px 20px;
                box-shadow:0 2px 8px rgba(0,0,0,0.3);font-family:Arial;font-size:13px;
                color:#222">
      <div style="font-size:17px;font-weight:bold">Recommendations for
        {traj.split('#')[-1]}</div>
      <div style="color:#555;margin:6px 0 8px">
        {len(visited)} observed POIs (numbered) &middot;
        {', '.join(sorted(meta.get('purposes', [])))} &middot;
        {meta.get('engagement', '')}</div>
      <table style="border-collapse:collapse;width:100%">
        <tr style="color:#666;text-align:left"><th></th><th>POI</th>
            <th>Score</th><th>Leading signals</th></tr>{body}</table>
    </div>"""
    m.get_root().html.add_child(folium.Element(panel))
    m.fit_bounds([[p['latitude'], p['longitude']] for p in visited + recs],
                 padding_top_left=(40, 40), padding_bottom_right=(620, 40))
    return osm_base(m)


if __name__ == '__main__':
    main()
