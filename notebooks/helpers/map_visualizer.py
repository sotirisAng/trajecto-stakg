"""
Map Visualizer Module for Trajecto

This module provides Folium-based map visualization functions for:
- Trajectory visualization (POI sequences on maps)
- Recommendation visualization (visited vs recommended POIs)
- Trajectory comparison (side-by-side or overlaid)
- Category heatmaps (POI type concentrations)

Usage:
    from helpers.map_visualizer import TrajectoMapVisualizer

    visualizer = TrajectoMapVisualizer()
    trajectory_map = visualizer.create_trajectory_map(poi_sequence)
    recommendation_map = visualizer.create_recommendation_map(visited, recommendations)
"""

import folium
from folium.plugins import MarkerCluster, HeatMap
from typing import List, Dict, Any, Optional, Tuple

# Category color scheme for consistent visualization
CATEGORY_COLORS = {
    # Cultural
    'Museum': '#9C27B0',
    'Gallery': '#7B1FA2',
    'Artwork': '#AB47BC',
    'Exhibit': '#BA68C8',
    'Cultural': '#9C27B0',
    # Historical
    'HistoricalLandmark': '#795548',
    'HistoricalSite': '#6D4C41',
    'Monument': '#8D6E63',
    'Historical': '#795548',
    # Food & Drink
    'Restaurant': '#FF5722',
    'Cafe': '#FF7043',
    'FastFood': '#FF8A65',
    'FoodCourt': '#FFAB91',
    'Bar': '#E64A19',
    # Shopping
    'ShoppingMall': '#2196F3',
    'RetailStore': '#42A5F5',
    'MarketArea': '#64B5F6',
    'CommercialStreet': '#90CAF9',
    'Commercial': '#2196F3',
    # Education
    'University': '#009688',
    'Library': '#26A69A',
    'School': '#4DB6AC',
    'Educational': '#009688',
    # Entertainment
    'Cinema': '#E91E63',
    'Theater': '#EC407A',
    'PerformanceVenue': '#F06292',
    'Nightclub': '#AD1457',
    'NightlifeFacility': '#C2185B',
    'Entertainment': '#E91E63',
    # Recreation
    'Park': '#4CAF50',
    'Beach': '#00BCD4',
    'LeisureFacility': '#8BC34A',
    'SportsFacility': '#CDDC39',
    'Recreational': '#4CAF50',
    # Other
    'Accommodation': '#607D8B',
    'Information': '#9E9E9E',
    'Landmark': '#FFC107',
    'Religious': '#3F51B5',
    # Default
    'Unknown': '#757575',
    'Other': '#757575',
}

# Category icons (FontAwesome icons)
CATEGORY_ICONS = {
    'Museum': 'university',
    'Gallery': 'image',
    'Artwork': 'palette',
    'Exhibit': 'eye',
    'Cultural': 'university',
    'HistoricalLandmark': 'landmark',
    'HistoricalSite': 'monument',
    'Monument': 'monument',
    'Historical': 'landmark',
    'Restaurant': 'utensils',
    'Cafe': 'coffee',
    'FastFood': 'hamburger',
    'FoodCourt': 'utensils',
    'Bar': 'glass-martini',
    'ShoppingMall': 'shopping-cart',
    'RetailStore': 'store',
    'MarketArea': 'store-alt',
    'CommercialStreet': 'shopping-bag',
    'Commercial': 'shopping-cart',
    'University': 'graduation-cap',
    'Library': 'book',
    'School': 'school',
    'Educational': 'graduation-cap',
    'Cinema': 'film',
    'Theater': 'theater-masks',
    'PerformanceVenue': 'music',
    'Nightclub': 'glass-cheers',
    'NightlifeFacility': 'moon',
    'Entertainment': 'star',
    'Park': 'tree',
    'Beach': 'umbrella-beach',
    'LeisureFacility': 'running',
    'SportsFacility': 'futbol',
    'Recreational': 'tree',
    'Accommodation': 'bed',
    'Information': 'info-circle',
    'Landmark': 'map-marker-alt',
    'Religious': 'place-of-worship',
    'Unknown': 'question',
    'Other': 'map-pin',
}


def get_category_color(category: str) -> str:
    """Get color for a POI category."""
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS['Unknown'])


def get_category_icon(category: str) -> str:
    """Get FontAwesome icon name for a POI category."""
    return CATEGORY_ICONS.get(category, CATEGORY_ICONS['Unknown'])


class TrajectoMapVisualizer:
    """
    Creates interactive Folium maps for trajectory visualization.

    Usage:
        visualizer = TrajectoMapVisualizer()
        map_obj = visualizer.create_trajectory_map(poi_list)
        map_obj.save('trajectory.html')
    """

    def __init__(self, default_location: Tuple[float, float] = (39.9042, 116.4074),
                 default_zoom: int = 12):
        """
        Initialize the map visualizer.

        Args:
            default_location: Default map center (lat, lon). Defaults to Beijing.
            default_zoom: Default zoom level
        """
        self.default_location = default_location
        self.default_zoom = default_zoom

    def _create_base_map(self, center: Optional[Tuple[float, float]] = None,
                         zoom: Optional[int] = None) -> folium.Map:
        """Create a base Folium map with tiles."""
        center = center or self.default_location
        zoom = zoom or self.default_zoom

        m = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles='cartodbpositron'
        )

        # Add layer control for switching tiles
        folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(m)
        folium.TileLayer('cartodbdark_matter', name='Dark Mode').add_to(m)

        return m

    def _create_poi_popup(self, poi: Dict[str, Any], show_explanation: bool = False) -> str:
        """
        Create HTML popup content for a POI marker.

        Args:
            poi: POI data dictionary
            show_explanation: Whether to include recommendation explanation

        Returns:
            HTML string for popup
        """
        label = poi.get('label', 'Unknown POI')
        category = poi.get('category', 'Unknown')
        category_group = poi.get('category_group', '')

        html = f"""
        <div style="min-width: 200px; font-family: Arial, sans-serif;">
            <h4 style="margin: 0 0 8px 0; color: #333;">{label}</h4>
            <p style="margin: 0 0 4px 0; color: #666;">
                <strong>Category:</strong> {category}
            </p>
        """

        if category_group:
            html += f"""
            <p style="margin: 0 0 4px 0; color: #666;">
                <strong>Type:</strong> {category_group}
            </p>
            """

        if poi.get('dwell_time_label'):
            html += f"""
            <p style="margin: 0 0 4px 0; color: #666;">
                <strong>Typical visit:</strong> {poi['dwell_time_label']}
            </p>
            """

        if poi.get('popularity_rank'):
            html += f"""
            <p style="margin: 0 0 4px 0; color: #666;">
                <strong>Popularity:</strong> {poi['popularity_rank']}
            </p>
            """

        if show_explanation and poi.get('explanation'):
            explanation = poi['explanation']
            html += f"""
            <hr style="margin: 8px 0; border: none; border-top: 1px solid #eee;">
            <p style="margin: 0 0 4px 0; color: #333;">
                <strong>Why recommended:</strong>
            </p>
            <p style="margin: 0 0 4px 0; color: #666; font-style: italic;">
                {explanation.get('primary_reason', '')}
            </p>
            """

            if explanation.get('supporting_reasons'):
                html += '<ul style="margin: 4px 0; padding-left: 16px; color: #666;">'
                for reason in explanation['supporting_reasons'][:3]:
                    html += f'<li style="margin: 2px 0;">{reason}</li>'
                html += '</ul>'

        if poi.get('confidence'):
            confidence_colors = {'High': '#4CAF50', 'Medium': '#FFC107', 'Low': '#FF5722'}
            conf_color = confidence_colors.get(poi['confidence'], '#757575')
            html += f"""
            <p style="margin: 8px 0 0 0;">
                <span style="background: {conf_color}; color: white; padding: 2px 8px;
                       border-radius: 4px; font-size: 12px;">
                    {poi['confidence']} Confidence
                </span>
            </p>
            """

        html += '</div>'
        return html

    def _create_visited_popup(self, poi: Dict[str, Any], order: int = None) -> str:
        """Create popup for a visited POI."""
        label = poi.get('label', 'Unknown POI')
        category = poi.get('category', 'Unknown')

        html = f"""
        <div style="min-width: 180px; font-family: Arial, sans-serif;">
            <h4 style="margin: 0 0 8px 0; color: #1565C0;">
                {f'#{order} ' if order else ''}{label}
            </h4>
            <p style="margin: 0 0 4px 0; color: #666;">
                <strong>Category:</strong> {category}
            </p>
        """

        dwell = poi.get('dwell_seconds')
        if dwell and dwell > 0:
            if dwell >= 3600:
                dwell_str = f"{dwell / 3600:.1f}h"
            elif dwell >= 60:
                dwell_str = f"{dwell / 60:.0f}min"
            else:
                dwell_str = f"{dwell:.0f}s"
            html += f"""
            <p style="margin: 0 0 4px 0; color: #666;">
                <strong>Time spent:</strong> {dwell_str}
            </p>
            """

        int_type = poi.get('interaction_type', '')
        if int_type:
            int_label = int_type.replace('Interaction', '').strip()
            int_colors = {
                'Immersive': '#4CAF50', 'Dominant': '#2E7D32',
                'Meaningful': '#FF9800', 'Passing': '#9E9E9E',
            }
            int_color = int_colors.get(int_label, '#757575')
            html += f"""
            <p style="margin: 0 0 4px 0;">
                <span style="background: {int_color}; color: white; padding: 2px 8px;
                       border-radius: 4px; font-size: 11px;">
                    {int_label}
                </span>
            </p>
            """

        html += """
            <p style="margin: 4px 0 0 0;">
                <span style="background: #1565C0; color: white; padding: 2px 8px;
                       border-radius: 4px; font-size: 12px;">
                    Visited
                </span>
            </p>
        </div>
        """
        return html

    def create_trajectory_map(self, poi_sequence: List[Dict[str, Any]],
                              show_path: bool = True,
                              show_order: bool = True,
                              title: str = None) -> folium.Map:
        """
        Create a map showing a trajectory's POI sequence.

        Args:
            poi_sequence: List of POI dictionaries with lat/lon, label, category
            show_path: Whether to draw lines connecting POIs
            show_order: Whether to show visit order numbers
            title: Optional title for the map

        Returns:
            Folium Map object
        """
        # Filter POIs with valid coordinates
        valid_pois = [p for p in poi_sequence
                      if p.get('latitude') and p.get('longitude')]

        if not valid_pois:
            return self._create_base_map()

        # Calculate center
        lats = [p['latitude'] for p in valid_pois]
        lons = [p['longitude'] for p in valid_pois]
        center = (sum(lats) / len(lats), sum(lons) / len(lons))

        # Create map
        m = self._create_base_map(center=center)

        # Add title if provided
        if title:
            title_html = f'''
            <div style="position: fixed; top: 10px; left: 60px; z-index: 1000;
                        background: white; padding: 10px 20px; border-radius: 4px;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial;">
                <h3 style="margin: 0; color: #333;">{title}</h3>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))

        # Add POI markers
        coordinates = []
        for i, poi in enumerate(valid_pois, 1):
            lat, lon = poi['latitude'], poi['longitude']
            coordinates.append([lat, lon])

            category = poi.get('category', 'Unknown')
            color = get_category_color(category)
            icon_name = get_category_icon(category)

            # Create marker with order number
            if show_order:
                icon = folium.DivIcon(
                    html=f'''
                    <div style="background: {color}; color: white; border-radius: 50%;
                                width: 28px; height: 28px; display: flex;
                                align-items: center; justify-content: center;
                                font-weight: bold; font-size: 12px;
                                border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                        {i}
                    </div>
                    ''',
                    icon_size=(28, 28),
                    icon_anchor=(14, 14)
                )
            else:
                icon = folium.Icon(color='blue', icon=icon_name, prefix='fa')

            popup = self._create_visited_popup(poi, order=i if show_order else None)

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup, max_width=300),
                icon=icon,
                tooltip=f"{i}. {poi.get('label', 'POI')}"
            ).add_to(m)

        # Add path connecting POIs
        if show_path and len(coordinates) > 1:
            folium.PolyLine(
                coordinates,
                color='#1565C0',
                weight=3,
                opacity=0.7,
                dash_array='10'
            ).add_to(m)


        # Add layer control
        folium.LayerControl().add_to(m)

        # Fit bounds
        if len(coordinates) > 1:
            m.fit_bounds(coordinates)

        return m

    def _build_info_panel_html(self, title: str, trajectory_info: Dict[str, Any]) -> str:
        """Build expandable trajectory info panel HTML."""
        user = trajectory_info.get('user', '')
        user_short = user.split('#')[-1] if user else 'Unknown'
        purposes = trajectory_info.get('purposes', set())
        nav_styles = trajectory_info.get('nav_styles', set())
        engagement = trajectory_info.get('engagement', '')
        n_visited = trajectory_info.get('n_visited', 0)
        n_recs = trajectory_info.get('n_recommendations', 0)
        classes = trajectory_info.get('classes', [])  # list of (name, ic) tuples

        # Build detail rows
        rows = ''

        rows += f'''
        <tr><td style="color:#888;padding:3px 10px 3px 0;vertical-align:top;white-space:nowrap;">
            User</td><td style="padding:3px 0;color:#333;">{user_short}</td></tr>
        '''

        if purposes:
            purpose_str = ', '.join(sorted(purposes))
            rows += f'''
            <tr><td style="color:#888;padding:3px 10px 3px 0;vertical-align:top;white-space:nowrap;">
                Purpose</td><td style="padding:3px 0;color:#333;">{purpose_str}</td></tr>
            '''

        if nav_styles:
            nav_str = ', '.join(sorted(nav_styles))
            rows += f'''
            <tr><td style="color:#888;padding:3px 10px 3px 0;vertical-align:top;white-space:nowrap;">
                Navigation</td><td style="padding:3px 0;color:#333;">{nav_str}</td></tr>
            '''

        if engagement:
            rows += f'''
            <tr><td style="color:#888;padding:3px 10px 3px 0;vertical-align:top;white-space:nowrap;">
                Engagement</td><td style="padding:3px 0;color:#333;">{engagement}</td></tr>
            '''

        rows += f'''
        <tr><td style="color:#888;padding:3px 10px 3px 0;vertical-align:top;white-space:nowrap;">
            POIs</td><td style="padding:3px 0;color:#333;">{n_visited} visited, {n_recs} recommended</td></tr>
        '''

        # OWL classes section
        class_html = ''
        if classes:
            tags = ''
            for name, ic in classes[:12]:
                if ic >= 5.5:
                    bg, fg = '#7B1FA2', 'white'
                elif ic >= 4.5:
                    bg, fg = '#1565C0', 'white'
                else:
                    bg, fg = '#E0E0E0', '#333'
                tags += (f'<span style="display:inline-block;background:{bg};color:{fg};'
                         f'padding:2px 7px;border-radius:3px;font-size:11px;margin:2px 3px 2px 0;">'
                         f'{name} <span style="opacity:0.7;">({ic:.1f})</span></span>')
            if len(classes) > 12:
                tags += (f'<span style="display:inline-block;color:#888;font-size:11px;'
                         f'margin:2px 0;">+{len(classes) - 12} more</span>')
            class_html = f'''
            <tr><td colspan="2" style="padding:6px 0 2px 0;">
                <span style="color:#888;font-size:12px;">OWL Classes</span><br>
                <div style="margin-top:3px;line-height:1.8;">{tags}</div>
            </td></tr>
            '''

        # Unique panel ID from title hash
        panel_id = f'tp_{abs(hash(title)) % 100000}'

        return f'''
        <div id="{panel_id}" style="position:fixed;top:10px;left:60px;z-index:1000;
                    background:white;border-radius:4px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.3);font-family:Arial;
                    max-width:420px;max-height:80vh;overflow:hidden;">
            <div onclick="(function(){{
                    var d=document.getElementById('{panel_id}_detail');
                    var a=document.getElementById('{panel_id}_arrow');
                    if(d.style.display==='none'){{d.style.display='block';a.innerHTML='&#9660;';}}
                    else{{d.style.display='none';a.innerHTML='&#9654;';}}
                 }})()"
                 style="padding:10px 16px;cursor:pointer;user-select:none;">
                <div style="display:flex;align-items:center;justify-content:space-between;">
                    <h3 style="margin:0;color:#333;font-size:14px;">{title}</h3>
                    <span id="{panel_id}_arrow" style="margin-left:12px;color:#888;font-size:12px;">&#9654;</span>
                </div>
                <div style="font-size:12px;color:#666;margin-top:4px;">
                    <span style="color:#1565C0;">&#9679;</span> Visited &nbsp;
                    <span style="color:#4CAF50;">&#9679;</span> Recommended
                </div>
            </div>
            <div id="{panel_id}_detail" style="display:none;padding:0 16px 12px 16px;
                        border-top:1px solid #eee;max-height:60vh;overflow-y:auto;">
                <table style="font-size:12px;border-collapse:collapse;margin-top:8px;width:100%;">
                    {rows}
                    {class_html}
                </table>
            </div>
        </div>
        '''

    def create_recommendation_map(self, visited_pois: List[Dict[str, Any]],
                                   recommendations: List[Dict[str, Any]],
                                   title: str = None,
                                   show_explanations: bool = True,
                                   trajectory_info: Dict[str, Any] = None) -> folium.Map:
        """
        Create a map showing visited POIs and recommendations.

        Args:
            visited_pois: List of visited POI dictionaries
            recommendations: List of recommended POI dictionaries
            title: Optional title for the map
            show_explanations: Whether to show recommendation explanations in popups
            trajectory_info: Optional dict with trajectory metadata for expandable panel.
                Keys: user, purposes, nav_styles, engagement, n_visited,
                      n_recommendations, classes (list of (name, ic) tuples)

        Returns:
            Folium Map object
        """
        # Filter valid coordinates
        valid_visited = [p for p in visited_pois
                         if p.get('latitude') and p.get('longitude')]
        valid_recs = [p for p in recommendations
                      if p.get('latitude') and p.get('longitude')]

        all_pois = valid_visited + valid_recs

        if not all_pois:
            return self._create_base_map()

        # Calculate center
        lats = [p['latitude'] for p in all_pois]
        lons = [p['longitude'] for p in all_pois]
        center = (sum(lats) / len(lats), sum(lons) / len(lons))

        # Create map
        m = self._create_base_map(center=center)

        # Add title / info panel
        if title:
            if trajectory_info:
                panel_html = self._build_info_panel_html(title, trajectory_info)
            else:
                panel_html = f'''
                <div style="position: fixed; top: 10px; left: 60px; z-index: 1000;
                            background: white; padding: 10px 20px; border-radius: 4px;
                            box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial;">
                    <h3 style="margin: 0 0 8px 0; color: #333;">{title}</h3>
                    <div style="font-size: 12px; color: #666;">
                        <span style="color: #1565C0;">&#9679;</span> Visited &nbsp;
                        <span style="color: #4CAF50;">&#9679;</span> Recommended
                    </div>
                </div>
                '''
            m.get_root().html.add_child(folium.Element(panel_html))

        # Create feature groups
        visited_group = folium.FeatureGroup(name='Visited POIs')
        rec_group = folium.FeatureGroup(name='Recommendations')

        # Add visited POIs (blue numbered markers) with connecting path
        visited_coords = []
        for i, poi in enumerate(valid_visited, 1):
            lat, lon = poi['latitude'], poi['longitude']
            visited_coords.append([lat, lon])

            category = poi.get('category', 'Unknown')
            color = get_category_color(category)

            icon = folium.DivIcon(
                html=f'''
                <div style="background: {color}; color: white; border-radius: 50%;
                            width: 28px; height: 28px; display: flex;
                            align-items: center; justify-content: center;
                            font-weight: bold; font-size: 12px;
                            border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                    {i}
                </div>
                ''',
                icon_size=(28, 28),
                icon_anchor=(14, 14)
            )
            popup = self._create_visited_popup(poi, order=i)

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup, max_width=300),
                icon=icon,
                tooltip=f"#{i}: {poi.get('label', 'POI')}"
            ).add_to(visited_group)

        # Draw trajectory path connecting visited POIs in sequence order
        if len(visited_coords) > 1:
            folium.PolyLine(
                visited_coords,
                color='#1565C0',
                weight=3,
                opacity=0.7,
                dash_array='10'
            ).add_to(visited_group)

        # Add recommended POIs (green markers, sized by score)
        if valid_recs:
            max_score = max(r.get('score', 1) for r in valid_recs)

            for rank, poi in enumerate(valid_recs, 1):
                lat, lon = poi['latitude'], poi['longitude']
                score = poi.get('score', 0)

                # Size marker by score
                norm_score = score / max_score if max_score > 0 else 0.5
                radius = 8 + norm_score * 12  # 8-20 pixel radius

                popup = self._create_poi_popup(poi, show_explanation=show_explanations)

                # Use CircleMarker for size variation
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    color='#2E7D32',
                    fill=True,
                    fill_color='#4CAF50',
                    fill_opacity=0.7,
                    popup=folium.Popup(popup, max_width=350),
                    tooltip=f"#{rank}: {poi.get('label', 'POI')} (score: {score:.2f})"
                ).add_to(rec_group)

        # Add groups to map
        visited_group.add_to(m)
        rec_group.add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Fit bounds
        all_coords = [[p['latitude'], p['longitude']] for p in all_pois]
        if len(all_coords) > 1:
            m.fit_bounds(all_coords)

        return m

    def create_comparison_map(self, trajectory1: List[Dict[str, Any]],
                               trajectory2: List[Dict[str, Any]],
                               label1: str = 'Trajectory 1',
                               label2: str = 'Trajectory 2',
                               title: str = None) -> folium.Map:
        """
        Create a map comparing two trajectories.

        Args:
            trajectory1: First trajectory POI list
            trajectory2: Second trajectory POI list
            label1: Label for first trajectory
            label2: Label for second trajectory
            title: Optional title

        Returns:
            Folium Map object
        """
        valid_t1 = [p for p in trajectory1
                    if p.get('latitude') and p.get('longitude')]
        valid_t2 = [p for p in trajectory2
                    if p.get('latitude') and p.get('longitude')]

        all_pois = valid_t1 + valid_t2

        if not all_pois:
            return self._create_base_map()

        # Calculate center
        lats = [p['latitude'] for p in all_pois]
        lons = [p['longitude'] for p in all_pois]
        center = (sum(lats) / len(lats), sum(lons) / len(lons))

        # Create map
        m = self._create_base_map(center=center)

        # Add title
        if title:
            title_html = f'''
            <div style="position: fixed; top: 10px; left: 60px; z-index: 1000;
                        background: white; padding: 10px 20px; border-radius: 4px;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial;">
                <h3 style="margin: 0 0 8px 0; color: #333;">{title}</h3>
                <div style="font-size: 12px; color: #666;">
                    <span style="color: #1565C0;">&#9679;</span> {label1} &nbsp;
                    <span style="color: #E65100;">&#9679;</span> {label2}
                </div>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(title_html))

        # Find common POIs (by label)
        t1_labels = {p.get('label') for p in valid_t1}
        t2_labels = {p.get('label') for p in valid_t2}
        common_labels = t1_labels & t2_labels

        # Create feature groups
        t1_group = folium.FeatureGroup(name=label1)
        t2_group = folium.FeatureGroup(name=label2)
        common_group = folium.FeatureGroup(name='Common POIs')

        # Add trajectory 1 (blue)
        coords1 = []
        for i, poi in enumerate(valid_t1, 1):
            lat, lon = poi['latitude'], poi['longitude']
            coords1.append([lat, lon])
            label = poi.get('label', '')

            if label in common_labels:
                continue  # Will be added to common group

            icon = folium.DivIcon(
                html=f'''
                <div style="background: #1565C0; color: white; border-radius: 50%;
                            width: 24px; height: 24px; display: flex;
                            align-items: center; justify-content: center;
                            font-weight: bold; font-size: 11px;
                            border: 2px solid white;">
                    {i}
                </div>
                ''',
                icon_size=(24, 24),
                icon_anchor=(12, 12)
            )

            folium.Marker(
                location=[lat, lon],
                icon=icon,
                tooltip=f"{label1} #{i}: {label}"
            ).add_to(t1_group)

        # Add path for trajectory 1
        if len(coords1) > 1:
            folium.PolyLine(
                coords1, color='#1565C0', weight=3, opacity=0.7, dash_array='5'
            ).add_to(t1_group)

        # Add trajectory 2 (orange)
        coords2 = []
        for i, poi in enumerate(valid_t2, 1):
            lat, lon = poi['latitude'], poi['longitude']
            coords2.append([lat, lon])
            label = poi.get('label', '')

            if label in common_labels:
                continue  # Will be added to common group

            icon = folium.DivIcon(
                html=f'''
                <div style="background: #E65100; color: white; border-radius: 50%;
                            width: 24px; height: 24px; display: flex;
                            align-items: center; justify-content: center;
                            font-weight: bold; font-size: 11px;
                            border: 2px solid white;">
                    {i}
                </div>
                ''',
                icon_size=(24, 24),
                icon_anchor=(12, 12)
            )

            folium.Marker(
                location=[lat, lon],
                icon=icon,
                tooltip=f"{label2} #{i}: {label}"
            ).add_to(t2_group)

        # Add path for trajectory 2
        if len(coords2) > 1:
            folium.PolyLine(
                coords2, color='#E65100', weight=3, opacity=0.7, dash_array='5'
            ).add_to(t2_group)

        # Add common POIs (purple)
        for poi in valid_t1:
            label = poi.get('label', '')
            if label in common_labels:
                lat, lon = poi['latitude'], poi['longitude']

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=12,
                    color='#7B1FA2',
                    fill=True,
                    fill_color='#9C27B0',
                    fill_opacity=0.8,
                    tooltip=f"Common: {label}"
                ).add_to(common_group)

        # Add groups
        t1_group.add_to(m)
        t2_group.add_to(m)
        common_group.add_to(m)

        folium.LayerControl().add_to(m)

        # Fit bounds
        all_coords = coords1 + coords2
        if len(all_coords) > 1:
            m.fit_bounds(all_coords)

        return m

    def create_category_heatmap(self, pois: List[Dict[str, Any]],
                                 category_filter: str = None,
                                 title: str = None) -> folium.Map:
        """
        Create a heatmap showing POI category concentrations.

        Args:
            pois: List of POI dictionaries with lat/lon, category
            category_filter: Optional category to filter (show only that category)
            title: Optional title

        Returns:
            Folium Map object
        """
        valid_pois = [p for p in pois
                      if p.get('latitude') and p.get('longitude')]

        if category_filter:
            valid_pois = [p for p in valid_pois
                          if p.get('category') == category_filter]

        if not valid_pois:
            return self._create_base_map()

        # Calculate center
        lats = [p['latitude'] for p in valid_pois]
        lons = [p['longitude'] for p in valid_pois]
        center = (sum(lats) / len(lats), sum(lons) / len(lons))

        # Create map
        m = self._create_base_map(center=center)

        # Add title
        title_text = title or f'POI Heatmap{f" - {category_filter}" if category_filter else ""}'
        title_html = f'''
        <div style="position: fixed; top: 10px; left: 60px; z-index: 1000;
                    background: white; padding: 10px 20px; border-radius: 4px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial;">
            <h3 style="margin: 0; color: #333;">{title_text}</h3>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        # Prepare heat data
        heat_data = [[p['latitude'], p['longitude']] for p in valid_pois]

        # Add heatmap layer
        HeatMap(
            heat_data,
            min_opacity=0.3,
            max_zoom=18,
            radius=15,
            blur=10,
            gradient={'0.4': 'blue', '0.65': 'lime', '1': 'red'}
        ).add_to(m)

        # Add optional marker cluster for individual POIs
        marker_cluster = MarkerCluster(name='Individual POIs')

        for poi in valid_pois:
            lat, lon = poi['latitude'], poi['longitude']
            category = poi.get('category', 'Unknown')
            label = poi.get('label', 'POI')
            color = get_category_color(category)

            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=f"{label} ({category})"
            ).add_to(marker_cluster)

        marker_cluster.add_to(m)

        folium.LayerControl().add_to(m)

        # Fit bounds
        if len(heat_data) > 1:
            m.fit_bounds(heat_data)

        return m

    def create_recommended_route_map(self, start_poi, recommendations,
                                      max_stops=5, title=None,
                                      visited_pois=None):
        """
        Create a map showing visited trajectory with recommendations inserted
        in geographically logical positions.

        Args:
            start_poi: Starting POI dict (legacy) or ignored if visited_pois provided
            recommendations: List of recommended POI dictionaries
            max_stops: Maximum number of recommendation stops to include
            title: Optional title
            visited_pois: Full list of visited POI dicts in visit order.
                          If provided, the route shows the full trajectory context
                          with recommendations interleaved by proximity.

        Returns:
            Folium Map object
        """
        valid_recs = [p for p in recommendations
                      if p.get('latitude') and p.get('longitude')][:max_stops]

        # Build the visited sequence
        if visited_pois:
            visited = [p for p in visited_pois
                       if p.get('latitude') and p.get('longitude')]
        else:
            # Legacy: single start_poi
            if not start_poi.get('latitude') or not start_poi.get('longitude'):
                return self._create_base_map()
            visited = [start_poi]

        if not visited:
            return self._create_base_map()

        all_points = visited + valid_recs
        lats = [p['latitude'] for p in all_points]
        lons = [p['longitude'] for p in all_points]
        center = (sum(lats) / len(lats), sum(lons) / len(lons))

        m = self._create_base_map(center=center)

        # Title
        title_text = title or 'Recommended Route'
        title_html = f'''
        <div style="position: fixed; top: 10px; left: 60px; z-index: 1000;
                    background: white; padding: 10px 20px; border-radius: 4px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-family: Arial;">
            <h3 style="margin: 0 0 8px 0; color: #333;">{title_text}</h3>
            <div style="font-size: 12px; color: #666;">
                <span style="color: #1565C0;">&#9679;</span> Visited &nbsp;
                <span style="color: #4CAF50;">&#9679;</span> Recommended
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        # --- Insert recommendations into the visited sequence by proximity ---
        # For each recommendation, find the visited segment it's closest to
        # and insert it after that visited POI.
        import math

        def _haversine(lat1, lon1, lat2, lon2):
            R = 6371000
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlam = math.radians(lon2 - lon1)
            a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Build the merged route: visited POIs with recs inserted
        # Each element is (poi_dict, type) where type is 'visited' or 'recommended'
        merged = [(p, 'visited') for p in visited]

        for rec in valid_recs:
            rlat, rlon = rec['latitude'], rec['longitude']
            best_idx = 0
            best_dist = float('inf')

            # Find closest visited POI (insert after it)
            for i, (v, _) in enumerate(merged):
                if v.get('latitude') and v.get('longitude'):
                    d = _haversine(rlat, rlon, v['latitude'], v['longitude'])
                    if d < best_dist:
                        best_dist = d
                        best_idx = i

            # Insert after the closest visited POI
            merged.insert(best_idx + 1, (rec, 'recommended'))

        # --- Draw the merged route ---
        coordinates = []
        rec_counter = 0
        visited_counter = 0

        for poi, ptype in merged:
            lat, lon = poi['latitude'], poi['longitude']
            coordinates.append([lat, lon])

            if ptype == 'visited':
                visited_counter += 1
                folium.Marker(
                    location=[lat, lon],
                    popup=f"Visited {visited_counter}: {poi.get('label', 'POI')}",
                    icon=folium.Icon(color='blue', icon='info-sign'),
                    tooltip=f"Visited {visited_counter}: {poi.get('label', 'POI')[:25]}"
                ).add_to(m)
            else:
                rec_counter += 1
                icon = folium.DivIcon(
                    html=f'''
                    <div style="background: #4CAF50; color: white; border-radius: 50%;
                                width: 28px; height: 28px; display: flex;
                                align-items: center; justify-content: center;
                                font-weight: bold; font-size: 12px;
                                border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                        {rec_counter}
                    </div>
                    ''',
                    icon_size=(28, 28),
                    icon_anchor=(14, 14)
                )

                popup = self._create_poi_popup(poi, show_explanation=True)
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup, max_width=350),
                    icon=icon,
                    tooltip=f"Rec {rec_counter}: {poi.get('label', 'POI')[:25]}"
                ).add_to(m)

        # Draw the full route line
        if len(coordinates) > 1:
            folium.PolyLine(
                coordinates,
                color='#2E7D32',
                weight=4,
                opacity=0.8
            ).add_to(m)

        folium.LayerControl().add_to(m)

        if len(coordinates) > 1:
            m.fit_bounds(coordinates)

        return m


def create_poi_legend(categories: List[str]) -> str:
    """
    Create an HTML legend for POI categories.

    Args:
        categories: List of category names

    Returns:
        HTML string for the legend
    """
    legend_items = ""
    for cat in categories:
        color = get_category_color(cat)
        legend_items += f'''
        <div style="display: flex; align-items: center; margin: 4px 0;">
            <span style="background: {color}; width: 16px; height: 16px;
                         border-radius: 50%; display: inline-block; margin-right: 8px;">
            </span>
            <span style="color: #333;">{cat}</span>
        </div>
        '''

    return f'''
    <div style="background: white; padding: 12px; border-radius: 4px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.2); font-family: Arial;">
        <h4 style="margin: 0 0 8px 0; color: #333;">Categories</h4>
        {legend_items}
    </div>
    '''
