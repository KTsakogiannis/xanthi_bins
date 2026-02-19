import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from shapely.geometry import Point, Polygon
from shapely.ops import transform
import pyproj
import math

# Attempt to import network analysis libraries
try:
    import osmnx as ox
    import networkx as nx
    HAS_NETWORK_LIBS = True
except ImportError:
    HAS_NETWORK_LIBS = False

# --- 1. PAGE CONFIG & STYLING ---
st.set_page_config(layout="wide", page_title="Waste Management Analysis")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-container {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
        gap: 12px;
        justify-content: flex-start;
        align-items: center;
        margin-bottom: 20px;
        margin-top: 10px;
    }
    .metric-card {
        color: white;
        padding: 10px 18px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 1px 1px 4px rgba(0,0,0,0.15);
        min-width: 120px;
    }
            
    /* DYNAMIC SQUARE MAP HACK */
    /* Target the container of the folium map to force a 1:1 aspect ratio */
    iframe[title="streamlit_folium.st_folium"] {
        border-radius: 12px;
        border: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🗑️ Bin Management: Collection Routes & Analysis")

# Initialize session state for persistent data
if 'route_results' not in st.session_state:
    st.session_state.route_results = None

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    return pd.read_csv('ΠΕΡΙΞ_ΕΝΩΠΟΙΗΜΕΝΟ.csv')

df_full = load_data()

# --- 3. SIDEBAR FILTERS ---
st.sidebar.header("🗺️ Map Display Filters")
all_types = sorted(df_full['Type'].unique().tolist())
selected_types = st.sidebar.multiselect("Visible Bin Types", all_types, default=all_types)

all_places = sorted(df_full['Place'].unique().tolist())
selected_places = st.sidebar.multiselect("Neighbourhoods", all_places, default=all_places)

filtered_df = df_full[
    df_full['Type'].isin(selected_types) & 
    df_full['Place'].isin(selected_places)
].copy()

# --- 4. COLOR THEME & METRIC CARDS ---
def get_color(bin_type):
    return {
        'Trash': '#28a745', 'Recycle': '#007bff', 'Organic': '#8b4513', 
        'Underground': '#e83e8c', 'Total': '#343a40'
    }.get(bin_type, '#6c757d')

st.subheader("Map View Statistics")

m_html = '<div class="metric-container">'
m_html += f'<div class="metric-card" style="background-color: {get_color("Total")};">'
m_html += f'<div style="font-size:10px; opacity:0.8; text-transform:uppercase;">Total Visible</div>'
m_html += f'<div style="font-size:22px; font-weight:bold;">{len(filtered_df)}</div></div>'

counts = filtered_df['Type'].value_counts()
for b_type in selected_types:
    m_html += f'<div class="metric-card" style="background-color: {get_color(b_type)};">'
    m_html += f'<div style="font-size:10px; opacity:0.8; text-transform:uppercase;">{b_type}</div>'
    m_html += f'<div style="font-size:22px; font-weight:bold;">{counts.get(b_type, 0)}</div></div>'
m_html += '</div>'
st.write(m_html, unsafe_allow_html=True)

"""# --- 5. MAIN INTERACTIVE MAP ---
c_lat = filtered_df['Lat'].mean() if not filtered_df.empty else df_full['Lat'].mean()
c_lng = filtered_df['Lng'].mean() if not filtered_df.empty else df_full['Lng'].mean()

main_map = folium.Map(location=[c_lat, c_lng], zoom_start=15, tiles="cartodbpositron")
Draw(export=False, position='topleft', 
     draw_options={'polyline': False, 'marker': False, 'circlemarker': False}).add_to(main_map)

grouped = filtered_df.groupby(['Lat', 'Lng'])
for (lat, lng), group in grouped:
    if len(group) > 1:
        icon_html = f'<div style="background-color:red; border:2px solid white; border-radius:50%; width:24px; height:24px; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold; font-size:10px;">{len(group)}</div>'
        folium.Marker([lat, lng], icon=folium.DivIcon(html=icon_html)).add_to(main_map)
    else:
        row = group.iloc[0]
        folium.CircleMarker([lat, lng], radius=6, color='white', weight=1, fill=True, 
                           fill_color=get_color(row['Type']), fill_opacity=0.9).add_to(main_map)

map_output = st_folium(main_map, use_container_width=True, height=600, key="main_map")"""

# --- 5. MAIN INTERACTIVE MAP (SQUARE & DYNAMIC) ---
c_lat = filtered_df['Lat'].mean() if not filtered_df.empty else df_full['Lat'].mean()
c_lng = filtered_df['Lng'].mean() if not filtered_df.empty else df_full['Lng'].mean()

main_map = folium.Map(location=[c_lat, c_lng], zoom_start=15, tiles="cartodbpositron")
Draw(export=False, position='topleft', 
     draw_options={'polyline': False, 'marker': False, 'circlemarker': False}).add_to(main_map)

# Group bins by coordinates
grouped = filtered_df.groupby(['Lat', 'Lng'])

for (lat, lng), group in grouped:
    if len(group) > 1:
        # 1. GENERATE POPUP CONTENT FOR MULTI-POINTS
        counts = group['Type'].value_counts()
        popup_html = f"<div style='min-width: 120px;'><b>Multiple Bins:</b><br>"
        for b_type, count in counts.items():
            popup_html += f"• {b_type}: {count}<br>"
        popup_html += "</div>"
        
        # 2. CREATE CLUSTER MARKER
        icon_html = f'''
            <div style="background-color:red; border:2px solid white; border-radius:50%; 
                        width:26px; height:26px; display:flex; align-items:center; 
                        justify-content:center; color:white; font-weight:bold; font-size:11px;">
                {len(group)}
            </div>'''
        
        folium.Marker(
            [lat, lng], 
            icon=folium.DivIcon(html=icon_html),
            popup=folium.Popup(popup_html, max_width=200) # Added Clickable Popup
        ).add_to(main_map)
        
    else:
        # SINGLE BIN MARKER
        row = group.iloc[0]
        folium.CircleMarker(
            [lat, lng], 
            radius=7, 
            color='white', 
            weight=1, 
            fill=True, 
            fill_color=get_color(row['Type']), 
            fill_opacity=0.9,
            tooltip=f"Type: {row['Type']}" # Show type on hover
        ).add_to(main_map)

# Render map
map_output = st_folium(main_map, use_container_width=True, key="main_map")


# --- 6. SPATIAL ANALYSIS ---
def calculate_area(geom):
    proj = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True).transform
    return transform(proj, geom).area / 1_000_000

if map_output and map_output.get('all_drawings'):
    last = map_output['all_drawings'][-1]
    geom_type = last['geometry']['type']
    sel_df = []
    a_km2 = 0
    
    if geom_type == 'Polygon':
        poly = Polygon(last['geometry']['coordinates'][0])
        sel_df = df_full[df_full.apply(lambda r: poly.contains(Point(r['Lng'], r['Lat'])), axis=1)]
        a_km2 = calculate_area(poly)
    elif geom_type == 'Circle':
        center = last['geometry']['coordinates']
        rad = last['properties']['radius']
        a_km2 = (math.pi * (rad**2)) / 1_000_000
        def hv(lat1, lon1, lat2, lon2):
            R = 6371000
            p1, p2, dl, dp = map(math.radians, [lat1, lat2, lon2-lon1, lat2-lat1])
            a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        sel_df = df_full[df_full.apply(lambda r: hv(center[1], center[0], r['Lat'], r['Lng']) <= rad, axis=1)]

    if len(sel_df) > 0:
        st.divider()
        st.subheader("📊 Selected Area Analysis")
        
        col_table, col_empty = st.columns([1, 1])
        with col_table:
            st.write(f"**Density Table ({a_km2:.4f} km²)**")
            d_rows = []
            f_counts = sel_df['Type'].value_counts()
            for t in sorted(df_full['Type'].unique()):
                c = f_counts.get(t, 0)
                d_rows.append({"Type": t, "Count": c, "Density (/km²)": round(c/a_km2, 2)})
            st.table(pd.DataFrame(d_rows).set_index("Type"))

        if HAS_NETWORK_LIBS:
            st.write("---")
            col_l, col_r = st.columns(2)

            with col_l:
                st.write("**🚶 Average Walkable Symmetric Matrix**")
                if st.button("Calculate Walkable Matrix"):
                    with st.spinner("Analyzing walking paths..."):
                        G_w = ox.graph_from_point((sel_df['Lat'].mean(), sel_df['Lng'].mean()), dist=1000, network_type='walk')
                        clean_w = sel_df.drop_duplicates(subset=['Lat', 'Lng', 'Type']).copy()
                        clean_w['node'] = ox.distance.nearest_nodes(G_w, X=clean_w['Lng'], Y=clean_w['Lat'])
                        u_types = sorted(clean_w['Type'].unique())
                        raw_dists = {}
                        for t1 in u_types:
                            for t2 in u_types:
                                n_f, n_t = clean_w[clean_w['Type'] == t1]['node'].unique(), clean_w[clean_w['Type'] == t2]['node'].unique()
                                paths = []
                                for s in n_f:
                                    tgs = [n for n in n_t if n != s]
                                    if tgs:
                                        try:
                                            d_d = nx.single_source_dijkstra_path_length(G_w, s, weight='length')
                                            v = [d_d[t] for t in tgs if t in d_d]
                                            if v: paths.append(min(v))
                                        except: continue
                                raw_dists[(t1, t2)] = paths
                        w_matrix = []
                        for t1 in u_types:
                            row = {"Type": t1}
                            for t2 in u_types:
                                comb = raw_dists.get((t1, t2), []) + raw_dists.get((t2, t1), [])
                                row[t2] = f"{int(sum(comb)/len(comb))}m" if comb else "-"
                            w_matrix.append(row)
                        st.table(pd.DataFrame(w_matrix).set_index("Type"))

            with col_r:
                st.write("**🚛 Shortest Collection Routes (TSP)**")
                if st.button("Calculate All Optimal Routes"):
                    with st.spinner("Calculating road-legal routes for all bin types..."):
                        try:
                            G_d = ox.graph_from_point((sel_df['Lat'].mean(), sel_df['Lng'].mean()), dist=1200, network_type='drive')
                            u_types = sorted(sel_df['Type'].unique())
                            all_results = {}

                            for t in u_types:
                                t_bins = sel_df[sel_df['Type'] == t].drop_duplicates(subset=['Lat', 'Lng'])
                                if len(t_bins) < 2:
                                    all_results[t] = {"length": 0, "path": [], "stops": []}
                                    continue
                                
                                nodes = ox.distance.nearest_nodes(G_d, X=t_bins['Lng'], Y=t_bins['Lat'])
                                unique_nodes = list(dict.fromkeys(nodes))
                                tsp_g = nx.Graph()
                                for i, n1 in enumerate(unique_nodes):
                                    try:
                                        lengths = nx.single_source_dijkstra_path_length(G_d, n1, weight='length')
                                        for j, n2 in enumerate(unique_nodes):
                                            if i < j and n2 in lengths: tsp_g.add_edge(n1, n2, weight=lengths[n2])
                                    except: continue
                                
                                if len(tsp_g.nodes) > 1:
                                    tsp_nodes = nx.approximation.traveling_salesman_problem(tsp_g, weight='weight', cycle=True)
                                    full_coords = []
                                    total_m = 0
                                    for i in range(len(tsp_nodes)-1):
                                        u, v = tsp_nodes[i], tsp_nodes[i+1]
                                        path = nx.shortest_path(G_d, u, v, weight='length')
                                        total_m += nx.path_weight(G_d, path, weight='length')
                                        for node in path: full_coords.append((G_d.nodes[node]['y'], G_d.nodes[node]['x']))
                                    
                                    # Identify stop sequence
                                    seen = set()
                                    stops = []
                                    for node in tsp_nodes:
                                        if node in unique_nodes and node not in seen:
                                            stops.append((G_d.nodes[node]['y'], G_d.nodes[node]['x']))
                                            seen.add(node)
                                    
                                    all_results[t] = {"length": int(total_m), "path": full_coords, "stops": stops}
                            
                            st.session_state.route_results = all_results
                        except Exception as e:
                            st.error(f"Routing error: {e}")

                # Display Results if calculated
                if st.session_state.route_results:
                    summary = [{"Bin Type": t, "Route Length": f"{data['length']}m"} for t, data in st.session_state.route_results.items()]
                    st.table(pd.DataFrame(summary).set_index("Bin Type"))
                    
                    st.write("---")
                    st.subheader("🗺️ Route Sequence Visualizer")
                    disp_type = st.selectbox("Select Route to Display", list(st.session_state.route_results.keys()))
                    
                    res = st.session_state.route_results[disp_type]
                    if res['length'] > 0:
                        route_map = folium.Map(location=[sel_df['Lat'].mean(), sel_df['Lng'].mean()], zoom_start=16, tiles="cartodbpositron")
                        folium.PolyLine(res['path'], color=get_color(disp_type), weight=5, opacity=0.7).add_to(route_map)
                        
                        for idx, (lat, lng) in enumerate(res['stops']):
                            num_icon = f'<div style="background-color:{get_color(disp_type)}; color:white; border:2px solid white; border-radius:50%; width:22px; height:22px; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:11px;">{idx+1}</div>'
                            folium.Marker([lat, lng], icon=folium.DivIcon(html=num_icon), tooltip=f"Stop {idx+1}").add_to(route_map)
                        
                        st_folium(route_map, width=1400, height=500, key="route_viewer")
                    else:
                        st.info("Not enough bins of this type for a route.")