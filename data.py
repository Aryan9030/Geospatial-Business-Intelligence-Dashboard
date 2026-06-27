import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION & STYLING
# ============================================================
plt.style.use('dark_background')

COLORS = {
    'bg_dark':      '#0A0E1A',
    'bg_card':      '#111827',
    'bg_panel':     '#1F2937',
    'accent_blue':  '#3B82F6',
    'accent_cyan':  '#06B6D4',
    'accent_green': '#10B981',
    'accent_amber': '#F59E0B',
    'accent_red':   '#EF4444',
    'accent_purple':'#8B5CF6',
    'text_primary': '#F9FAFB',
    'text_muted':   '#9CA3AF',
    'grid':         '#374151',
}

np.random.seed(42)

# ============================================================
# 1. DATA GENERATION — Realistic Multi-Region Sales Data
# ============================================================

def generate_business_data():
    """Generate realistic geospatial business data across US regions."""
    
    # --- Major Metropolitan Areas (Population Centers) ---
    cities = {
        'New York':       {'lat': 40.71, 'lon': -74.00, 'pop': 8.3,  'income': 72000},
        'Los Angeles':    {'lat': 34.05, 'lon': -118.24,'pop': 4.0,  'income': 65000},
        'Chicago':        {'lat': 41.88, 'lon': -87.63, 'pop': 2.7,  'income': 58000},
        'Houston':        {'lat': 29.76, 'lon': -95.37, 'pop': 2.3,  'income': 52000},
        'Phoenix':        {'lat': 33.45, 'lon': -112.07,'pop': 1.6,  'income': 55000},
        'Philadelphia':   {'lat': 39.95, 'lon': -75.17, 'pop': 1.6,  'income': 46000},
        'San Antonio':    {'lat': 29.42, 'lon': -98.49, 'pop': 1.4,  'income': 48000},
        'San Diego':      {'lat': 32.72, 'lon': -117.16,'pop': 1.4,  'income': 70000},
        'Dallas':         {'lat': 32.78, 'lon': -96.80, 'pop': 1.3,  'income': 57000},
        'San Jose':       {'lat': 37.34, 'lon': -121.89,'pop': 1.0,  'income': 115000},
        'Austin':         {'lat': 30.27, 'lon': -97.74, 'pop': 0.98, 'income': 71000},
        'Jacksonville':   {'lat': 30.33, 'lon': -81.66, 'pop': 0.91, 'income': 50000},
        'San Francisco':  {'lat': 37.77, 'lon': -122.42,'pop': 0.87, 'income': 130000},
        'Columbus':       {'lat': 39.96, 'lon': -82.99, 'pop': 0.90, 'income': 54000},
        'Seattle':        {'lat': 47.61, 'lon': -122.33,'pop': 0.74, 'income': 95000},
        'Denver':         {'lat': 39.74, 'lon': -104.98,'pop': 0.72, 'income': 68000},
        'Nashville':      {'lat': 36.17, 'lon': -86.78, 'pop': 0.69, 'income': 60000},
        'Oklahoma City':  {'lat': 35.47, 'lon': -97.52, 'pop': 0.65, 'income': 47000},
        'El Paso':        {'lat': 31.76, 'lon': -106.49,'pop': 0.68, 'income': 42000},
        'Portland':       {'lat': 45.52, 'lon': -122.68,'pop': 0.65, 'income': 72000},
        'Las Vegas':      {'lat': 36.17, 'lon': -115.14,'pop': 0.64, 'income': 52000},
        'Memphis':        {'lat': 35.15, 'lon': -90.05, 'pop': 0.63, 'income': 40000},
        'Louisville':     {'lat': 38.25, 'lon': -85.76, 'pop': 0.61, 'income': 48000},
        'Baltimore':      {'lat': 39.29, 'lon': -76.61, 'pop': 0.59, 'income': 51000},
        'Milwaukee':      {'lat': 43.04, 'lon': -87.91, 'pop': 0.59, 'income': 45000},
        'Albuquerque':    {'lat': 35.08, 'lon': -106.65,'pop': 0.56, 'income': 50000},
        'Tucson':         {'lat': 32.22, 'lon': -110.93,'pop': 0.55, 'income': 44000},
        'Fresno':         {'lat': 36.75, 'lon': -119.77,'pop': 0.54, 'income': 41000},
        'Sacramento':     {'lat': 38.58, 'lon': -121.49,'pop': 0.52, 'income': 60000},
        'Kansas City':    {'lat': 39.10, 'lon': -94.58, 'pop': 0.50, 'income': 55000},
        'Atlanta':        {'lat': 33.75, 'lon': -84.39, 'pop': 0.50, 'income': 62000},
        'Miami':          {'lat': 25.77, 'lon': -80.19, 'pop': 0.47, 'income': 41000},
        'Minneapolis':    {'lat': 44.98, 'lon': -93.27, 'pop': 0.43, 'income': 63000},
        'Tampa':          {'lat': 27.95, 'lon': -82.46, 'pop': 0.40, 'income': 53000},
        'Charlotte':      {'lat': 35.23, 'lon': -80.84, 'pop': 0.87, 'income': 58000},
        'Raleigh':        {'lat': 35.78, 'lon': -78.64, 'pop': 0.47, 'income': 65000},
        'Boston':         {'lat': 42.36, 'lon': -71.06, 'pop': 0.69, 'income': 78000},
        'Detroit':        {'lat': 42.33, 'lon': -83.05, 'pop': 0.67, 'income': 32000},
        'Salt Lake City': {'lat': 40.76, 'lon': -111.89,'pop': 0.20, 'income': 62000},
        'Richmond':       {'lat': 37.54, 'lon': -77.43, 'pop': 0.23, 'income': 53000},
    }
    
    city_records = []
    for city, info in cities.items():
        # Market demand based on population + income + growth trends
        base_demand = info['pop'] * 1000 + info['income'] * 0.01
        demand_score = base_demand * np.random.uniform(0.8, 1.3)
        
        city_records.append({
            'city': city,
            'lat': info['lat'],
            'lon': info['lon'],
            'population_M': info['pop'],
            'median_income': info['income'],
            'market_demand': demand_score,
        })
    
    cities_df = pd.DataFrame(city_records)
    
    # --- Existing Store Locations (Company Presence) ---
    # Intentionally biased toward East Coast + some gaps
    store_cities_primary = [
        'New York', 'Philadelphia', 'Boston', 'Baltimore', 
        'Chicago', 'Detroit', 'Columbus', 'Milwaukee',
        'Los Angeles', 'San Francisco', 'San Diego',
        'Houston', 'Dallas', 'Atlanta', 'Miami',
    ]
    
    stores = []
    store_id = 1
    for city_name in store_cities_primary:
        city_info = cities[city_name]
        num_stores = max(1, int(city_info['pop'] * 3 * np.random.uniform(0.7, 1.4)))
        for _ in range(num_stores):
            lat_jitter = np.random.normal(0, 0.15)
            lon_jitter = np.random.normal(0, 0.2)
            monthly_sales = (
                city_info['pop'] * 120000 +
                city_info['income'] * 2.5 +
                np.random.normal(0, 50000)
            )
            stores.append({
                'store_id': f'STR-{store_id:04d}',
                'city': city_name,
                'lat': city_info['lat'] + lat_jitter,
                'lon': city_info['lon'] + lon_jitter,
                'monthly_sales': max(20000, monthly_sales),
                'store_size_sqft': np.random.choice([800, 1200, 1800, 2500], 
                                                     p=[0.3, 0.35, 0.25, 0.1]),
                'years_operating': np.random.randint(1, 15),
                'customer_rating': np.random.uniform(3.2, 4.9),
                'operating_cost': np.random.uniform(15000, 45000),
            })
            store_id += 1
    
    stores_df = pd.DataFrame(stores)
    stores_df['profit_margin'] = (
        (stores_df['monthly_sales'] - stores_df['operating_cost']) / 
        stores_df['monthly_sales']
    )
    
    # --- Customer Transaction Data (Demand Heatmap Source) ---
    transactions = []
    demand_hotspots = [
        # (lat, lon, intensity, label)
        (40.71, -74.00, 1.0,  'NYC Metro'),
        (34.05, -118.24, 0.85, 'LA Basin'),
        (41.88, -87.63, 0.75, 'Chicago'),
        (37.77, -122.42, 0.7, 'SF Bay'),
        (47.61, -122.33, 0.65, 'Seattle'),
        (33.75, -84.39, 0.6, 'Atlanta'),      # Gap — underserved
        (30.27, -97.74, 0.65, 'Austin'),       # Gap — underserved
        (39.74, -104.98, 0.55, 'Denver'),      # Gap — underserved
        (36.17, -115.14, 0.5, 'Las Vegas'),    # Gap — underserved
        (44.98, -93.27, 0.5, 'Minneapolis'),   # Gap — underserved
        (35.23, -80.84, 0.55, 'Charlotte'),    # Gap — underserved
        (45.52, -122.68, 0.52, 'Portland'),    # Gap — underserved
        (42.36, -71.06, 0.68, 'Boston'),
        (25.77, -80.19, 0.55, 'Miami'),
        (29.76, -95.37, 0.7, 'Houston'),
        (32.78, -96.80, 0.6, 'Dallas'),
    ]
    
    for _ in range(5000):
        hotspot = demand_hotspots[np.random.choice(len(demand_hotspots))]
        lat = hotspot[0] + np.random.normal(0, 1.8)
        lon = hotspot[1] + np.random.normal(0, 2.2)
        lat = np.clip(lat, 24, 49)
        lon = np.clip(lon, -125, -65)
        transactions.append({
            'transaction_lat': lat,
            'transaction_lon': lon,
            'amount': np.random.exponential(85) + 15,
            'category': np.random.choice(
                ['Electronics', 'Apparel', 'Home', 'Sports', 'Beauty'],
                p=[0.25, 0.30, 0.20, 0.15, 0.10]
            ),
        })
    
    transactions_df = pd.DataFrame(transactions)
    
    return cities_df, stores_df, transactions_df, cities

# ============================================================
# 2. SPATIAL ANALYTICS ENGINE
# ============================================================

class SpatialAnalyzer:
    """Core spatial analytics for gap detection and opportunity scoring."""
    
    def __init__(self, stores_df, cities_df):
        self.stores_df  = stores_df
        self.cities_df  = cities_df
        self._build_store_tree()
    
    def _build_store_tree(self):
        """Build KD-Tree for efficient nearest-neighbor queries."""
        store_coords = self.stores_df[['lat', 'lon']].values
        self.store_tree = KDTree(store_coords)
    
    def nearest_store_distance(self, lat, lon):
        """Return distance (degrees) to nearest existing store."""
        dist, _ = self.store_tree.query([lat, lon])
        return dist * 111  # rough km conversion
    
    def compute_city_opportunity(self):
        """
        Opportunity Score = f(demand, gap_from_nearest_store, income, growth)
        High score = High demand + Low presence = EXPAND HERE
        """
        records = []
        for _, row in self.cities_df.iterrows():
            dist_km = self.nearest_store_distance(row['lat'], row['lon'])
            
            # Normalize components
            demand_norm  = row['market_demand'] / self.cities_df['market_demand'].max()
            dist_norm    = min(dist_km / 500, 1.0)          # max 500 km penalty
            income_norm  = row['median_income'] / 130000
            
            # Composite opportunity score (weighted)
            opportunity = (
                0.40 * demand_norm    +   # demand weight
                0.35 * dist_norm      +   # coverage gap weight  
                0.25 * income_norm        # purchasing power weight
            )
            
            # Classification
            if opportunity > 0.65:
                tier = 'PRIME'
            elif opportunity > 0.45:
                tier = 'STRONG'
            elif opportunity > 0.28:
                tier = 'MODERATE'
            else:
                tier = 'LOW'
            
            records.append({
                **row.to_dict(),
                'dist_nearest_store_km': dist_km,
                'opportunity_score': opportunity,
                'tier': tier,
            })
        
        return pd.DataFrame(records).sort_values('opportunity_score', ascending=False)
    
    def cluster_stores(self, n_clusters=5):
        """K-Means clustering of existing stores to identify coverage zones."""
        coords = self.stores_df[['lat', 'lon']].values
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        # Find optimal k via silhouette
        best_k, best_score = n_clusters, -1
        for k in range(3, 9):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(coords_scaled)
            score  = silhouette_score(coords_scaled, labels)
            if score > best_score:
                best_score, best_k = score, k
        
        km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        self.stores_df['cluster'] = km_final.fit_predict(coords_scaled)
        
        cluster_stats = self.stores_df.groupby('cluster').agg(
            store_count=('store_id', 'count'),
            avg_sales=('monthly_sales', 'mean'),
            total_sales=('monthly_sales', 'sum'),
            avg_rating=('customer_rating', 'mean'),
            centroid_lat=('lat', 'mean'),
            centroid_lon=('lon', 'mean'),
        ).reset_index()
        
        return cluster_stats
    
    def demand_heatmap_grid(self, transactions_df, resolution=80):
        """Create demand density grid for heatmap visualization."""
        lat_bins = np.linspace(24, 49, resolution)
        lon_bins = np.linspace(-125, -65, resolution)
        
        heat_grid = np.zeros((resolution, resolution))
        
        for _, row in transactions_df.iterrows():
            lat_idx = np.digitize(row['transaction_lat'], lat_bins) - 1
            lon_idx = np.digitize(row['transaction_lon'], lon_bins) - 1
            if 0 <= lat_idx < resolution and 0 <= lon_idx < resolution:
                heat_grid[lat_idx, lon_idx] += row['amount']
        
        # Smooth with Gaussian kernel
        heat_grid = gaussian_filter(heat_grid, sigma=2.5)
        return lat_bins, lon_bins, heat_grid

# ============================================================
# 3. MAIN VISUALIZATION DASHBOARD
# ============================================================

def create_dashboard():
    """Build the complete 6-panel geospatial analytics dashboard."""
    
    print("⟳ Generating business data...")
    cities_df, stores_df, transactions_df, cities_raw = generate_business_data()
    
    print("⟳ Running spatial analysis...")
    analyzer = SpatialAnalyzer(stores_df, cities_df)
    opportunity_df = analyzer.compute_city_opportunity()
    cluster_stats  = analyzer.cluster_stores()
    lat_bins, lon_bins, heat_grid = analyzer.demand_heatmap_grid(transactions_df)
    
    print("⟳ Rendering dashboard...")
    
    # --- Figure Setup ---
    fig = plt.figure(figsize=(26, 20), facecolor=COLORS['bg_dark'])
    
    gs = GridSpec(
        3, 3,
        figure=fig,
        hspace=0.38,
        wspace=0.32,
        left=0.04, right=0.97,
        top=0.93, bottom=0.04
    )
    
    # Panel assignments
    ax_main    = fig.add_subplot(gs[0, :2])   # Main map — top-left 2/3
    ax_opp     = fig.add_subplot(gs[0, 2])    # Opportunity ranking — top-right
    ax_heat    = fig.add_subplot(gs[1, :2])   # Demand heatmap — middle-left 2/3
    ax_cluster = fig.add_subplot(gs[1, 2])    # Cluster analysis — middle-right
    ax_scatter = fig.add_subplot(gs[2, 0])    # Sales scatter — bottom-left
    ax_gap     = fig.add_subplot(gs[2, 1])    # Gap analysis — bottom-middle
    ax_kpi     = fig.add_subplot(gs[2, 2])    # KPI summary — bottom-right
    
    # ----------------------------------------------------------------
    # PANEL 1: Main Geographic Map — Stores + Opportunity Cities
    # ----------------------------------------------------------------
    _style_panel(ax_main, 'Geographic Business Presence & Expansion Opportunities')
    
    _draw_us_outline(ax_main)
    ax_main.set_xlim(-127, -63)
    ax_main.set_ylim(23, 50)
    
    # Demand heatmap overlay
    heat_extent = [-125, -65, 24, 49]
    heat_norm = heat_grid / heat_grid.max()
    heat_display = np.flipud(heat_norm)
    
    cmap_heat = LinearSegmentedColormap.from_list(
        'demand', ['#0A0E1A', '#1E3A5F', '#0EA5E9', '#F59E0B', '#EF4444'], N=256
    )
    ax_main.imshow(
        heat_display, extent=heat_extent, alpha=0.35,
        cmap=cmap_heat, aspect='auto', zorder=1
    )
    
    # Existing stores
    ax_main.scatter(
        stores_df['lon'], stores_df['lat'],
        c=stores_df['monthly_sales'], cmap='RdYlGn',
        s=60, alpha=0.8, zorder=4,
        vmin=stores_df['monthly_sales'].quantile(0.1),
        vmax=stores_df['monthly_sales'].quantile(0.9),
        edgecolors='white', linewidths=0.3,
        label='Existing Stores'
    )
    
    # Opportunity cities — sized by score, colored by tier
    tier_colors = {
        'PRIME':    COLORS['accent_red'],
        'STRONG':   COLORS['accent_amber'],
        'MODERATE': COLORS['accent_cyan'],
        'LOW':      COLORS['text_muted'],
    }
    
    for tier in ['LOW', 'MODERATE', 'STRONG', 'PRIME']:
        subset = opportunity_df[opportunity_df['tier'] == tier]
        if len(subset) == 0:
            continue
        sizes = 80 + subset['opportunity_score'] * 300
        ax_main.scatter(
            subset['lon'], subset['lat'],
            s=sizes, c=tier_colors[tier],
            marker='D', alpha=0.85, zorder=5,
            edgecolors='white', linewidths=0.5,
            label=f'{tier} Opportunity'
        )
    
    # Label top-5 prime opportunities
    prime = opportunity_df[opportunity_df['tier'] == 'PRIME'].head(5)
    for _, row in prime.iterrows():
        ax_main.annotate(
            row['city'],
            xy=(row['lon'], row['lat']),
            xytext=(row['lon'] + 1.2, row['lat'] + 0.8),
            fontsize=7, color='white', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='white', lw=0.8),
            zorder=6
        )
    
    # Legend
    legend = ax_main.legend(
        loc='lower left', fontsize=7, framealpha=0.3,
        facecolor=COLORS['bg_panel'], edgecolor=COLORS['grid'],
        labelcolor='white', markerscale=0.8
    )
    
    _axis_labels(ax_main, 'Longitude', 'Latitude', fontsize=8)
    
    # Colorbar for store sales
    sm = plt.cm.ScalarMappable(
        cmap='RdYlGn',
        norm=Normalize(
            vmin=stores_df['monthly_sales'].quantile(0.1)/1000,
            vmax=stores_df['monthly_sales'].quantile(0.9)/1000
        )
    )
    cbar = plt.colorbar(sm, ax=ax_main, orientation='vertical',
                        fraction=0.02, pad=0.01, shrink=0.7)
    cbar.set_label('Monthly Sales ($K)', color=COLORS['text_muted'], fontsize=7)
    cbar.ax.yaxis.set_tick_params(color='white', labelsize=6)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # ----------------------------------------------------------------
    # PANEL 2: Opportunity City Ranking
    # ----------------------------------------------------------------
    _style_panel(ax_opp, 'Top Expansion Targets')
    
    top_n = opportunity_df.head(12)
    colors_bar = [tier_colors[t] for t in top_n['tier']]
    
    bars = ax_opp.barh(
        range(len(top_n)),
        top_n['opportunity_score'],
        color=colors_bar, alpha=0.85, height=0.7,
        edgecolor='white', linewidth=0.3
    )
    
    # Add score labels
    for i, (bar, row) in enumerate(zip(bars, top_n.itertuples())):
        ax_opp.text(
            bar.get_width() + 0.01, i,
            f'{row.opportunity_score:.2f}',
            va='center', ha='left', fontsize=7.5,
            color=COLORS['text_primary'], fontweight='bold'
        )
        ax_opp.text(
            0.005, i,
            row.city[:14],
            va='center', ha='left', fontsize=7,
            color='white', fontweight='bold'
        )
    
    ax_opp.set_yticks([])
    ax_opp.set_xlim(0, 1.15)
    ax_opp.set_xlabel('Opportunity Score', color=COLORS['text_muted'], fontsize=8)
    ax_opp.invert_yaxis()
    
    # Tier legend
    tier_patches = [
        mpatches.Patch(color=tier_colors[t], label=t)
        for t in ['PRIME', 'STRONG', 'MODERATE']
    ]
    ax_opp.legend(handles=tier_patches, loc='lower right', fontsize=7,
                  framealpha=0.3, facecolor=COLORS['bg_panel'],
                  edgecolor=COLORS['grid'], labelcolor='white')
    
    # Reference line
    ax_opp.axvline(x=0.65, color=COLORS['accent_red'], linestyle='--',
                   linewidth=0.8, alpha=0.7, label='Prime threshold')
    ax_opp.axvline(x=0.45, color=COLORS['accent_amber'], linestyle='--',
                   linewidth=0.8, alpha=0.7)
    
    # ----------------------------------------------------------------
    # PANEL 3: Demand Heatmap (Isolated)
    # ----------------------------------------------------------------
    _style_panel(ax_heat, 'Customer Demand Density vs Store Coverage Gaps')
    
    _draw_us_outline(ax_heat)
    ax_heat.set_xlim(-127, -63)
    ax_heat.set_ylim(23, 50)
    
    # Demand heatmap (stronger alpha)
    im = ax_heat.imshow(
        heat_display, extent=heat_extent, alpha=0.7,
        cmap=cmap_heat, aspect='auto', zorder=1
    )
    
    # Store coverage circles (service radius = ~250km = ~2.2 degrees)
    for _, s in stores_df.iterrows():
        circle = plt.Circle(
            (s['lon'], s['lat']), radius=2.2,
            color=COLORS['accent_green'], alpha=0.04,
            zorder=2, fill=True
        )
        ax_heat.add_patch(circle)
    
    # Mark gap zones (high demand, far from stores)
    gap_cities = opportunity_df[
        (opportunity_df['tier'].isin(['PRIME', 'STRONG'])) &
        (opportunity_df['dist_nearest_store_km'] > 150)
    ]
    
    ax_heat.scatter(
        gap_cities['lon'], gap_cities['lat'],
        s=120, c=COLORS['accent_red'], marker='X',
        alpha=0.95, zorder=5, edgecolors='white', linewidths=0.5,
        label='Underserved High-Demand Zone'
    )
    
    ax_heat.scatter(
        stores_df['lon'], stores_df['lat'],
        s=25, c=COLORS['accent_green'], marker='o',
        alpha=0.9, zorder=4, edgecolors='none',
        label='Existing Store'
    )
    
    ax_heat.legend(loc='lower left', fontsize=7.5, framealpha=0.35,
                   facecolor=COLORS['bg_panel'], edgecolor=COLORS['grid'],
                   labelcolor='white')
    
    plt.colorbar(im, ax=ax_heat, orientation='vertical',
                 fraction=0.02, pad=0.01, shrink=0.7,
                 label='Demand Intensity').ax.yaxis.set_tick_params(
                     color='white', labelsize=6)
    
    _axis_labels(ax_heat, 'Longitude', 'Latitude', fontsize=8)
    
    # ----------------------------------------------------------------
    # PANEL 4: Store Cluster Analysis
    # ----------------------------------------------------------------
    _style_panel(ax_cluster, 'Store Performance Clusters')
    
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_stats)))
    
    for _, row in cluster_stats.iterrows():
        c_idx = int(row['cluster'])
        mask  = stores_df['cluster'] == c_idx
        
        ax_cluster.scatter(
            stores_df.loc[mask, 'lon'],
            stores_df.loc[mask, 'lat'],
            s=35, color=cluster_colors[c_idx], alpha=0.7,
            edgecolors='none', zorder=3
        )
        
        # Cluster centroid + stats bubble
        ax_cluster.scatter(
            row['centroid_lon'], row['centroid_lat'],
            s=200, color=cluster_colors[c_idx], alpha=1.0,
            marker='*', zorder=5, edgecolors='white', linewidths=0.8
        )
        ax_cluster.annotate(
            f"C{c_idx}\n${row['avg_sales']/1000:.0f}K\n({int(row['store_count'])} stores)",
            xy=(row['centroid_lon'], row['centroid_lat']),
            xytext=(row['centroid_lon'] + 1.5, row['centroid_lat'] + 1.2),
            fontsize=6.5, color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['bg_panel'],
                      edgecolor=cluster_colors[c_idx], alpha=0.8),
            arrowprops=dict(arrowstyle='->', color=cluster_colors[c_idx], lw=0.7),
            zorder=6
        )
    
    _draw_us_outline(ax_cluster)
    ax_cluster.set_xlim(-127, -63)
    ax_cluster.set_ylim(23, 50)
    _axis_labels(ax_cluster, 'Longitude', 'Latitude', fontsize=8)
    
    # ----------------------------------------------------------------
    # PANEL 5: Sales vs Distance Scatter (Opportunity Matrix)
    # ----------------------------------------------------------------
    _style_panel(ax_scatter, 'Demand vs Presence Gap Matrix')
    
    sc = ax_scatter.scatter(
        opportunity_df['dist_nearest_store_km'],
        opportunity_df['market_demand'],
        c=opportunity_df['opportunity_score'],
        cmap='RdYlGn', s=80, alpha=0.85,
        edgecolors='white', linewidths=0.4,
        vmin=0, vmax=1, zorder=3
    )
    
    # Quadrant dividers
    median_dist   = opportunity_df['dist_nearest_store_km'].median()
    median_demand = opportunity_df['market_demand'].median()
    
    ax_scatter.axvline(x=median_dist, color=COLORS['grid'],
                       linestyle='--', linewidth=1, alpha=0.6)
    ax_scatter.axhline(y=median_demand, color=COLORS['grid'],
                       linestyle='--', linewidth=1, alpha=0.6)
    
    # Quadrant labels
    ymax = opportunity_df['market_demand'].max()
    ax_scatter.text(median_dist*0.1, ymax*0.9, 'WELL\nSERVED',
                    fontsize=8, color=COLORS['accent_green'],
                    alpha=0.8, fontweight='bold')
    ax_scatter.text(median_dist*1.7, ymax*0.9, '🎯 PRIME\nOPPORTUNITY',
                    fontsize=8, color=COLORS['accent_red'],
                    alpha=0.9, fontweight='bold')
    ax_scatter.text(median_dist*0.1, ymax*0.05, 'LOW\nPRIORITY',
                    fontsize=8, color=COLORS['text_muted'],
                    alpha=0.7, fontweight='bold')
    ax_scatter.text(median_dist*1.7, ymax*0.05, 'SATURATE\nFIRST',
                    fontsize=8, color=COLORS['accent_amber'],
                    alpha=0.8, fontweight='bold')
    
    # Label top opportunities
    for _, row in opportunity_df.head(5).iterrows():
        ax_scatter.annotate(
            row['city'], fontsize=7, color='white',
            xy=(row['dist_nearest_store_km'], row['market_demand']),
            xytext=(row['dist_nearest_store_km']+8, row['market_demand']+50),
            arrowprops=dict(arrowstyle='->', color='white', lw=0.6),
            zorder=7
        )
    
    plt.colorbar(sc, ax=ax_scatter, fraction=0.04, pad=0.02,
                 label='Opportunity Score').ax.yaxis.set_tick_params(
                     color='white', labelsize=6)
    
    _axis_labels(ax_scatter, 'Distance to Nearest Store (km)',
                 'Market Demand Score', fontsize=8)
    
    # ----------------------------------------------------------------
    # PANEL 6: Gap Analysis — Underserved Cities
    # ----------------------------------------------------------------
    _style_panel(ax_gap, 'Underserved Market Gap Analysis')
    
    gap_df = opportunity_df[
        opportunity_df['dist_nearest_store_km'] > 100
    ].sort_values('opportunity_score', ascending=False).head(10)
    
    x = np.arange(len(gap_df))
    width = 0.35
    
    bars1 = ax_gap.bar(x - width/2,
                       gap_df['market_demand'] / gap_df['market_demand'].max() * 100,
                       width, label='Demand Index', color=COLORS['accent_red'],
                       alpha=0.85, edgecolor='white', linewidth=0.3)
    
    bars2 = ax_gap.bar(x + width/2,
                       (1 - gap_df['dist_nearest_store_km'] /
                        gap_df['dist_nearest_store_km'].max()) * 100,
                       width, label='Proximity Score (inv.)',
                       color=COLORS['accent_cyan'], alpha=0.85,
                       edgecolor='white', linewidth=0.3)
    
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels(
        [c[:10] for c in gap_df['city'].tolist()],
        rotation=35, ha='right', fontsize=7, color=COLORS['text_muted']
    )
    ax_gap.set_ylabel('Normalized Score (0-100)', color=COLORS['text_muted'], fontsize=8)
    ax_gap.set_ylim(0, 115)
    ax_gap.legend(fontsize=7.5, framealpha=0.3, facecolor=COLORS['bg_panel'],
                  edgecolor=COLORS['grid'], labelcolor='white', loc='upper right')
    ax_gap.yaxis.grid(True, alpha=0.2, color=COLORS['grid'])
    ax_gap.set_axisbelow(True)
    
    # ----------------------------------------------------------------
    # PANEL 7: KPI Summary Cards
    # ----------------------------------------------------------------
    _style_panel(ax_kpi, 'Analytics Summary')
    ax_kpi.axis('off')
    
    # Compute KPIs
    total_stores   = len(stores_df)
    total_sales_M  = stores_df['monthly_sales'].sum() / 1e6
    prime_targets  = len(opportunity_df[opportunity_df['tier'] == 'PRIME'])
    avg_gap_km     = opportunity_df['dist_nearest_store_km'].mean()
    best_city      = opportunity_df.iloc[0]['city']
    best_score     = opportunity_df.iloc[0]['opportunity_score']
    coverage_pct   = len(opportunity_df[opportunity_df['dist_nearest_store_km'] < 150]) / \
                     len(opportunity_df) * 100
    
    kpis = [
        ('🏪', 'Active Stores',        f'{total_stores}',        COLORS['accent_green']),
        ('💰', 'Monthly Revenue',       f'${total_sales_M:.1f}M', COLORS['accent_blue']),
        ('🎯', 'Prime Opportunities',   f'{prime_targets} Cities', COLORS['accent_red']),
        ('📍', 'Avg Gap Distance',      f'{avg_gap_km:.0f} km',   COLORS['accent_amber']),
        ('🏆', '#1 Target Market',      best_city,                COLORS['accent_purple']),
        ('📊', 'Market Coverage',       f'{coverage_pct:.0f}%',   COLORS['accent_cyan']),
    ]
    
    cols = 2
    rows = 3
    card_w, card_h = 0.46, 0.28
    pad_x, pad_y   = 0.03, 0.04
    
    for i, (icon, label, value, color) in enumerate(kpis):
        row = i // cols
        col = i % cols
        x = col * (card_w + pad_x)
        y = 1.0 - (row + 1) * (card_h + pad_y)
        
        rect = mpatches.FancyBboxPatch(
            (x, y), card_w, card_h,
            boxstyle='round,pad=0.02',
            facecolor=COLORS['bg_panel'], edgecolor=color,
            linewidth=1.5, transform=ax_kpi.transAxes, zorder=2
        )
        ax_kpi.add_patch(rect)
        
        mid_x = x + card_w / 2
        mid_y = y + card_h / 2
        
        ax_kpi.text(mid_x, mid_y + 0.055, icon,
                    ha='center', va='center', fontsize=14,
                    transform=ax_kpi.transAxes, zorder=3)
        ax_kpi.text(mid_x, mid_y - 0.015, value,
                    ha='center', va='center', fontsize=10,
                    fontweight='bold', color=color,
                    transform=ax_kpi.transAxes, zorder=3)
        ax_kpi.text(mid_x, mid_y - 0.075, label,
                    ha='center', va='center', fontsize=7,
                    color=COLORS['text_muted'],
                    transform=ax_kpi.transAxes, zorder=3)
    
    # ----------------------------------------------------------------
    # Master Title
    # ----------------------------------------------------------------
    fig.text(
        0.5, 0.965,
        '◈  GEOSPATIAL BUSINESS INTELLIGENCE DASHBOARD  ◈',
        ha='center', va='top', fontsize=17, fontweight='bold',
        color=COLORS['text_primary'],
        path_effects=[pe.withStroke(linewidth=3, foreground=COLORS['bg_dark'])]
    )
    fig.text(
        0.5, 0.947,
        'Retail Expansion Opportunity Analysis  ·  Demand–Presence Gap Detection  ·  Market Coverage Optimization',
        ha='center', va='top', fontsize=9, color=COLORS['text_muted']
    )
    
    plt.savefig(
        'geospatial_business_analysis.png',
        dpi=160, bbox_inches='tight',
        facecolor=COLORS['bg_dark'], edgecolor='none'
    )
    print("✅ Dashboard saved → geospatial_business_analysis.png")
    plt.show()
    
    return opportunity_df, stores_df, cluster_stats


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def _style_panel(ax, title):
    """Apply consistent dark styling to a panel."""
    ax.set_facecolor(COLORS['bg_card'])
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS['grid'])
        spine.set_linewidth(0.8)
    ax.tick_params(colors=COLORS['text_muted'], labelsize=7.5)
    ax.xaxis.label.set_color(COLORS['text_muted'])
    ax.yaxis.label.set_color(COLORS['text_muted'])
    ax.title.set_color(COLORS['text_primary'])
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6,
                 color=COLORS['text_primary'])
    ax.grid(False)

def _axis_labels(ax, xlabel, ylabel, fontsize=9):
    ax.set_xlabel(xlabel, color=COLORS['text_muted'], fontsize=fontsize)
    ax.set_ylabel(ylabel, color=COLORS['text_muted'], fontsize=fontsize)

def _draw_us_outline(ax):
    """Draw a simplified US continental outline."""
    us_outline_lon = [
        -124.7, -124.2, -123.5, -122.5, -120.0, -117.1, -114.8, -111.0,
        -109.0, -104.0, -100.0, -97.0, -94.0, -90.0, -88.0, -84.5, -82.0,
        -79.8, -75.5, -72.0, -70.0, -67.0, -66.9,
        -67.0, -68.0, -70.0, -71.5, -75.5, -76.0, -75.8, -75.0, -80.0,
        -81.0, -81.5, -82.0, -84.0, -87.0, -88.5, -90.0, -88.0, -89.0,
        -91.0, -93.6, -96.0, -97.2, -97.0, -96.5, -100.0, -104.0, -109.0,
        -111.0, -114.0, -117.0, -119.3, -120.0, -122.0, -124.3, -124.7,
    ]
    us_outline_lat = [
        48.4, 47.5, 46.2, 45.0, 43.0, 41.0, 37.7, 35.0,
        31.3, 31.7, 31.0, 29.8, 29.0, 28.9, 30.0, 30.5, 30.3,
        29.9, 32.0, 41.0, 42.0, 44.5, 47.5,
        47.5, 47.3, 46.8, 45.0, 44.0, 43.5, 41.0, 38.9, 38.0,
        31.9, 30.4, 29.6, 30.0, 30.4, 30.2, 29.5, 31.0, 33.0,
        34.5, 36.5, 36.0, 33.5, 32.8, 33.8, 36.5, 38.0, 37.0,
        40.5, 41.0, 42.0, 42.8, 45.0, 48.0, 48.3, 48.4,
    ]
    ax.plot(us_outline_lon, us_outline_lat,
            color=COLORS['grid'], linewidth=0.8, alpha=0.6, zorder=2)


# ============================================================
# 5. STRATEGIC REPORT GENERATOR
# ============================================================

def print_strategic_report(opportunity_df, stores_df, cluster_stats):
    """Print a comprehensive text-based strategic expansion report."""
    
    sep = "═" * 68
    
    print(f"\n{sep}")
    print("  GEOSPATIAL EXPANSION STRATEGY REPORT")
    print(f"{sep}\n")
    
    # --- Executive Summary ---
    print("📋 EXECUTIVE SUMMARY")
    print("─" * 68)
    prime = opportunity_df[opportunity_df['tier'] == 'PRIME']
    strong = opportunity_df[opportunity_df['tier'] == 'STRONG']
    
    print(f"  Total Markets Analyzed:    {len(opportunity_df):>6}")
    print(f"  Existing Store Locations:  {len(stores_df):>6}")
    print(f"  Prime Expansion Targets:   {len(prime):>6}")
    print(f"  Strong Expansion Targets:  {len(strong):>6}")
    print(f"  Monthly Revenue (Current): ${stores_df['monthly_sales'].sum()/1e6:>5.1f}M")
    print(f"  Avg Store Monthly Sales:   ${stores_df['monthly_sales'].mean()/1000:>5.0f}K\n")
    
    # --- Top Expansion Targets ---
    print("🎯 TOP 10 EXPANSION TARGETS (Ranked by Opportunity Score)")
    print("─" * 68)
    print(f"  {'Rank':<5} {'City':<18} {'Score':>6} {'Tier':<10} "
          f"{'Gap(km)':>8} {'Demand':>8} {'Income':>8}")
    print(f"  {'─'*4} {'─'*17} {'─'*6} {'─'*9} {'─'*8} {'─'*8} {'─'*8}")
    
    for i, (_, row) in enumerate(opportunity_df.head(10).iterrows(), 1):
        tier_icon = {'PRIME': '🔴', 'STRONG': '🟡',
                     'MODERATE': '🔵', 'LOW': '⚪'}[row['tier']]
        print(f"  {i:<5} {row['city']:<18} {row['opportunity_score']:>5.3f} "
              f" {tier_icon}{row['tier']:<8} {row['dist_nearest_store_km']:>7.0f} "
              f" {row['market_demand']:>7.0f}  ${row['median_income']/1000:>5.0f}K")
    
    # --- Coverage Analysis ---
    print(f"\n📡 MARKET COVERAGE ANALYSIS")
    print("─" * 68)
    bins = [0, 50, 150, 300, 500, float('inf')]
    labels = ['<50km', '50-150km', '150-300km', '300-500km', '>500km']
    counts = pd.cut(
        opportunity_df['dist_nearest_store_km'],
        bins=bins, labels=labels
    ).value_counts().sort_index()
    
    for label, count in counts.items():
        bar_len = int(count / len(opportunity_df) * 40)
        pct = count / len(opportunity_df) * 100
        bar = '█' * bar_len + '░' * (40 - bar_len)
        print(f"  {label:<10} {bar} {count:>3} cities ({pct:>4.1f}%)")
    
    # --- Cluster Performance ---
    print(f"\n🏪 STORE CLUSTER PERFORMANCE")
    print("─" * 68)
    print(f"  {'Cluster':<10} {'Stores':>7} {'Avg Sales':>12} "
          f"{'Total Sales':>14} {'Avg Rating':>11}")
    print(f"  {'─'*9} {'─'*7} {'─'*12} {'─'*14} {'─'*11}")
    
    for _, row in cluster_stats.sort_values('avg_sales', ascending=False).iterrows():
        stars = '★' * int(row['avg_rating']) + '☆' * (5 - int(row['avg_rating']))
        print(f"  Zone-{int(row['cluster']):<5} {int(row['store_count']):>7} "
              f"  ${row['avg_sales']/1000:>8.0f}K  "
              f"  ${row['total_sales']/1e6:>9.1f}M  "
              f"  {stars} {row['avg_rating']:.1f}")
    
    # --- Strategic Recommendations ---
    print(f"\n💡 STRATEGIC RECOMMENDATIONS")
    print("─" * 68)
    top3 = opportunity_df.head(3)
    
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        print(f"\n  Priority {i}: {row['city'].upper()}")
        print(f"  ├─ Opportunity Score: {row['opportunity_score']:.3f} ({row['tier']})")
        print(f"  ├─ Distance to Nearest Store: {row['dist_nearest_store_km']:.0f} km")
        print(f"  ├─ Market Demand Index: {row['market_demand']:.0f}")
        print(f"  ├─ Median Household Income: ${row['median_income']:,}")
        est_revenue = row['market_demand'] * 150
        print(f"  └─ Est. First-Year Monthly Revenue: ${est_revenue/1000:.0f}K–"
              f"${est_revenue*1.4/1000:.0f}K")
    
    print(f"\n{sep}")
    print("  END OF REPORT")
    print(f"{sep}\n")


# ============================================================
# 6. ENTRY POINT
# ============================================================

if __name__ == '__main__':
    opportunity_df, stores_df, cluster_stats = create_dashboard()
    print_strategic_report(opportunity_df, stores_df, cluster_stats)