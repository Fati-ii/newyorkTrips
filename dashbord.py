import streamlit as st
import pandas as pd
import glob
import os
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="NYC Taxi Analytics | Premium", layout="wide", initial_sidebar_state="expanded")

# --- Design System & Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    :root {
        --primary: #FFB800;
        --bg-light: #F0F2F6;
        --card-bg: #FFFFFF;
        --glass-border: rgba(0, 0, 0, 0.1);
        --text-main: #31333F;
        --text-muted: #555555;
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: var(--text-main);
    }

    .stApp {
        background-color: var(--bg-light);
    }

    /* Premium Light Card */
    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--glass-border);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin: 10px 0;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: var(--primary);
        box-shadow: 0 8px 24px rgba(255, 184, 0, 0.15);
    }

    .metric-label {
        color: var(--text-muted);
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: var(--text-main);
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.2;
    }

    /* Style Sidebar Light */
    .stSidebar {
        background-color: #FFFFFF !important;
        border-right: 1px solid var(--glass-border);
    }
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def load_data(path):
    files = glob.glob(os.path.join(path, "*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    return df

def custom_metric(label, value, is_best=False):
    style = 'border: 2px solid var(--primary);' if is_best else 'border: 1px solid var(--glass-border);'
    st.markdown(f"""
    <div class="metric-card" style="{style}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# Chemins des données
BASE_PATH = r"c:\Users\DELL\Downloads\aws\Resulats"
ANALYTICS_PATH = os.path.join(BASE_PATH, "analytics_20260221_000657")
METRICS_PATH = os.path.join(BASE_PATH, "metrics_20260221_000657")
PREDICTIONS_PATH = os.path.join(BASE_PATH, "predictions_20260221_000657")

# --- Sidebar Enhanced ---
with st.sidebar:
    st.image("C:/Users/DELL/Pictures/Screenshots/Screenshot 2026-02-18 222240.png", width=100)
    st.markdown("## Control Center")
    section = st.radio("Navigation", ["Overview", "Time Analytics", "Heatmap Zones", "Predictions"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### System Status")
    st.success("🟢 Online & Connected")

# --- Title Section ---
st.markdown(f"# NYC Taxi Intelligence Portal")
st.markdown("### Insights & Predictions Machine Learning")
st.markdown("---")

# --- Data Loading ---
metrics_df = load_data(METRICS_PATH)
hour_df = load_data(os.path.join(ANALYTICS_PATH, "trips_by_hour"))
dist_df = load_data(os.path.join(ANALYTICS_PATH, "avg_dist_by_day"))
zones_df = load_data(os.path.join(ANALYTICS_PATH, "active_zones"))

if section == "Overview":
    st.markdown("## Performance Summary")
    if not metrics_df.empty:
        metrics_df = metrics_df.sort_values(by='rmse', ascending=True).reset_index(drop=True)
        best_model = metrics_df.iloc[0]
        
        c1, c2, c3 = st.columns(3)
        with c1: custom_metric("Best Model", best_model['model_name'], is_best=True)
        with c2: custom_metric("Minimum RMSE", f"{float(best_model['rmse']):.4f}")
        with c3: custom_metric("Best R2 Score", f"{float(best_model['r2']):.4f}")

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Explore Full Model Metrics", expanded=False):
            st.dataframe(metrics_df, width='stretch')
    else:
        st.warning("No metrics data found.")

    # Show a teaser of analytics
    st.markdown("## Key Insights Teaser")
    t1, t2 = st.columns(2)
    with t1:
        if not hour_df.empty:
            h_col = next((c for c in hour_df.columns if c.lower() in ['nb_trips', 'count', 'num_trips', 'nb_trip']), None)
            hour_col = next((c for c in hour_df.columns if c.lower() in ['pickup_hour', 'hour', 'hr']), None)
            
            if h_col and hour_col:
                peak_hour = hour_df.loc[hour_df[h_col].idxmax()][hour_col]
                st.info(f"💡 Peak traffic hour: **{peak_hour}:00**")
            else:
                st.info("💡 Trip volume or hour data not found.")
    with t2:
        if not zones_df.empty:
            z_count = next((c for c in zones_df.columns if c.lower() in ['nb_trips', 'count', 'num_trips', 'nb_trip']), None)
            if z_count:
                top_zone = zones_df.sort_values(by=z_count, ascending=False).iloc[0]
                lat_val = top_zone.get('lat_zone')
                lon_val = top_zone.get('lon_zone')
                
                if lat_val and lon_val:
                    st.markdown(f"<div class='metric-label'>📍 Most Active Zone (Lat/Lon)</div>", unsafe_allow_html=True)
                    teaser_map_data = pd.DataFrame({'lat': [lat_val], 'lon': [lon_val]})
                    fig_teaser = px.scatter_map(teaser_map_data, lat="lat", lon="lon", 
                                                zoom=13, height=200)
                    fig_teaser.update_traces(marker=dict(size=20, color='#FF4B4B'))
                    fig_teaser.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
                    st.plotly_chart(fig_teaser, use_container_width=True)
                else:
                    st.info("📍 Coordinates missing in Parquet.")

elif section == "Time Analytics":
    st.markdown("##  Temporal Trends")
    template = "plotly_white"
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trips Volume per Hour")
        h_col = 'nb_trips' if 'nb_trips' in hour_df.columns else ('count' if 'count' in hour_df.columns else None)
        hour_col = 'pickup_hour' if 'pickup_hour' in hour_df.columns else ('hour' if 'hour' in hour_df.columns else None)

        if not hour_df.empty and hour_col and h_col:
            fig_hour = px.bar(hour_df.sort_values(by=hour_col), x=hour_col, y=h_col, 
                             color=h_col, color_continuous_scale='Plasma',
                             template=template,
                             labels={hour_col: 'Hour', h_col: 'Trips'})
            st.plotly_chart(fig_hour, width='stretch')
        else:
            st.info("Data not available.")

    with col2:
        st.subheader("Average Distance by Day")
        if not dist_df.empty and 'day' in dist_df.columns and 'avg_distance' in dist_df.columns:
            fig_dist = px.line(dist_df.sort_values(by='day'), x='day', y='avg_distance',
                              template=template, markers=True)
            fig_dist.update_traces(line_color='#FFB800', line_width=4)
            st.plotly_chart(fig_dist, width='stretch')
        else:
            st.info("Data not available.")

elif section == "Heatmap Zones":
    st.markdown("## Geospatial Distribution")
    if not zones_df.empty:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Top Activity Zones")
            c_col = 'nb_trips' if 'nb_trips' in zones_df.columns else ('num_trips' if 'num_trips' in zones_df.columns else ('count' if 'count' in zones_df.columns else None))
            if c_col:
                top_zones = zones_df.sort_values(by=c_col, ascending=False).head(10)
                st.table(top_zones[['lat_zone', 'lon_zone', c_col]])
        
        with col2:
            st.subheader("Interactive Zone Map")
            lat_col = 'lat_zone' if 'lat_zone' in zones_df.columns else ('lat' if 'lat' in zones_df.columns else None)
            lon_col = 'lon_zone' if 'lon_zone' in zones_df.columns else ('lon' if 'lon' in zones_df.columns else None)
            if lat_col and lon_col:
                map_data = zones_df.copy().rename(columns={lat_col: 'lat', lon_col: 'lon'})
                fig_map = px.scatter_map(map_data, lat="lat", lon="lon", 
                                          size=c_col if c_col else None,
                                          color=c_col if c_col else None,
                                          color_continuous_scale='Reds',
                                          zoom=10.5, height=500)
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning("Coordinates missing.")
    else:
        st.info("Zones data not available.")

elif section == "Predictions":
    st.markdown("## ML Model Performance Analysis")
    pred_df = load_data(PREDICTIONS_PATH)
    
    if not pred_df.empty:
        # Sampling for performance (max 50,000 points)
        if len(pred_df) > 50000:
            sample_df = pred_df.sample(50000, random_state=42)
            st.caption(f"Showing a random sample of 50,000 from {len(pred_df):,} total predictions.")
        else:
            sample_df = pred_df

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Actual vs Predicted Comparison")
            fig_scatter = px.scatter(sample_df, x='total_amount', y='prediction', 
                                   opacity=0.4,
                                   labels={'total_amount': 'Actual Total Amount', 'prediction': 'Predicted Amount'},
                                   template="plotly_white")
            fig_scatter.update_traces(marker=dict(color='#FFB800'))
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            st.subheader("Prediction Error Distribution")
            sample_df['error'] = sample_df['prediction'] - sample_df['total_amount']
            fig_hist = px.histogram(sample_df, x='error', nbins=50,
                                  labels={'error': 'Residual (Predicted - Actual)'},
                                  template="plotly_white")
            fig_hist.update_traces(marker_color='#FF4B4B')
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("###  Quick Stats")
        mae = sample_df['error'].abs().mean()
        me = sample_df['error'].mean()
        s1, s2, s3 = st.columns(3)
        with s1: st.metric("Mean Absolute Error (Sample)", f"${mae:.2f}")
        with s2: st.metric("Mean Bias Error (Sample)", f"${me:.2f}")
        with s3: st.metric("Total Records", f"{len(pred_df):,}")
    else:
        st.warning("Prediction data not found or empty.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888; padding: 20px;'>Dashboard réalisé par <span style='color: var(--primary); font-weight: 600;'>Fatimata KA et Ndeye Galass Gaye</span></div>", unsafe_allow_html=True)
