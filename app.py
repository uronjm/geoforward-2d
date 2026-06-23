"""
GeoForward 2D — Aplikasi Web Forward Modelling Geolistrik
Berbasis Streamlit
UI Design: Enterprise Geoscience Platform (Referensi 4)
"""

import streamlit as st
import numpy as np
import pandas as pd
import io, time

from inv_parser     import parse_inv_file, generate_demo_inv
from forward_model  import run_forward, compute_layer_averages, classify_material, ARRAY_CONFIGS
from plotting       import (plot_pseudosection, plot_true_section, plot_rms_convergence,
                            plot_comparison_spasi, plot_comparison_array, plot_layer_bar)
from app_extra      import build_html_report

# ── Konfigurasi halaman ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoForward 2D",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS — Enterprise Geoscience Platform (Referensi 4)
# Terinspirasi: Esri ArcGIS Pro, Schlumberger Petrel, Kingdom Suite
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ─────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
}

/* ── Streamlit chrome hide ────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stSidebarNav"] { display: none; }
.block-container {
  padding-top: 0 !important;
  padding-bottom: 1rem !important;
  max-width: 100% !important;
}

/* ── TOP BAR ──────────────────────────────────────── */
.geo-topbar {
  background: #042c53;
  padding: 0 20px;
  height: 44px;
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 0;
  border-bottom: 1px solid #0c447c;
}
.geo-brand {
  display: flex;
  align-items: center;
  gap: 9px;
  text-decoration: none;
}
.geo-brand-icon {
  width: 26px;
  height: 26px;
  background: #185fa5;
  border-radius: 5px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  color: #e6f1fb;
  flex-shrink: 0;
}
.geo-brand-name {
  font-size: 13px;
  font-weight: 600;
  color: #b5d4f4;
  letter-spacing: 0.02em;
}
.geo-brand-ver {
  font-size: 10px;
  color: #378add;
  background: #0c447c;
  padding: 1px 6px;
  border-radius: 3px;
  font-family: 'JetBrains Mono', monospace;
}
.geo-nav {
  display: flex;
  gap: 0;
  margin-left: 10px;
}
.geo-nav-item {
  font-size: 11.5px;
  color: #85b7eb;
  padding: 0 13px;
  height: 44px;
  display: inline-flex;
  align-items: center;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  letter-spacing: 0.01em;
  transition: background 0.15s;
}
.geo-nav-item:hover {
  background: rgba(55,138,221,0.12);
  color: #e6f1fb;
}
.geo-nav-item.active {
  color: #e6f1fb;
  border-bottom: 2px solid #378add;
  background: rgba(55,138,221,0.1);
}
.geo-status {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}
.geo-status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #1d9e75;
}
.geo-status-dot.inactive {
  background: #888780;
}
.geo-status-txt {
  font-size: 11px;
  color: #5dcaa5;
  font-family: 'JetBrains Mono', monospace;
}
.geo-status-txt.inactive {
  color: #888780;
}

/* ── MENU BAR ─────────────────────────────────────── */
.geo-menubar {
  background: #f4f6f9;
  border-bottom: 1px solid #dde2ea;
  display: flex;
  align-items: center;
  padding: 0 16px;
  height: 30px;
  gap: 0;
}
.geo-menu-item {
  font-size: 11px;
  color: #4a5568;
  padding: 0 10px;
  height: 30px;
  display: inline-flex;
  align-items: center;
  cursor: pointer;
  border-radius: 0;
}
.geo-menu-item:hover {
  background: #e8edf4;
  color: #1a2744;
}
.geo-menu-sep {
  width: 1px;
  background: #dde2ea;
  height: 18px;
  margin: 0 4px;
}

/* ── SIDEBAR ──────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: #f8fafc;
  border-right: 1px solid #dde2ea;
}
section[data-testid="stSidebar"] > div:first-child {
  padding-top: 0 !important;
}
.sidebar-panel-header {
  background: #f0f3f8;
  border-bottom: 1px solid #dde2ea;
  padding: 7px 14px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: #6b7a99;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0;
}
.sidebar-section-label {
  font-size: 9.5px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: #8899bb;
  padding: 10px 14px 5px;
  display: block;
  border-top: 1px solid #e8edf4;
  margin-top: 6px;
}
.sidebar-section-label:first-child {
  border-top: none;
  margin-top: 0;
}
.file-status-ok {
  background: #f0faf5;
  border: 1px solid #9fe1cb;
  border-radius: 5px;
  padding: 8px 11px;
  font-size: 10.5px;
  color: #085041;
  margin: 6px 14px;
  line-height: 1.55;
}
.file-status-warn {
  background: #faeeda;
  border: 1px solid #fac775;
  border-radius: 5px;
  padding: 8px 11px;
  font-size: 10.5px;
  color: #633806;
  margin: 6px 14px;
  line-height: 1.55;
}
.sidebar-prop-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 14px;
  border-bottom: 1px solid #f0f3f8;
  font-size: 10.5px;
}
.sidebar-prop-key {
  color: #8899bb;
}
.sidebar-prop-val {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 500;
  color: #1a2744;
}
.sidebar-footer {
  font-size: 9.5px;
  color: #aab2c8;
  padding: 10px 14px;
  line-height: 1.6;
  border-top: 1px solid #e8edf4;
  margin-top: 8px;
}

/* ── MAIN CONTENT ─────────────────────────────────── */

/* Section headers — styled like panel labels */
.sec-head {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: #6b7a99;
  padding: 14px 0 6px;
  border-bottom: 1px solid #dde2ea;
  margin-bottom: 12px;
}
.sec-sub {
  font-size: 11px;
  color: #8899bb;
  margin: -8px 0 12px;
}

/* KPI / Stat Cards */
.kpi-row {
  display: flex;
  gap: 1px;
  background: #dde2ea;
  border: 1px solid #dde2ea;
  border-radius: 6px;
  overflow: hidden;
  margin-bottom: 16px;
}
.kpi-card {
  background: #ffffff;
  padding: 11px 16px;
  flex: 1;
}
.kpi-val {
  font-size: 20px;
  font-weight: 500;
  color: #0c447c;
  font-family: 'JetBrains Mono', monospace;
  line-height: 1.2;
}
.kpi-label {
  font-size: 9.5px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #8899bb;
  margin-top: 3px;
}
.kpi-status-ok {
  display: inline-block;
  font-size: 9px;
  background: #e1f5ee;
  color: #085041;
  border: 1px solid #9fe1cb;
  padding: 1px 6px;
  border-radius: 2px;
  margin-top: 4px;
}
.kpi-status-warn {
  display: inline-block;
  font-size: 9px;
  background: #faeeda;
  color: #633806;
  border: 1px solid #fac775;
  padding: 1px 6px;
  border-radius: 2px;
  margin-top: 4px;
}

/* stat-card (lama — mapped ke kpi-card style) */
.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0; }
.stat-card {
  background: #ffffff;
  border: 1px solid #dde2ea;
  border-radius: 6px;
  padding: 11px 15px;
  flex: 1;
  min-width: 110px;
}
.stat-card .sv {
  font-size: 1.35rem;
  font-weight: 600;
  color: #0c447c;
  line-height: 1.2;
  font-family: 'JetBrains Mono', monospace;
}
.stat-card .sl {
  font-size: 9.5px;
  color: #8899bb;
  margin-top: 3px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
}

/* Info / Warn / OK boxes */
.info-box {
  background: #eff6ff;
  border: 1px solid #b5d4f4;
  border-radius: 5px;
  padding: 11px 15px;
  margin: 8px 0;
  font-size: 12px;
  color: #0c447c;
  line-height: 1.6;
}
.warn-box {
  background: #faeeda;
  border: 1px solid #fac775;
  border-radius: 5px;
  padding: 10px 15px;
  margin: 8px 0;
  font-size: 12px;
  color: #633806;
  line-height: 1.6;
}
.ok-box {
  background: #e1f5ee;
  border: 1px solid #9fe1cb;
  border-radius: 5px;
  padding: 10px 15px;
  margin: 8px 0;
  font-size: 12px;
  color: #085041;
  line-height: 1.6;
}

/* Layer rows */
.layer-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  border-radius: 4px;
  margin-bottom: 4px;
  border: 1px solid #dde2ea;
  background: #ffffff;
}
.layer-dot { width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0; }
.layer-info { flex: 1; }
.layer-name { font-size: 11.5px; font-weight: 500; color: #1a2744; }
.layer-val  { font-size: 10.5px; color: #8899bb; }
.layer-rho  {
  font-size: 12px;
  font-weight: 600;
  color: #185fa5;
  min-width: 74px;
  text-align: right;
  font-family: 'JetBrains Mono', monospace;
}

/* Properties panel (right column) */
.prop-table {
  border: 1px solid #dde2ea;
  border-radius: 5px;
  overflow: hidden;
  font-size: 11px;
}
.prop-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 12px;
  border-bottom: 1px solid #f0f3f8;
}
.prop-row:last-child { border-bottom: none; }
.prop-key { color: #8899bb; font-size: 10.5px; }
.prop-val {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 500;
  color: #1a2744;
  font-size: 10.5px;
}

/* Tab overrides — match enterprise style */
div[data-baseweb="tab-list"] {
  background: #f4f6f9;
  border-bottom: 1px solid #dde2ea;
  gap: 0;
  padding: 0 8px;
}
div[data-baseweb="tab"] {
  font-size: 11.5px !important;
  font-weight: 500 !important;
  color: #6b7a99 !important;
  padding: 8px 14px !important;
  background: transparent !important;
  border-bottom: 2px solid transparent;
}
div[aria-selected="true"][data-baseweb="tab"] {
  color: #0c447c !important;
  border-bottom: 2px solid #185fa5 !important;
  background: #ffffff !important;
}

/* Toolbar badge row */
.toolbar-row {
  background: #f4f6f9;
  border-bottom: 1px solid #dde2ea;
  padding: 5px 16px;
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.toolbar-badge {
  font-size: 10px;
  padding: 3px 9px;
  border: 1px solid #b5d4f4;
  border-radius: 3px;
  background: #e6f1fb;
  color: #0c447c;
  font-weight: 500;
  letter-spacing: 0.02em;
}
.toolbar-sep {
  width: 1px;
  background: #dde2ea;
  height: 18px;
  margin: 0 4px;
}
.toolbar-label {
  font-size: 10px;
  color: #8899bb;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

/* Step / guide */
.guide-step {
  display: flex;
  gap: 14px;
  margin-bottom: 16px;
  align-items: flex-start;
}
.guide-num {
  background: #185fa5;
  color: white;
  border-radius: 50%;
  width: 26px;
  height: 26px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  font-weight: 600;
  flex-shrink: 0;
  margin-top: 2px;
}
.guide-body { flex: 1; }
.guide-title { font-size: 13px; font-weight: 600; color: #0c447c; margin-bottom: 3px; }
.guide-desc  { font-size: 12px; color: #374151; line-height: 1.55; }
.guide-code  {
  background: #f4f6f9;
  border: 1px solid #dde2ea;
  border-radius: 4px;
  padding: 5px 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: #0c447c;
  margin-top: 5px;
  display: block;
}

/* Rec cards */
.rec-card { border-radius: 5px; padding: 14px; height: 100%; }

/* Status bar */
.geo-statusbar {
  background: #f4f6f9;
  border-top: 1px solid #dde2ea;
  padding: 4px 18px;
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 10px;
  color: #8899bb;
  font-family: 'JetBrains Mono', monospace;
  margin-top: 24px;
}
.geo-sb-sep {
  width: 1px;
  background: #dde2ea;
  height: 12px;
}

/* Footer */
.footer {
  text-align: center;
  color: #aab2c8;
  font-size: 10.5px;
  padding: 10px 0 4px;
  border-top: 1px solid #e8edf4;
  margin-top: 16px;
  font-family: 'JetBrains Mono', monospace;
}

/* Streamlit metric overrides */
div[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid #dde2ea;
  border-radius: 5px;
  padding: 10px 14px !important;
}
div[data-testid="stMetricLabel"] {
  font-size: 9.5px !important;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: #8899bb !important;
}
div[data-testid="stMetricValue"] {
  font-size: 1.25rem !important;
  font-weight: 600 !important;
  color: #0c447c !important;
  font-family: 'JetBrains Mono', monospace !important;
}

/* Button overrides */
.stButton > button {
  background: #185fa5;
  color: #ffffff;
  border: none;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.02em;
  padding: 7px 14px;
  transition: background 0.15s;
}
.stButton > button:hover {
  background: #0c447c;
  color: #ffffff;
  border: none;
}
.stButton > button[disabled] {
  background: #dde2ea;
  color: #aab2c8;
}

/* Download button */
.stDownloadButton > button {
  background: #ffffff;
  color: #185fa5;
  border: 1px solid #b5d4f4;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
}
.stDownloadButton > button:hover {
  background: #e6f1fb;
  color: #0c447c;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
  border: 1px solid #dde2ea;
  border-radius: 5px;
  overflow: hidden;
}

/* Expander */
div[data-testid="stExpander"] {
  border: 1px solid #dde2ea;
  border-radius: 5px;
}
div[data-testid="stExpander"] summary {
  font-size: 12px;
  font-weight: 500;
  color: #1a2744;
  padding: 8px 14px;
  background: #f4f6f9;
}

/* Plotly chart container */
div[data-testid="stPlotlyChart"] {
  border: 1px solid #dde2ea;
  border-radius: 5px;
  padding: 4px;
  background: #ffffff;
}

/* File uploader */
div[data-testid="stFileUploader"] {
  border-radius: 5px;
}

/* Multiselect tags */
span[data-baseweb="tag"] {
  background: #e6f1fb !important;
  color: #0c447c !important;
}

/* Sidebar multiselect */
section[data-testid="stSidebar"] span[data-baseweb="tag"] {
  background: #dde2ea !important;
  color: #1a2744 !important;
}

</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
for k, v in {
    "inv_parsed": None,
    "forward_results": {},
    "demo_loaded": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR — Enterprise header
# ══════════════════════════════════════════════════════════════════════════════
inv_state  = st.session_state.inv_parsed
has_data   = inv_state is not None and inv_state.get("success")
fwd_state  = st.session_state.forward_results
has_fwd    = len(fwd_state) > 0

if has_data:
    fname    = inv_state.get("filename", "")
    rms_val  = inv_state.get("final_rms", 0)
    status_html = f"""
      <div class="geo-status-dot"></div>
      <span class="geo-status-txt">Model aktif &nbsp;—&nbsp; {fname}</span>
    """
else:
    status_html = """
      <div class="geo-status-dot inactive"></div>
      <span class="geo-status-txt inactive">Belum ada data — upload .INV atau gunakan Demo</span>
    """

st.markdown(f"""
<div class="geo-topbar">
  <div class="geo-brand">
    <div class="geo-brand-icon">&#x25C8;</div>
    <span class="geo-brand-name">GeoForward 2D</span>
    <span class="geo-brand-ver">v2.4</span>
  </div>
  <nav class="geo-nav">
    <span class="geo-nav-item active">Analysis</span>
    <span class="geo-nav-item">View</span>
    <span class="geo-nav-item">Export</span>
    <span class="geo-nav-item">Help</span>
  </nav>
  <div class="geo-status">
    {status_html}
  </div>
</div>
<div class="geo-menubar">
  <span class="geo-menu-item">New Session</span>
  <span class="geo-menu-item">Open .INV</span>
  <div class="geo-menu-sep"></div>
  <span class="geo-menu-item">Run Forward</span>
  <span class="geo-menu-item">Compare Arrays</span>
  <div class="geo-menu-sep"></div>
  <span class="geo-menu-item">Generate Report</span>
  <span class="geo-menu-item">Export CSV</span>
</div>
""", unsafe_allow_html=True)

# Toolbar badges
if has_data:
    inv_disp = st.session_state.inv_parsed
    rms_disp = inv_disp.get("final_rms", 0)
    arrays_active = "W-Sch &nbsp; Wenner &nbsp; DD"
    spacing_active = "1m &nbsp; 2m &nbsp; 3m"
    rms_badge_cls  = "kpi-status-ok" if rms_disp and rms_disp <= 5 else "kpi-status-warn"
    st.markdown(f"""
    <div class="toolbar-row">
      <span class="toolbar-label">Array:</span>
      <span class="toolbar-badge">Wenner-Schlumberger</span>
      <span class="toolbar-badge">Wenner</span>
      <span class="toolbar-badge">Dipole-Dipole</span>
      <div class="toolbar-sep"></div>
      <span class="toolbar-label">Spasi:</span>
      <span class="toolbar-badge">1 m</span>
      <span class="toolbar-badge">2 m</span>
      <span class="toolbar-badge">3 m</span>
      <div class="toolbar-sep"></div>
      <span class="toolbar-label">RMS:</span>
      <span class="{rms_badge_cls}" style="font-size:10px;padding:2px 8px;border-radius:3px">{rms_disp:.1f}%</span>
      <div class="toolbar-sep"></div>
      <span class="toolbar-label">Simulasi: {len(fwd_state)} run selesai</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Project Explorer / Control Panel
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown('<div class="sidebar-panel-header">Project Explorer &nbsp; / &nbsp; Control Panel</div>',
                unsafe_allow_html=True)

    # ── 1. DATA INPUT ──────────────────────────────────────────────────────
    st.markdown('<span class="sidebar-section-label">1. Data Input</span>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload file .INV (RES2DINV)",
        type=["inv", "INV", "txt"],
        help="Output file .INV dari RES2DINV versi 3.x – 10.x",
        label_visibility="collapsed",
    )

    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        demo_btn  = st.button("Demo", use_container_width=True,
                              help="Muat data sintetik lintasan 30 m")
    with col_d2:
        clear_btn = st.button("Reset", use_container_width=True,
                              help="Hapus semua data")

    if clear_btn:
        st.session_state.inv_parsed      = None
        st.session_state.forward_results = {}
        st.session_state.demo_loaded     = False
        st.rerun()

    if demo_btn:
        with st.spinner("Memuat data demo..."):
            content = generate_demo_inv()
            parsed  = parse_inv_file(content)
            parsed["filename"] = "demo_lintasan_30m_WS.INV"
            parsed["is_demo"]  = True
            st.session_state.inv_parsed      = parsed
            st.session_state.demo_loaded     = True
            st.session_state.forward_results = {}
        st.success("File demo dimuat.")

    if uploaded and not st.session_state.demo_loaded:
        content = uploaded.read().decode("utf-8", errors="replace")
        with st.spinner("Memproses file..."):
            parsed = parse_inv_file(content)
            parsed["filename"] = uploaded.name
            parsed["is_demo"]  = False
            st.session_state.inv_parsed      = parsed
            st.session_state.forward_results = {}
        if parsed["success"]:
            st.success("File berhasil dibaca.")
        else:
            st.error(f"Error: {parsed.get('error', 'Format tidak dikenal')}")

    # File status panel
    inv = st.session_state.inv_parsed
    if inv and inv.get("success"):
        rms = inv.get("final_rms", 0)
        rms_ok = rms and rms <= 5
        box_cls = "file-status-ok" if rms_ok else "file-status-warn"
        icon    = "✓" if rms_ok else "⚠"
        st.markdown(f"""
        <div class="{box_cls}">
          <b>{icon} {inv.get("filename","")}</b><br>
          RMS {rms:.1f}% &nbsp;·&nbsp; {inv.get("n_cols",0)}×{inv.get("n_rows",0)} blok &nbsp;·&nbsp; a = {inv.get("electrode_spacing",0)} m
        </div>
        """, unsafe_allow_html=True)

        # Properties mini-table
        st.markdown("""
        <div class="prop-table" style="margin:6px 14px">
        """, unsafe_allow_html=True)
        props = [
            ("Iterasi",     inv.get("iteration_used", "–")),
            ("Grid kolom",  inv.get("n_cols", "–")),
            ("Grid baris",  inv.get("n_rows", "–")),
            ("Spasi a",     f"{inv.get('electrode_spacing','–')} m"),
            ("RMS",         f"{rms:.1f} %"),
        ]
        rows_html = ""
        for k, v in props:
            rows_html += f'<div class="prop-row"><span class="prop-key">{k}</span><span class="prop-val">{v}</span></div>'
        st.markdown(rows_html + "</div>", unsafe_allow_html=True)

    # ── 2. FORWARD PARAMETERS ─────────────────────────────────────────────
    st.markdown('<span class="sidebar-section-label">2. Parameter Forward</span>', unsafe_allow_html=True)

    spasi_opts = st.multiselect(
        "Spasi elektroda (m)",
        options=[1, 2, 3, 4, 5],
        default=[1, 2, 3],
        help="Semua spasi yang dipilih akan disimulasikan",
    )
    array_opts = st.multiselect(
        "Konfigurasi elektroda",
        options=["Wenner-Schlumberger", "Wenner", "Dipole-Dipole"],
        default=["Wenner-Schlumberger", "Wenner", "Dipole-Dipole"],
    )

    # ── 3. ADVANCED ───────────────────────────────────────────────────────
    st.markdown('<span class="sidebar-section-label">3. Opsi Lanjutan</span>', unsafe_allow_html=True)

    noise_pct = st.slider("Noise sintetik (%)", 0, 20, 0, 1,
                          help="Tambahkan noise acak ke data forward")
    show_vals = st.checkbox("Tampilkan nilai ρa di plot", False)

    col_lo, col_hi = st.columns(2)
    with col_lo:
        rho_min_inp = st.number_input("ρ min (Ω·m)", 1.0, 9999.0, 10.0, 1.0)
    with col_hi:
        rho_max_inp = st.number_input("ρ maks (Ω·m)", 1.0, 99999.0, 2000.0, 100.0)

    # ── RUN BUTTON ────────────────────────────────────────────────────────
    st.markdown('<span class="sidebar-section-label">4. Eksekusi</span>', unsafe_allow_html=True)

    can_run = (inv is not None and inv.get("success") and
               len(spasi_opts) > 0 and len(array_opts) > 0)

    run_btn = st.button(
        "▶  Run Forward Modelling",
        use_container_width=True,
        type="primary",
        disabled=not can_run,
    )

    if not can_run and inv is None:
        st.markdown('<div class="file-status-warn" style="margin-top:4px">Upload .INV atau gunakan Demo terlebih dahulu.</div>',
                    unsafe_allow_html=True)

    # ── Sidebar footer ────────────────────────────────────────────────────
    st.markdown("""
    <div class="sidebar-footer">
      GeoForward 2D v2.4<br>
      Kalkulasi: Weighted IDW<br>
      Ref: Loke (2004) RES2DMOD<br>
      Python · Streamlit · Plotly
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD RUN
# ══════════════════════════════════════════════════════════════════════════════
if run_btn and inv:
    rho_mat = inv.get("rho_matrix")
    x_pos   = inv.get("x_positions", [])
    depths  = inv.get("depths", [])
    n_elec  = inv.get("n_electrodes", 16) or 16

    if rho_mat is None or rho_mat.size == 0:
        st.error("Model resistivitas kosong. Periksa format file .INV.")
    else:
        total = len(spasi_opts) * len(array_opts)
        prog  = st.progress(0, "Menjalankan simulasi forward modelling...")
        done  = 0
        results = {}
        t0 = time.time()

        for s in spasi_opts:
            for arr in array_opts:
                key = f"{s}_{arr}"
                cfg = ARRAY_CONFIGS.get(arr, {})
                res = run_forward(
                    rho_matrix=rho_mat, x_positions=x_pos, depths=depths,
                    array_name=arr, electrode_spacing=float(s),
                    n_max=cfg.get("max_n_default", 6),
                    n_electrodes=n_elec,
                    noise_level=noise_pct / 100,
                )
                results[key] = res
                done += 1
                prog.progress(done / total, f"Simulasi {done}/{total}: {arr} a={s}m  ✓")

        st.session_state.forward_results = results
        prog.empty()
        elapsed = time.time() - t0
        st.success(f"✓  {total} simulasi selesai dalam {elapsed:.1f} detik — lihat hasil di tab di bawah.")


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE — belum ada data
# ══════════════════════════════════════════════════════════════════════════════
inv = st.session_state.inv_parsed
if inv is None:
    st.markdown('<div class="sec-head">Getting Started</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    cards = [
        ("ti-file-upload", "#e6f1fb", "#0c447c",
         "Upload File .INV",
         "File output RES2DINV langsung di-parse otomatis. Semua format v3.x–10.x didukung."),
        ("ti-settings",    "#f0faf5", "#085041",
         "Atur Parameter",
         "Pilih variasi spasi (1–5 m) dan konfigurasi elektroda. Bisa pilih lebih dari satu sekaligus."),
        ("ti-chart-bar",   "#faeeda", "#633806",
         "Analisis Hasil",
         "Pseudosection interaktif, komparasi visual, tabel statistik, dan export laporan HTML."),
    ]
    for col, (icon, bg, txt, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(f"""
            <div style='background:{bg};border:1px solid {txt}33;border-radius:5px;
              padding:18px 16px;text-align:left;min-height:150px;'>
              <div style='font-size:11px;font-weight:600;text-transform:uppercase;
                letter-spacing:.08em;color:{txt};margin-bottom:6px'>{title}</div>
              <p style='font-size:11.5px;color:#374151;margin:0;line-height:1.6'>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="info-box" style="margin-top:12px">Belum punya file .INV? Klik tombol <b>Demo</b> di sidebar untuk mencoba aplikasi dengan data sintetik lintasan 30 m.</div>',
                unsafe_allow_html=True)

    with st.expander("Panduan Penggunaan Aplikasi", expanded=True):
        steps = [
            ("Upload / Demo",
             "Klik <b>Demo</b> di sidebar untuk data contoh, atau upload file <code>.INV</code> hasil RES2DINV.",
             "File .INV → sidebar → Upload"),
            ("Pilih Parameter",
             "Centang variasi <b>Spasi elektroda</b> dan <b>Konfigurasi</b> yang ingin dibandingkan.",
             "Spasi: [1, 2, 3] m  |  Konfigurasi: [WS, Wenner, DD]"),
            ("Jalankan Simulasi",
             "Klik <b>▶ Run Forward Modelling</b>. Semua kombinasi dijalankan otomatis.",
             "Total simulasi = jumlah spasi × jumlah konfigurasi"),
            ("Analisis Hasil",
             "Gunakan tab di bawah untuk melihat penampang, membandingkan variasi, dan export laporan.",
             "Tab: Data Inversi → Model → Forward → Komparasi Spasi → Komparasi Konfigurasi → Laporan"),
        ]
        for i, (title, desc, code) in enumerate(steps, 1):
            st.markdown(f"""
            <div class="guide-step">
              <div class="guide-num">{i}</div>
              <div class="guide-body">
                <div class="guide-title">{title}</div>
                <div class="guide-desc">{desc}</div>
                <code class="guide-code">{code}</code>
              </div>
            </div>""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# TAB NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Data Inversi",
    "Model True ρ",
    "Forward Modelling",
    "Komparasi Spasi",
    "Komparasi Konfigurasi",
    "Laporan & Export",
])

fwd = st.session_state.forward_results


# ═══════════════════════════════════════════════════
# TAB 1 — DATA INVERSI
# ═══════════════════════════════════════════════════
with tab1:
    if inv.get("is_demo"):
        st.markdown('<div class="warn-box">Mode demo aktif — data sintetik lintasan 30 m, Wenner-Schlumberger.</div>',
                    unsafe_allow_html=True)

    st.markdown('<div class="sec-head">Ringkasan File Inversi</div>', unsafe_allow_html=True)

    rms    = inv.get("final_rms", 0)
    rms_ok = rms and rms <= 5
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("File",       inv.get("filename", "-")[:20])
    c2.metric("Iterasi",    inv.get("iteration_used", "-"))
    c3.metric("RMS Error",  f"{rms:.1f}%" if rms else "-",
              delta="Konvergen ✓" if rms_ok else "Perlu evaluasi",
              delta_color="normal" if rms_ok else "inverse")
    c4.metric("Grid blok",  f"{inv.get('n_cols',0)} × {inv.get('n_rows',0)}")
    c5.metric("Spasi a",    f"{inv.get('electrode_spacing','-')} m")

    if inv.get("rms_history"):
        st.markdown('<div class="sec-head">Konvergensi Iterasi Inversi</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_rms_convergence(inv["rms_history"]), use_container_width=True)
        if rms_ok:
            st.markdown(f'<div class="ok-box">✓ RMS Error akhir <b>{rms:.1f}%</b> di bawah batas 5%. Model inversi valid dan siap digunakan untuk forward modelling.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warn-box">⚠ RMS Error {rms:.1f}% masih di atas 5%. Pertimbangkan untuk menambah iterasi di RES2DINV atau memeriksa kualitas data lapangan.</div>',
                        unsafe_allow_html=True)

    rho_mat = inv.get("rho_matrix")
    depths  = inv.get("depths", [])
    if rho_mat is not None and len(depths) > 0:
        st.markdown('<div class="sec-head">Nilai ρ Per Lapisan</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Rata-rata resistivitas dari file .INV — digunakan sebagai model acuan forward modelling</div>',
                    unsafe_allow_html=True)
        la = compute_layer_averages(rho_mat, depths)
        col_l, col_r = st.columns([1, 2])
        with col_l:
            for l in la:
                mat, color = classify_material(l["avg_rho"])
                st.markdown(f"""
                <div class="layer-row">
                  <div class="layer-dot" style="background:{color}"></div>
                  <div class="layer-info">
                    <div class="layer-name">Lapisan {l['layer']} &nbsp;(z = {l['depth']:.1f} m)</div>
                    <div class="layer-val">{mat}</div>
                  </div>
                  <div class="layer-rho">{l['avg_rho']:.0f} Ω·m</div>
                </div>""", unsafe_allow_html=True)
        with col_r:
            st.plotly_chart(plot_layer_bar(la), use_container_width=True)

        with st.expander("Tabel nilai ρ per blok lengkap"):
            x_pos = inv.get("x_positions", list(range(inv.get("n_cols", 0))))
            rows  = []
            for ri, d in enumerate(depths):
                if ri < rho_mat.shape[0]:
                    row = {"Kedalaman (m)": d}
                    for ci, x in enumerate(x_pos):
                        if ci < rho_mat.shape[1]:
                            row[f"x={x:.1f}m"] = round(float(rho_mat[ri, ci]), 1)
                    valid = rho_mat[ri][rho_mat[ri] > 0]
                    row["Rata-rata (Ω·m)"] = round(float(np.mean(valid)), 1) if len(valid) > 0 else 0
                    rows.append(row)
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, height=220)
            st.download_button("Download CSV",
                               data=df.to_csv(index=False),
                               file_name="rho_blok.csv", mime="text/csv")


# ═══════════════════════════════════════════════════
# TAB 2 — MODEL TRUE ρ
# ═══════════════════════════════════════════════════
with tab2:
    rho_mat = inv.get("rho_matrix")
    x_pos   = inv.get("x_positions", [])
    depths  = inv.get("depths", [])

    if rho_mat is None or rho_mat.size == 0:
        st.warning("Model resistivitas tidak tersedia.")
    else:
        st.markdown('<div class="sec-head">Penampang Resistivitas Sejati (True ρ)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">Model 2D hasil inversi RES2DINV — ini adalah model bumi yang akan digunakan sebagai input forward modelling</div>',
                    unsafe_allow_html=True)
        vlo, vhi = rho_min_inp, rho_max_inp
        st.plotly_chart(
            plot_true_section(rho_mat, x_pos, depths, rho_min=vlo, rho_max=vhi, show_values=show_vals),
            use_container_width=True)

        la = compute_layer_averages(rho_mat, depths)
        n_cols = 4
        cols   = st.columns(n_cols)
        for i, l in enumerate(la):
            mat, color = classify_material(l["avg_rho"])
            with cols[i % n_cols]:
                st.markdown(f"""
                <div class="layer-row">
                  <div class="layer-dot" style="background:{color}"></div>
                  <div class="layer-info">
                    <div class="layer-name">Lap. {l['layer']} · {l['depth']:.1f} m</div>
                    <div class="layer-val">{mat}</div>
                  </div>
                  <div class="layer-rho">{l['avg_rho']:.0f}</div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# TAB 3 — FORWARD MODELLING
# ═══════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-head">Hasil Forward Modelling</div>', unsafe_allow_html=True)

    if not fwd:
        st.markdown('<div class="warn-box">Belum ada hasil simulasi. Jalankan forward modelling dari sidebar.</div>',
                    unsafe_allow_html=True)
    else:
        vlo, vhi = rho_min_inp, rho_max_inp
        keys = sorted(fwd.keys())
        sel  = st.selectbox("Pilih simulasi", keys,
                            format_func=lambda k: f"{k.split('_',1)[1]}  —  a = {k.split('_',1)[0]} m")
        res  = fwd[sel]
        dp   = res.datum_points

        # KPI row
        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi-card">
            <div class="kpi-val">{res.array_name.split('-')[0]}</div>
            <div class="kpi-label">Konfigurasi</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-val">{res.electrode_spacing} m</div>
            <div class="kpi-label">Spasi a</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-val">{len(dp)}</div>
            <div class="kpi-label">Total datum</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-val">{dp[:,2].min():.0f}</div>
            <div class="kpi-label">ρa min (Ω·m)</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-val">{dp[:,2].max():.0f}</div>
            <div class="kpi-label">ρa maks (Ω·m)</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-val">{dp[:,2].mean():.0f}</div>
            <div class="kpi-label">ρa rata-rata (Ω·m)</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sec-head">Pseudosection Apparent Resistivity</div>', unsafe_allow_html=True)
        st.plotly_chart(
            plot_pseudosection(res, rho_min=vlo, rho_max=vhi, show_values=show_vals),
            use_container_width=True)

        # Download .dat
        dat_lines = [
            f"{sel}",
            f"{res.electrode_spacing:.1f}",
            f"{7 if 'Schlumberger' in res.array_name else (1 if 'Wenner' in res.array_name else 3)}",
            "1",
            f"{len(dp)}",
        ]
        for row in dp:
            dat_lines.append(f"  {row[0]:.2f}   {row[1]:.3f}   {row[2]:.2f}")
        dat_str = "\n".join(dat_lines)

        col_dl1, col_dl2, _ = st.columns([1, 1, 2])
        with col_dl1:
            st.download_button(
                "Download .dat (RES2DINV)",
                data=dat_str,
                file_name=f"forward_{sel}.dat",
                mime="text/plain",
                use_container_width=True,
            )
        with col_dl2:
            df_dp = pd.DataFrame(dp, columns=["X_mid_m", "Depth_m", "Rho_a_Ohmm"])
            st.download_button(
                "Download CSV",
                data=df_dp.to_csv(index=False),
                file_name=f"forward_{sel}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════
# TAB 4 — KOMPARASI SPASI
# ═══════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-head">Komparasi Variasi Spasi Elektroda</div>', unsafe_allow_html=True)

    if not fwd:
        st.markdown('<div class="warn-box">Belum ada hasil simulasi.</div>', unsafe_allow_html=True)
    else:
        arr_sel = st.selectbox(
            "Konfigurasi yang dibandingkan",
            options=list({r.array_name for r in fwd.values()}),
        )
        spasi_res = {k: v for k, v in fwd.items() if v.array_name == arr_sel}

        if len(spasi_res) < 2:
            st.markdown('<div class="info-box">Pilih minimal 2 variasi spasi untuk perbandingan.</div>', unsafe_allow_html=True)
        else:
            vlo, vhi = rho_min_inp, rho_max_inp
            st.plotly_chart(
                plot_comparison_spasi(spasi_res, rho_min=vlo, rho_max=vhi),
                use_container_width=True)

            st.markdown('<div class="sec-head">Statistik Per Spasi</div>', unsafe_allow_html=True)
            rows = []
            for k, r in sorted(spasi_res.items()):
                dp = r.datum_points
                rows.append({
                    "Spasi a (m)": r.electrode_spacing,
                    "Total datum": len(dp),
                    "ρa min (Ω·m)": round(dp[:, 2].min(), 1),
                    "ρa maks (Ω·m)": round(dp[:, 2].max(), 1),
                    "ρa rata-rata (Ω·m)": round(dp[:, 2].mean(), 1),
                    "Kedalaman maks (m)": round(dp[:, 1].max(), 2),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════
# TAB 5 — KOMPARASI KONFIGURASI
# ═══════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-head">Komparasi Konfigurasi Elektroda</div>', unsafe_allow_html=True)

    if not fwd:
        st.markdown('<div class="warn-box">Belum ada hasil simulasi.</div>', unsafe_allow_html=True)
    else:
        spasi_sel = st.selectbox(
            "Spasi yang dibandingkan",
            options=sorted({v.electrode_spacing for v in fwd.values()}),
            format_func=lambda x: f"a = {x} m",
        )
        arr_res = {v.array_name: v for k, v in fwd.items()
                   if v.electrode_spacing == spasi_sel}

        if len(arr_res) < 2:
            st.markdown('<div class="info-box">Pilih minimal 2 konfigurasi untuk perbandingan.</div>', unsafe_allow_html=True)
        else:
            vlo, vhi = rho_min_inp, rho_max_inp
            st.plotly_chart(
                plot_comparison_array(arr_res, rho_min=vlo, rho_max=vhi),
                use_container_width=True)

            st.markdown('<div class="sec-head">Statistik Perbandingan Konfigurasi</div>', unsafe_allow_html=True)
            cfg_info = {
                "Wenner-Schlumberger": ("Vertikal + lateral", "Sedang",  "Serbaguna"),
                "Wenner":              ("Vertikal",           "Tinggi",  "Lapisan horizontal"),
                "Dipole-Dipole":       ("Lateral",            "Rendah",  "Struktur vertikal / patahan"),
            }
            rows = []
            for arr, r in arr_res.items():
                dp   = r.datum_points
                info = cfg_info.get(arr, ("-", "-", "-"))
                rows.append({
                    "Konfigurasi":          arr,
                    "Datum":                len(dp),
                    "ρa min (Ω·m)":         round(dp[:, 2].min(), 1),
                    "ρa maks (Ω·m)":        round(dp[:, 2].max(), 1),
                    "ρa rata-rata (Ω·m)":   round(dp[:, 2].mean(), 1),
                    "Sensitivitas":          info[0],
                    "Rasio S/N":             info[1],
                    "Cocok untuk":           info[2],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown('<div class="sec-head">Rekomendasi Konfigurasi</div>', unsafe_allow_html=True)
            rc1, rc2, rc3 = st.columns(3)
            rec_data = [
                (rc1, "#eff6ff", "#b5d4f4", "#0c447c",
                 "Wenner-Schlumberger",
                 "Direkomendasikan untuk survei umum. Sensitif vertikal + lateral, cocok untuk mapping lapisan 2D serbaguna."),
                (rc2, "#e1f5ee", "#9fe1cb", "#085041",
                 "Wenner",
                 "Ideal untuk lapisan mendatar. Rasio S/N tertinggi — terbaik di area dengan noise lapangan tinggi."),
                (rc3, "#faeeda", "#fac775", "#633806",
                 "Dipole-Dipole",
                 "Optimal mendeteksi patahan dan kontak litologi vertikal. Resolusi lateral terbaik, sinyal lebih lemah."),
            ]
            for col, bg, border, txt, title, desc in rec_data:
                with col:
                    st.markdown(f"""
                    <div style='background:{bg};border:1px solid {border};border-radius:5px;
                      padding:14px;min-height:110px'>
                      <div style='font-size:11px;font-weight:600;text-transform:uppercase;
                        letter-spacing:.06em;color:{txt};margin-bottom:6px'>{title}</div>
                      <span style='font-size:11.5px;color:#374151;line-height:1.55'>{desc}</span>
                    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# TAB 6 — LAPORAN & EXPORT
# ═══════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="sec-head">Export & Laporan Penelitian</div>', unsafe_allow_html=True)

    if fwd:
        st.markdown('<div class="sec-sub">Semua simulasi yang telah dijalankan</div>', unsafe_allow_html=True)
        sum_rows = []
        for key, res in sorted(fwd.items()):
            dp = res.datum_points
            sum_rows.append({
                "Simulasi":              key,
                "Konfigurasi":           res.array_name,
                "Spasi a (m)":           res.electrode_spacing,
                "N-level maks":          res.n_max,
                "Total datum":           len(dp),
                "ρa min (Ω·m)":          round(dp[:, 2].min(), 1),
                "ρa maks (Ω·m)":         round(dp[:, 2].max(), 1),
                "ρa rata-rata (Ω·m)":    round(dp[:, 2].mean(), 1),
                "Depth maks (m)":        round(dp[:, 1].max(), 2),
            })
        df_sum = pd.DataFrame(sum_rows)
        st.dataframe(df_sum, use_container_width=True, hide_index=True)

        st.markdown('<div class="sec-head">Download File</div>', unsafe_allow_html=True)
        co1, co2, co3 = st.columns(3)

        with co1:
            all_rows = []
            for key, res in sorted(fwd.items()):
                dp = res.datum_points
                for row in dp:
                    all_rows.append({
                        "Simulasi":      key,
                        "Konfigurasi":   res.array_name,
                        "Spasi_a_m":     res.electrode_spacing,
                        "X_mid_m":       row[0],
                        "Depth_m":       row[1],
                        "Rho_a_Ohmm":    round(row[2], 2),
                    })
            df_all = pd.DataFrame(all_rows)
            st.download_button(
                "Download semua datum (CSV)",
                data=df_all.to_csv(index=False),
                file_name="geoforward_semua_datum.csv",
                mime="text/csv",
                use_container_width=True,
                help="Semua titik datum dari semua simulasi dalam satu file CSV",
            )

        with co2:
            st.download_button(
                "Download ringkasan statistik (CSV)",
                data=df_sum.to_csv(index=False),
                file_name="geoforward_ringkasan.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with co3:
            html_report = build_html_report(inv, fwd)
            st.download_button(
                "Download Laporan HTML",
                data=html_report,
                file_name="laporan_geoforward.html",
                mime="text/html",
                use_container_width=True,
                help="Laporan lengkap siap cetak / presentasi",
            )

        st.markdown("""
        <div class="info-box">
          <b>Cara menggunakan file export:</b><br>
          File <b>.csv</b> dapat dibuka di Excel/Spreadsheet untuk analisis lanjutan.<br>
          File <b>.dat</b> dapat di-import langsung ke RES2DINV untuk inverse modelling.<br>
          File <b>laporan HTML</b> dapat dibuka di browser dan dicetak (Ctrl+P) atau disimpan sebagai PDF.
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown('<div class="warn-box">Jalankan forward modelling terlebih dahulu untuk mengaktifkan fitur export.</div>',
                    unsafe_allow_html=True)

    with st.expander("Format file .dat untuk RES2DINV"):
        st.markdown("""
        File `.dat` yang dihasilkan memiliki format standar RES2DINV:
        ```
        Nama_Simulasi           ← baris 1: nama lintasan
        2.0                     ← baris 2: spasi elektroda (m)
        7                       ← baris 3: kode konfigurasi (1=W, 3=DD, 7=WS)
        1                       ← baris 4: flag
        84                      ← baris 5: jumlah datum
          2.00   0.519   128.40  ← datum: x_mid  pseudo_depth  rho_a
          4.00   0.519   143.20
          ...
        ```
        File ini dapat langsung dibuka di RES2DINV untuk proses inverse modelling lanjutan.
        """)

    st.markdown('<div class="sec-head">Referensi Ilmiah</div>', unsafe_allow_html=True)
    refs = [
        ("Loke, M.H. (2004)",
         "*2D and 3D electrical imaging surveys*. Geotomo Software Manual."),
        ("Loke, M.H. (2004)",
         "*RES2DMOD ver. 3.01 — Rapid 2D Resistivity Forward Modelling*. Geotomo Software."),
        ("Telford, Geldart & Sheriff (1990)",
         "*Applied Geophysics* (2nd ed.). Cambridge University Press."),
        ("Reynolds, J.M. (1997)",
         "*An Introduction to Applied and Environmental Geophysics*. Wiley."),
    ]
    for author, title in refs:
        st.markdown(f"- **{author}** — {title}")


# ══════════════════════════════════════════════════════════════════════════════
# STATUS BAR & FOOTER
# ══════════════════════════════════════════════════════════════════════════════
n_sim = len(fwd)
inv_fname = inv.get("filename", "–") if inv else "–"
inv_rms   = f"{inv.get('final_rms',0):.1f}%" if inv else "–"

st.markdown(f"""
<div class="geo-statusbar">
  <span>Ready</span>
  <div class="geo-sb-sep"></div>
  <span>{inv_fname}</span>
  <div class="geo-sb-sep"></div>
  <span>RMS: {inv_rms}</span>
  <div class="geo-sb-sep"></div>
  <span>{n_sim} forward run</span>
  <div class="geo-sb-sep"></div>
  <span style="margin-left:auto">GeoForward 2D v2.4 &nbsp;·&nbsp; Python · Streamlit · Plotly</span>
</div>
""", unsafe_allow_html=True)
