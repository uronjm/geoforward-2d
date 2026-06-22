"""
Modul visualisasi penampang geolistrik 2D menggunakan Plotly.
Menyediakan visualisasi interaktif yang menyerupai output standar RES2DINV.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from forward_model import ForwardResult, classify_material

# ─── Skala Warna Geolistrik Standar & RES2DINV ───────────────────────────────

COLORSCALE_GEO = [
    [0.00, "#185FA5"],   # Biru tua — sangat rendah
    [0.10, \"#1976D2\"],   # Biru
    [0.25, \"#29B6F6\"],   # Biru muda
    [0.40, \"#1D9E75\"],   # Hijau toska
    [0.55, \"#66BB6A\"],   # Hijau
    [0.65, \"#FDD835\"],   # Kuning
    [0.75, \"#EF9F27\"],   # Oranye
    [0.85, \"#D85A30\"],   # Oranye merah
    [0.92, \"#C62828\"],   # Merah
    [1.00, \"#4A1B0C\"],   # Merah tua
]

# Skala warna khusus penampang Inversi (True Resistivity) mirip RES2DINV
RES2DINV_COLORSCALE = [
    [0.00, "#1A237E"],   # Biru Sangat Tua (Low Resistivity / Lempung / Air)
    [0.15, "#1976D2"],   # Biru
    [0.30, "#4FC3F7"],   # Biru Muda
    [0.45, "#4CAF50"],   # Hijau (Silt / Pasir Pasiran)
    [0.60, "#FFEB3B"],   # Kuning
    [0.75, "#FB8C00"],   # Oranye
    [0.90, "#D32F2F"],   # Merah (Batuan Keras / Sandstone Kering)
    [1.00, "#4A148C"]    # Ungu Tua / Cokelat (Massive Bedrock)
]

ARRAY_COLORS = {
    "Wenner-Schlumberger": "#185FA5",
    "Wenner": "#0F6E56",
    "Dipole-Dipole": "#993C1D",
}

SPASI_COLORS = {
    1: "#7F77DD", 
    2: "#185FA5", 
    3: "#0F6E56", 
    4: "#EF9F27", 
    5: "#D85A30"
}


# ─── 1. Penampang True Resistivity (Optimasi Gaya RES2DINV) ──────────────────

def plot_true_section(inv_data: dict) -> go.Figure:
    """
    Memplot penampang True Resistivity 2D yang menyerupai output standar RES2DINV
    menggunakan kontur halus, grid interpolasi, dan pemotongan batas kontur (trapesium).
    """
    if inv_data.get("rho_matrix") is None or len(inv_data.get("blocks", [])) == 0:
        return go.Figure()

    # Ekstrak data blok hasil inversi
    blocks = inv_data["blocks"]
    x_coords = np.array([b["x_pos"] for b in blocks])
    z_coords = np.array([b["depth"] for b in blocks])
    rho_values = np.array([b["resistivity"] for b in blocks])

    # Menggunakan skala logaritmik untuk nilai resistivitas agar kontur lebih sensitif
    log_rho = np.log10(rho_values)

    # Buat grid halus (Fine Grid) untuk interpolasi kontur yang mulus
    x_min, x_max = x_coords.min(), x_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()
    
    grid_x, grid_z = np.mgrid[x_min:x_max:300j, z_min:z_max:150j]

    # Interpolasi data titik blok ke grid halus menggunakan Cubic Spline
    grid_rho_log = griddata(
        (x_coords, z_coords), 
        log_rho, 
        (grid_x, grid_z), 
        method='cubic'
    )
    
    # Kembalikan nilai ke skala resistivitas asli setelah interpolasi
    grid_rho = 10**grid_rho_log

    # Masking / Pemotongan Trapesium (Khas RES2DINV)
    # Seiring bertambahnya kedalaman (z), jangkauan horizontal akan menyempit
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            x_val = grid_x[i, j]
            z_val = grid_z[i, j]
            
            # Batas toleransi pemotongan kiri dan kanan berdasarkan kedalaman
            left_bound = x_min + (z_val - z_min) * 0.8
            right_bound = x_max - (z_val - z_min) * 0.8
            
            if x_val < left_bound or x_val > right_bound:
                grid_rho[i, j] = np.nan

    # Tentukan level kontur secara logaritmik agar pembagian kelas warna seimbang
    c_min = np.nanmin(rho_values)
    c_max = np.nanmax(rho_values)

    fig = go.Figure(data=go.Contour(
        x=grid_x[:, 0],
        y=grid_z[0, :],
        z=grid_rho.T,
        colorscale=RES2DINV_COLORSCALE,
        contours=dict(
            start=c_min,
            end=c_max,
            showlines=True,       # Memunculkan garis kontur hitam tipis khas RES2DINV
            coloring='heatmap'
        ),
        line=dict(width=0.5, color="rgba(40,40,40,0.5)"),
        colorbar=dict(
            title="Resistivitas (Ω·m)",
            titleside="right",
            ticks="outside",
            tickformat=".1f",
            exponentformat="none"
        ),
        hovertemplate="X: %{x:.1f} m<br>Kedalaman: %{y:.1f} m<br>True ρ: %{z:.1f} Ω·m<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=f"Penampang Inversi True Resistivity 2D (Model RMS: {inv_data.get('final_rms', 0):.2f}%)",
            font=dict(size=14, color="#1e293b"),
            x=0.01
        ),
        xaxis=dict(
            title="Jarak Lintasan (m)",
            gridcolor="rgba(200,200,200,0.2)",
            zeroline=False,
            mirror=True,
            showline=True,
            linecolor="#1e293b"
        ),
        yaxis=dict(
            title="Kedalaman (m)",
            autorange="reverse", # Membalik sumbu Y (0 di atas, kedalaman makin ke bawah)
            gridcolor="rgba(200,200,200,0.2)",
            zeroline=False,
            mirror=True,
            showline=True,
            linecolor="#1e293b"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=40, t=50, b=50),
        height=450
    )

    return fig


# ─── 2. Penampang Semu (Pseudosection Forward Modelling) ───────────────────

def plot_pseudosection(fwd_result: ForwardResult) -> go.Figure:
    """Memplot pseudosection hasil dari forward modelling."""
    pts = fwd_result.datum_points
    if pts.size == 0:
        return go.Figure()

    x = pts[:, 0]
    z = pts[:, 1]
    rho_a = pts[:, 2]

    fig = go.Figure()

    # Plot titik datum individu
    fig.add_trace(go.Scatter(
        x=x, y=z,
        mode="markers",
        marker=dict(
            size=10,
            color=rho_a,
            colorscale=COLORSCALE_GEO,
            showscale=True,
            colorbar=dict(
                title="ρa (Ω·m)",
                titleside="right",
                tickformat=".1f"
            ),
            line=dict(width=0.5, color="black")
        ),
        hovertemplate="X (Tengah): %{x:.1f} m<br>Pseudo-Depth: %{y:.2f} m<br>ρa: %{marker.color:.1f} Ω·m<extra></extra>"
    ))

    # Tambahkan kontur halus di latar belakang jika data mencukupi
    if len(x) > 3:
        xi = np.linspace(x.min(), x.max(), 200)
        zi = np.linspace(z.min(), z.max(), 100)
        xi_m, zi_m = np.meshgrid(xi, zi)
        try:
            rho_i = griddata((x, z), np.log10(rho_a), (xi_m, zi_m), method='linear')
            rho_i = 10**rho_i
            
            fig.add_trace(go.Contour(
                x=xi, y=zi, z=rho_i,
                colorscale=COLORSCALE_GEO,
                showscale=False,
                opacity=0.85,
                contours=dict(coloring='heatmap', showlines=False),
                hoverinfo='skip'
            ))
        except:
            pass # Lewati jika kegagalan interpolasi akibat keterbatasan geometri data

    # Urutan trace: Kontur di bawah, titik marker di atas
    if len(fig.data) > 1:
        fig.data = (fig.data[1], fig.data[0])

    fig.update_layout(
        title=dict(
            text=f"Pseudosection Semu — {fwd_result.array_name} (a={fwd_result.electrode_spacing}m)",
            font=dict(size=14), x=0.02
        ),
        xaxis=dict(title="Jarak Lintasan (m)", mirror=True, showline=True, linecolor="#1e293b"),
        yaxis=dict(title="Pseudo Kedalaman (m)", autorange="reverse", mirror=True, showline=True, linecolor="#1e293b"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        margin=dict(l=60, r=40, t=50, b=50),
    )
    return fig


# ─── 3. Plot Konvergensi RMS Inversi ─────────────────────────────────────────

def plot_rms_convergence(rms_history: list) -> go.Figure:
    """Memplot grafik kurva konvergensi RMS terhadap nomor iterasi."""
    fig = go.Figure()
    if not rms_history:
        return fig

    iters = list(range(1, len(rms_history) + 1))
    
    fig.add_trace(go.Scatter(
        x=iters, y=rms_history,
        mode="lines+markers",
        line=dict(color="#185FA5", width=3),
        marker=dict(size=8, color="#0a2744"),
        name="RMS Error"
    ))

    fig.update_layout(
        title=dict(text="Kurva Konvergensi Inversi RES2DINV", font=dict(size=14)),
        xaxis=dict(title="Iterasi Ke-", tickmode="linear", start=1),
        yaxis=dict(title="RMS Error (%)", gridcolor="rgba(200,200,200,0.3)"),
        plot_bgcolor="white",
        height=320,
        margin=dict(l=50, r=30, t=40, b=40),
    )
    return fig


# ─── 4. Komparasi Variasi Spasi Elektroda ───────────────────────────────────

def plot_comparison_spasi(results_list: list) -> go.Figure:
    """Membandingkan kurva nilai ρa profil lateral pada kedalaman pseudo level-1 dari berbagai spasi."""
    fig = go.Figure()
    valid = False

    for res in results_list:
        pts = res.datum_points
        if pts.size == 0:
            continue
        
        # Ambil level pseudo kedalaman teratas (terdangkal)
        z_min = pts[:, 1].min()
        mask = np.abs(pts[:, 1] - z_min) < 1e-3
        layer1 = pts[mask]
        
        if layer1.size == 0:
            continue
        
        # Urutkan berdasarkan posisi X horizontal
        layer1 = layer1[layer1[:, 0].argsort()]
        color = SPASI_COLORS.get(int(res.electrode_spacing), "#666666")
        
        fig.add_trace(go.Scatter(
            x=layer1[:, 0], y=layer1[:, 2],
            mode="lines+markers",
            name=f"Spasi {res.electrode_spacing} m",
            line=dict(color=color, width=2),
            marker=dict(size=5)
        ))
        valid = True

    if not valid:
        return go.Figure()

    fig.update_layout(
        xaxis=dict(title="Jarak Lintasan (m)", mirror=True, showline=True, linecolor="#1e293b"),
        yaxis=dict(title="Apparent Resistivity ρa (Ω·m)", gridcolor="rgba(200,200,200,0.3)"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11),
        title=dict(
            text="Komparasi Sinyal ρa Terhadap Jarak Lintasan — Variasi Spasi Elektroda",
            font=dict(size=14), x=0.02
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=50, b=50),
        height=400
    )
    return fig


# ─── 5. Komparasi Variasi Konfigurasi Elektroda ──────────────────────────────

def plot_comparison_array(results_list: list) -> go.Figure:
    """Membandingkan respons sensitivitas lateral ρa antar konfigurasi (Wenner, WS, DD)."""
    fig = go.Figure()
    valid = False

    for res in results_list:
        pts = res.datum_points
        if pts.size == 0:
            continue
        
        # Ambil level kedalaman teratas
        z_min = pts[:, 1].min()
        mask = np.abs(pts[:, 1] - z_min) < 1e-3
        layer1 = pts[mask]
        
        if layer1.size == 0:
            continue
            
        layer1 = layer1[layer1[:, 0].argsort()]
        color = ARRAY_COLORS.get(res.array_name, "#666666")

        fig.add_trace(go.Scatter(
            x=layer1[:, 0], y=layer1[:, 2],
            mode="lines",
            name=res.array_name,
            line=dict(color=color, width=2.5)
        ))
        valid = True

    if not valid:
        return go.Figure()

    fig.update_layout(
        xaxis=dict(title="Jarak Lintasan (m)", mirror=True, showline=True, linecolor="#1e293b"),
        yaxis=dict(title="Apparent Resistivity ρa (Ω·m)", gridcolor="rgba(200,200,200,0.3)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=50, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11),
        title=dict(
            text="Komparasi Pseudosection — Variasi Konfigurasi Elektroda",
            font=dict(size=14), x=0.02
        ),
    )
    return fig


# ─── 6. Bar Chart Rata-rata Lapisan ──────────────────────────────────────────

def plot_layer_bar(layer_avgs: list) -> go.Figure:
    """Bar chart horizontal rata-rata resistivitas per kedalaman lapisan."""
    if not layer_avgs:
        return go.Figure()

    labels = [f"Lap. {d['layer']}\\n(z={d['depth']:.1f}m)" for d in layer_avgs]
    values = [d["avg_rho"] for d in layer_avgs]
    materials = [classify_material(v)[0] for v in values]
    bar_colors = [classify_material(v)[1] for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="white", width=0.5)),
        text=[f"{v:.0f} Ω·m" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>ρ rata-rata = %{x:.1f} Ω·m<extra></extra>",
        customdata=materials,
    ))

    fig.update_layout(
        title=dict(text="Rata-rata Resistivitas dan Estimasi Litologi per Lapisan Model Inversi", font=dict(size=13)),
        xaxis=dict(title="True Resistivity (Ω·m)", side="bottom"),
        yaxis=dict(autorange="reverse"),
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=100, r=50, t=50, b=40),
    )
    return fig
