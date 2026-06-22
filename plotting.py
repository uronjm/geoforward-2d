"""
Modul visualisasi penampang geolistrik 2D menggunakan Plotly.
Dioptimalkan dengan gaya visual standard output RES2DINV.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from forward_model import ForwardResult, classify_material


# ─── Skala warna geolistrik (RES2DINV Standard Standard) ─────────────────────

COLORSCALE_GEO = [
    [0.00, "#185FA5"],   # Biru tua — sangat rendah
    [0.10, "#1976D2"],   # Biru
    [0.25, "#29B6F6"],   # Biru muda
    [0.40, "#1D9E75"],   # Hijau toska
    [0.55, "#66BB6A"],   # Hijau
    [0.65, "#FDD835"],   # Kuning
    [0.75, "#EF9F27"],   # Oranye
    [0.85, "#D85A30"],   # Oranye merah
    [0.92, "#C62828"],   # Merah
    [1.00, "#4A1B0C"],   # Merah tua
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
    4: "#D85A30",
    5: "#5D4037",
}


def plot_true_section(inv_data: dict) -> go.Figure:
    """
    Plot Model True Resistivity (Hasil Inversi) dengan gaya khas RES2DINV:
    - Kontur memblok/diskret (tanpa gradasi halus)
    - Batas kontur tegas dengan garis hitam tipis
    - Sumbu X (Distance) berada di posisi atas (Top)
    - Sumbu Y (Depth) terbalik ke bawah
    - Colorbar diletakkan secara horizontal di bagian bawah grafik
    """
    if inv_data is None or inv_data.get("rho_matrix") is None:
        return go.Figure()

    rho_matrix = inv_data["rho_matrix"]
    x_positions = np.array(inv_data["x_positions"])
    depths = np.array(inv_data["depths"])

    # Nilai minimum dan maksimum untuk penentuan range logaritmik
    min_rho = np.min(rho_matrix[rho_matrix > 0]) if np.any(rho_matrix > 0) else 1.0
    max_rho = np.max(rho_matrix) if np.any(rho_matrix > 0) else 1000.0

    # Membuat pembagian level kontur berbasis logaritmik agar diskretnya seimbang
    log_min = np.log10(min_rho)
    log_max = np.log10(max_rho)
    # Membuat 12 level batas nilai (menghasilkan 11 blok warna)
    log_levels = np.linspace(log_min, log_max, 12)
    levels = 10 ** log_levels

    # Pembulatan nilai tick pembacaan colorbar agar terlihat rapi
    tick_vals = [float(f"{x:.1f}") if x < 10 else int(round(x)) for x in levels]

    fig = go.Figure(data=go.Contour(
        z=rho_matrix,
        x=x_positions,
        y=depths,
        connectgaps=True,
        hoverongaps=False,
        hovertemplate=(
            "<b>Distance (X):</b> %{x:.2f} m<br>"
            "<b>Depth (Z):</b> %{y:.2f} m<br>"
            "<b>True ρ:</b> %{z:.2f} Ω·m<extra></extra>"
        ),
        colorscale=COLORSCALE_GEO,
        
        # Pengaturan kunci untuk gaya blok RES2DINV
        contours_coloring='heatmap',
        contours=dict(
            start=levels[0],
            end=levels[-1],
            size=(levels[-1] - levels[0]) / 12,
            showlines=True,        # Menampilkan garis batas blok
        ),
        line_width=0.4,            # Garis batas blok tipis
        line_color="#424242",      # Warna abu-abu gelap/hitam untuk batas blok
        
        # Modifikasi Colorbar menjadi Horizontal di bawah penampang
        colorbar=dict(
            title="Resistivity (Ω·m)",
            title_side="bottom",
            orientation="h",       # Membuat posisi horizontal
            y=-0.28,               # Menaruh di bawah sumbu horizontal
            x=0.5,
            xanchor="center",
            thickness=16,          # Ketebalan bar warna
            len=0.85,              # Panjang bar warna dibandingkan lebar grafik
            tickvals=tick_vals,    # Angka pembatas warna diskret
            tickmode="array",
            tickfont=dict(size=10)
        )
    ))

    # Konfigurasi Layout Sumbu Utama khas RES2DINV
    fig.update_layout(
        xaxis=dict(
            title="Distance (m)",
            side="top",            # Label jarak di taruh di bagian atas penampang
            mirror=True,
            showgrid=True,
            gridcolor="#E0E0E0",
            zeroline=False
        ),
        yaxis=dict(
            title="Depth (m)",
            autorange="reverse",   # Membalik nilai kedalaman ke arah bawah
            mirror=True,
            showgrid=True,
            gridcolor="#E0E0E0",
            zeroline=False
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=40, t=70, b=110), # Margin bawah diperlebar demi colorbar horizontal
        height=420,
        title=dict(
            text=f"Inverse Model True Resistivity Section — Iterasi {inv_data.get('iteration_used', '-')}",
            font=dict(size=13, color="#263238"),
            x=0.01,
            y=0.98
        )
    )
    return fig


def plot_pseudosection(result: ForwardResult) -> go.Figure:
    """Plot Apparent Resistivity Pseudosection dari hasil forward modelling."""
    if result is None or len(result.datum_points) == 0:
        return go.Figure()

    dp = result.datum_points
    x = dp[:, 0]
    z = dp[:, 1]
    rho_a = dp[:, 2]

    # Menggunakan Scatter untuk titik datum asli dan Contour untuk penampang semu
    fig = go.Figure()

    # Kontur pseudosection halus untuk data semu lapangan
    fig.add_trace(go.Contour(
        z=rho_a,
        x=x,
        y=z,
        connectgaps=True,
        colorscale=COLORSCALE_GEO,
        contours_coloring='best',
        line_width=0,
        colorbar=dict(
            title="Apparent ρ (Ω·m)",
            thickness=15,
            len=0.7,
            yanchor="middle",
            y=0.5
        ),
        hovertemplate="X: %{x:.1f} m<br>Pseudo Depth: %{y:.2f} m<br>ρa: %{z:.1f} Ω·m<extra></extra>"
    ))

    # Tambahkan titik hitam kecil penanda datum point asli (titik tengah elektrodaya)
    fig.add_trace(go.Scatter(
        x=x,
        y=z,
        mode="markers",
        marker=dict(size=4, color="black", opacity=0.6),
        name="Datum Point",
        hovertemplate="X: %{x:.1f} m<br>Pseudo Depth: %{y:.2f} m<br>ρa: %{customdata:.1f} Ω·m<extra></extra>",
        customdata=rho_a,
        showlegend=False
    ))

    fig.update_layout(
        xaxis=dict(title="Distance (m)", side="top", mirror=True, showgrid=True, gridcolor="#ECEFF1"),
        yaxis=dict(title="Pseudo-Depth (m)", autorange="reverse", mirror=True, showgrid=True, gridcolor="#ECEFF1"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=20, t=60, b=40),
        height=380,
        title=dict(
            text=f"Pseudosection Hasil Forward Modelling ({result.array_name}, Spasi: {result.electrode_spacing}m)",
            font=dict(size=12), x=0.02
        )
    )
    return fig


def plot_rms_convergence(rms_history: list) -> go.Figure:
    """Plot grafik konvergensi penurunan error RMS inversi."""
    if not rms_history:
        return go.Figure()

    iterations = list(range(1, len(rms_history) + 1))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iterations,
        y=rms_history,
        mode="lines+markers",
        line=dict(color="#C62828", width=2.5),
        marker=dict(size=8, color="#0A2744"),
        hovertemplate="Iterasi %{x}<br>RMS Error: %{y:.2f}%<extra></extra>"
    ))

    fig.update_layout(
        xaxis=dict(title="Nomor Iterasi", tickmode="linear", dtick=1),
        yaxis=dict(title="RMS Error (%)", side="left"),
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="white",
        margin=dict(l=50, r=20, t=40, b=40),
        height=280,
        title=dict(text="Kurva Konvergensi Inversi RES2DINV", font=dict(size=12), x=0.02)
    )
    return fig


def plot_comparison_spasi(results_dict: dict, current_array: str) -> go.Figure:
    """Subplot komparasi pseudosection untuk variasi spasi elektroda."""
    # Filter hasil berdasarkan konfigurasi yang dipilih
    filtered = {k: v for k, v in results_dict.items() if v.array_name == current_array}
    if not filtered:
        return go.Figure()

    spasis = sorted(list(filtered.keys()))
    n_plots = len(spasis)

    fig = make_subplots(
        rows=n_plots, cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Spasi Elektroda = {s} m" for s in spasis],
        vertical_spacing=0.12
    )

    for idx, s in enumerate(spasis):
        res = filtered[s]
        dp = res.datum_points
        
        # Tambah kontur penampang semu
        fig.add_trace(
            go.Contour(
                z=dp[:, 2], x=dp[:, 0], y=dp[:, 1],
                connectgaps=True, colorscale=COLORSCALE_GEO,
                showscale=False, hovertemplate="X: %{x:.1f} m<br>Z-sem: %{y:.2f} m<br>ρa: %{z:.1f} Ω·m<extra></extra>"
            ),
            row=idx+1, col=1
        )
        
        # Tambah titik datum penanda posisi tengah elektroda
        fig.add_trace(
            go.Scatter(
                x=dp[:, 0], y=dp[:, 1], mode="markers",
                marker=dict(size=3, color="black", opacity=0.5),
                showlegend=False, hoverinfo="skip"
            ),
            row=idx+1, col=1
        )
        
        # Atur agar sumbu Y terbalik di setiap subplot
        fig.update_yaxes(autorange="reverse", title="Depth (m)", row=idx+1, col=1, gridcolor="#ECEFF1")
        fig.update_xaxes(gridcolor="#ECEFF1", row=idx+1, col=1)

    fig.update_layout(
        margin=dict(l=50, r=20, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        title=dict(
            text=f"Komparasi Rentang Deteksi Geolistrik — Variasi Spasi ({current_array})",
            font=dict(size=13), x=0.02
        ),
        height=240 * n_plots + 60
    )
    return fig


def plot_comparison_array(results_list: list, spasi_ref: float) -> go.Figure:
    """Subplot komparasi bentuk pseudosection lintas variasi konfigurasi elektroda."""
    if not results_list:
        return go.Figure()

    n_plots = len(results_list)
    fig = make_subplots(
        rows=n_plots, cols=1,
        shared_xaxes=True,
        subplot_titles=[f"Konfigurasi: {r.array_name}" for r in results_list],
        vertical_spacing=0.12
    )

    for idx, res in enumerate(results_list):
        dp = res.datum_points
        
        fig.add_trace(
            go.Contour(
                z=dp[:, 2], x=dp[:, 0], y=dp[:, 1],
                connectgaps=True, colorscale=COLORSCALE_GEO,
                showscale=False, hovertemplate="X: %{x:.1f} m<br>Z-sem: %{y:.2f} m<br>ρa: %{z:.1f} Ω·m<extra></extra>"
            ),
            row=idx+1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dp[:, 0], y=dp[:, 1], mode="markers",
                marker=dict(size=3, color="black", opacity=0.5),
                showlegend=False, hoverinfo="skip"
            ),
            row=idx+1, col=1
        )
        
        fig.update_yaxes(autorange="reverse", title="Pseudo-Depth (m)", row=idx+1, col=1, gridcolor="#ECEFF1")
        fig.update_xaxes(gridcolor="#ECEFF1", row=idx+1, col=1)

    fig.update_layout(
        margin=dict(l=50, r=20, t=60, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        title=dict(
            text=f"Komparasi Pseudosection — Variasi Konfigurasi Elektroda (Spasi Ref: {spasi_ref}m)",
            font=dict(size=13), x=0.02
        ),
        height=240 * n_plots + 60
    )
    return fig


def plot_layer_bar(layer_avgs: list) -> go.Figure:
    """Bar chart representasi nilai rata-rata resistivitas per lapisan batuan."""
    if not layer_avgs:
        return go.Figure()

    labels = [f"Lap. {d['layer']}<br>(z={d['depth']:.1f}m)" for d in layer_avgs]
    values = [d["avg_rho"] for d in layer_avgs]
    materials = [classify_material(v)[0] for v in values]
    bar_colors = [classify_material(v)[1] for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="white", width=0.5)),
        text=[f"{v:.1f} Ω·m" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>ρ rata-rata = %{x:.1f} Ω·m<br>Prediksi: %{customdata}<extra></extra>",
        customdata=materials,
    ))

    fig.update_layout(
        title=dict(text="Rata-rata Resistivitas True Model per Lapisan", font=dict(size=12), x=0.02),
        xaxis=dict(title="Resistivitas (Ω·m)", showgrid=True, gridcolor="#F0F4F8"),
        yaxis=dict(autorange="reverse"), # Lapisan 1 berada di atas grafik batang
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=50, t=40, b=40),
        height=max(200, len(layer_avgs) * 35)
    )
    return fig
