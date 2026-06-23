"""
Modul visualisasi penampang geolistrik 2D menggunakan Plotly.
True-section dirender menyerupai output RES2DINV:
  - Bentuk trapezoid (menyempit ke bawah)
  - Kontur smooth (scipy interpolasi + filled contour)
  - Colorscale khas RES2DINV (biru-hijau-kuning-merah-ungu)
  - Titik datum di permukaan
  - Sumbu dan label sesuai konvensi RES2DINV
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    from scipy.interpolate import griddata
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

from forward_model import ForwardResult, classify_material


# ─── Skala warna ─────────────────────────────────────────────────────────────

# Colorscale khas RES2DINV: biru tua → cyan → hijau → kuning → oranye → merah → ungu/cokelat tua
COLORSCALE_RES2DINV = [
    [0.000, "#0000FF"],   # Biru tua
    [0.083, "#0040FF"],
    [0.167, "#0080FF"],
    [0.250, "#00BFFF"],   # Cyan
    [0.333, "#00FF80"],   # Hijau terang
    [0.417, "#40FF00"],
    [0.500, "#BFFF00"],   # Kuning hijau
    [0.583, "#FFFF00"],   # Kuning
    [0.667, "#FFC000"],   # Oranye
    [0.750, "#FF6000"],
    [0.833, "#FF0000"],   # Merah
    [0.917, "#C00060"],
    [1.000, "#800080"],   # Ungu/magenta
]

# Colorscale untuk pseudosection (tetap pakai yang lama)
COLORSCALE_GEO = [
    [0.00, "#185FA5"],
    [0.10, "#1976D2"],
    [0.25, "#29B6F6"],
    [0.40, "#1D9E75"],
    [0.55, "#66BB6A"],
    [0.65, "#FDD835"],
    [0.75, "#EF9F27"],
    [0.85, "#D85A30"],
    [0.92, "#C62828"],
    [1.00, "#4A1B0C"],
]

ARRAY_COLORS = {
    "Wenner-Schlumberger": "#185FA5",
    "Wenner": "#0F6E56",
    "Dipole-Dipole": "#993C1D",
}

SPASI_COLORS = {1: "#7F77DD", 2: "#185FA5", 3: "#0F6E56"}


def _log_scale(values: np.ndarray):
    """Konversi ke log10 untuk colorscale."""
    v = np.array(values, dtype=float)
    v = np.where(v <= 0, 0.1, v)
    return np.log10(v)


# ─── True Resistivity Section — gaya RES2DINV ────────────────────────────────

def plot_true_section(
    rho_matrix: np.ndarray,
    x_positions: list,
    depths: list,
    title: str = "Inverse Model Resistivity Section",
    height: int = 420,
    rho_min: float = None,
    rho_max: float = None,
) -> go.Figure:
    """
    Plot penampang true resistivity bergaya RES2DINV:
      - Bentuk trapezoid (tepi menyempit sesuai kedalaman)
      - Kontur smooth via griddata interpolasi
      - Colorscale identik RES2DINV
      - Titik elektroda di permukaan
      - Colorbar dengan tick label nilai resistivitas
    """
    if rho_matrix is None or rho_matrix.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada data model", xref="paper",
                           yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    mat = rho_matrix.copy().astype(float)
    mat = np.where(mat <= 0, np.nan, mat)

    x_arr   = np.array(x_positions, dtype=float)
    z_arr   = np.array(depths, dtype=float)
    n_rows  = mat.shape[0]
    n_cols  = mat.shape[1]

    # ── Pastikan dimensi konsisten ────────────────────────────────────────────
    if len(x_arr) != n_cols:
        x_arr = np.linspace(0, (n_cols - 1) * 1.0, n_cols)
    if len(z_arr) != n_rows:
        z_arr = np.linspace(0.5, n_rows * 0.5, n_rows)

    x_min, x_max = float(x_arr.min()), float(x_arr.max())
    z_min, z_max = float(z_arr.min()), float(z_arr.max())

    # ── Bangun titik scatter (x, z, rho) untuk interpolasi ───────────────────
    pts_x, pts_z, pts_rho = [], [], []
    for ri in range(n_rows):
        for ci in range(n_cols):
            v = mat[ri, ci]
            if not np.isnan(v):
                pts_x.append(x_arr[ci])
                pts_z.append(z_arr[ri])
                pts_rho.append(v)

    if len(pts_x) < 4:
        fig = go.Figure()
        fig.add_annotation(text="Data terlalu sedikit untuk interpolasi",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    pts_x   = np.array(pts_x)
    pts_z   = np.array(pts_z)
    pts_rho = np.array(pts_rho)

    # ── Grid halus untuk interpolasi ─────────────────────────────────────────
    n_xi = max(200, n_cols * 10)
    n_zi = max(100, n_rows * 10)
    xi   = np.linspace(x_min, x_max, n_xi)
    zi   = np.linspace(z_min, z_max, n_zi)
    XI, ZI = np.meshgrid(xi, zi)

    if SCIPY_OK:
        log_rho = np.log10(pts_rho)
        ZI_interp = griddata(
            (pts_x, pts_z), log_rho,
            (XI, ZI), method="linear"
        )
        # Isi NaN tepi dengan nearest
        ZI_nn = griddata(
            (pts_x, pts_z), log_rho,
            (XI, ZI), method="nearest"
        )
        ZI_interp = np.where(np.isnan(ZI_interp), ZI_nn, ZI_interp)
    else:
        # Fallback tanpa scipy: pakai matriks log asli
        log_mat = np.log10(mat)
        log_mat = np.where(np.isnan(log_mat), np.nanmean(log_mat), log_mat)
        ZI_interp = log_mat

    # ── Masking trapezoid ─────────────────────────────────────────────────────
    # RES2DINV: lebar investigasi menyempit sesuai kedalaman
    # Sisi kiri: x_min + (z/z_max)*margin  |  Sisi kanan: x_max - (z/z_max)*margin
    # margin ~ 1 spasi elektroda per sisi
    dx_total = x_max - x_min
    margin_factor = 0.5  # bisa disesuaikan

    for zi_idx, z_val in enumerate(zi):
        frac = (z_val - z_min) / (z_max - z_min + 1e-9)
        margin = frac * dx_total * margin_factor
        left_bound  = x_min + margin
        right_bound = x_max - margin
        mask_row = (xi < left_bound) | (xi > right_bound)
        ZI_interp[zi_idx, mask_row] = np.nan

    # ── Colorbar ticks ────────────────────────────────────────────────────────
    valid_log = ZI_interp[~np.isnan(ZI_interp)]
    if len(valid_log) == 0:
        valid_log = np.array([1.0, 3.0])

    vmin_log = np.log10(rho_min)  if (rho_min and rho_min > 0) else float(np.nanmin(valid_log))
    vmax_log = np.log10(rho_max)  if (rho_max and rho_max > 0) else float(np.nanmax(valid_log))
    vmin_log = max(vmin_log, 0.0)

    # Tick pada posisi log yang "bulat"
    tick_log_vals = np.linspace(vmin_log, vmax_log, 8)
    tick_rho_vals = np.power(10, tick_log_vals)
    # Pembulatan ke angka yang enak dibaca
    def _nice(v):
        if v < 10:
            return round(v, 1)
        elif v < 100:
            return round(v)
        elif v < 1000:
            return round(v / 5) * 5
        else:
            return round(v / 50) * 50

    tick_texts = [str(_nice(v)) for v in tick_rho_vals]

    # ── Gambar Heatmap ────────────────────────────────────────────────────────
    # NaN pada ZI_interp akan dirender transparan oleh Plotly secara otomatis
    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=ZI_interp,
        x=xi,
        y=zi,
        colorscale=COLORSCALE_RES2DINV,
        zmin=vmin_log,
        zmax=vmax_log,
        colorbar=dict(
            title=dict(text="Resistivity (Ω·m)", side="right", font=dict(size=11)),
            tickvals=tick_log_vals.tolist(),
            ticktext=tick_texts,
            thickness=16,
            len=0.92,
            tickfont=dict(size=10),
            outlinewidth=0.5,
            outlinecolor="#aaa",
        ),
        hovertemplate=(
            "x = %{x:.2f} m<br>"
            "z = %{y:.2f} m<br>"
            "ρ ≈ %{customdata:.1f} Ω·m<extra></extra>"
        ),
        customdata=np.power(10, np.where(np.isnan(ZI_interp), 0, ZI_interp)),
        zsmooth="best",
        connectgaps=False,
    ))

    # ── Overlay putih kiri dan kanan untuk membentuk trapezoid ───────────────
    # Strategi: gambar dua polygon putih di atas heatmap
    # Segitiga kiri: (x_min, z_min) → (x_min, z_max) → (bottom_left, z_max) → kembali
    # Segitiga kanan: (x_max, z_min) → (bottom_right, z_max) → (x_max, z_max) → kembali
    bottom_left  = x_min + dx_total * margin_factor
    bottom_right = x_max - dx_total * margin_factor

    # Polygon kiri (area luar kiri trapezoid)
    fig.add_trace(go.Scatter(
        x=[x_min, x_min, bottom_left, x_min],
        y=[z_min,  z_max, z_max,       z_min],
        mode="lines",
        fill="toself",
        fillcolor="white",
        line=dict(color="white", width=0),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Polygon kanan (area luar kanan trapezoid)
    fig.add_trace(go.Scatter(
        x=[x_max, bottom_right, x_max, x_max],
        y=[z_min,  z_max,        z_max, z_min],
        mode="lines",
        fill="toself",
        fillcolor="white",
        line=dict(color="white", width=0),
        hoverinfo="skip",
        showlegend=False,
    ))

    # ── Garis outline trapezoid (border hitam tipis) ──────────────────────────
    fig.add_trace(go.Scatter(
        x=[x_min, bottom_left,  bottom_right, x_max,  x_min],
        y=[z_min,  z_max,        z_max,         z_min,  z_min],
        mode="lines",
        line=dict(color="black", width=1.5),
        hoverinfo="skip",
        showlegend=False,
        fill=None,
    ))

    # ── Titik datum (elektroda) di permukaan ─────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_arr,
        y=np.full_like(x_arr, z_min),
        mode="markers",
        marker=dict(
            size=5,
            color="black",
            symbol="circle",
            line=dict(width=0),
        ),
        name="Elektroda",
        hovertemplate="Elektroda x=%{x:.1f}m<extra></extra>",
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=13, family="Arial", color="#111"),
            x=0.0, xanchor="left",
        ),
        xaxis=dict(
            title=dict(text="Distance (m)", font=dict(size=11)),
            tickfont=dict(size=10),
            side="top",                    # Skala jarak di atas, seperti RES2DINV
            showgrid=False,
            zeroline=False,
            range=[x_min - dx_total * 0.02, x_max + dx_total * 0.02],
            ticks="outside",
            ticklen=4,
        ),
        yaxis=dict(
            title=dict(text="Depth (m)", font=dict(size=11)),
            tickfont=dict(size=10),
            autorange="reversed",
            showgrid=False,
            zeroline=False,
            ticks="outside",
            ticklen=4,
        ),
        height=height,
        margin=dict(l=65, r=90, t=60, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=11),
        hovermode="closest",
        showlegend=False,
    )

    return fig


# ─── Pseudosection plot ───────────────────────────────────────────────────────

def plot_pseudosection(
    datum_points: np.ndarray,
    title: str = "Pseudosection Apparent Resistivity",
    colorbar_title: str = "ρa (Ω·m)",
    height: int = 350,
    rho_min: float = None,
    rho_max: float = None,
    show_values: bool = False,
) -> go.Figure:
    """
    Plot pseudosection sebagai scatter plot bubble.
    datum_points: array (N, 3) → [x_mid, pseudo_depth, rho_a]
    """
    if datum_points is None or len(datum_points) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    x = datum_points[:, 0]
    z = datum_points[:, 1]
    rho = datum_points[:, 2]

    log_rho = _log_scale(rho)
    vmin = _log_scale(np.array([rho_min]))[0] if rho_min else log_rho.min()
    vmax = _log_scale(np.array([rho_max]))[0] if rho_max else log_rho.max()

    hover_text = [
        f"<b>x</b> = {xi:.1f} m<br>"
        f"<b>Pseudo-depth</b> = {zi:.2f} m<br>"
        f"<b>ρa</b> = {r:.1f} Ω·m<br>"
        f"<b>Litologi</b>: {classify_material(r)[0]}"
        for xi, zi, r in zip(x, z, rho)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=z,
        mode="markers",
        marker=dict(
            size=14,
            color=log_rho,
            colorscale=COLORSCALE_GEO,
            cmin=vmin, cmax=vmax,
            colorbar=dict(
                title=dict(text=colorbar_title, side="right"),
                tickvals=[vmin, (vmin + vmax) / 2, vmax],
                ticktext=[
                    f"{10**vmin:.0f}",
                    f"{10**((vmin+vmax)/2):.0f}",
                    f"{10**vmax:.0f}"
                ],
                len=0.9,
                thickness=14,
            ),
            symbol="square",
            line=dict(width=0.3, color="rgba(0,0,0,0.2)"),
        ),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
    ))

    if show_values:
        fig.add_trace(go.Scatter(
            x=x, y=z,
            mode="text",
            text=[f"{r:.0f}" for r in rho],
            textfont=dict(size=8, color="white"),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1a1a1a"), x=0.02),
        xaxis=dict(title="Jarak (m)", gridcolor="#f0f0f0", zeroline=False),
        yaxis=dict(
            title="Pseudo-depth (m)",
            autorange="reversed",
            gridcolor="#f0f0f0",
            zeroline=False,
        ),
        height=height,
        margin=dict(l=60, r=80, t=45, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
        hovermode="closest",
    )
    return fig


# ─── RMS convergence plot ─────────────────────────────────────────────────────

def plot_rms_convergence(rms_history: list) -> go.Figure:
    """Plot grafik konvergensi RMS error."""
    if not rms_history:
        return go.Figure()

    iters = list(range(1, len(rms_history) + 1))
    colors = ["#D85A30" if v > 5 else "#0F6E56" for v in rms_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=rms_history,
        fill="tozeroy",
        fillcolor="rgba(24, 95, 165, 0.08)",
        line=dict(color="#185FA5", width=2.5),
        mode="lines+markers",
        marker=dict(size=10, color=colors, line=dict(width=1.5, color="white")),
        name="RMS Error",
        hovertemplate="Iterasi %{x}: RMS = %{y:.2f}%<extra></extra>",
    ))

    fig.add_hline(
        y=5.0,
        line=dict(color="#D85A30", dash="dash", width=1.5),
        annotation_text="Batas 5%",
        annotation_position="right",
        annotation_font=dict(color="#D85A30", size=11),
    )

    if rms_history:
        fig.add_annotation(
            x=iters[-1], y=rms_history[-1],
            text=f"  RMS akhir: {rms_history[-1]:.1f}%",
            showarrow=False,
            font=dict(size=11, color="#0F6E56" if rms_history[-1] <= 5 else "#D85A30"),
            xanchor="left",
        )

    fig.update_layout(
        title=dict(text="Konvergensi Iterasi Inversi", font=dict(size=13), x=0.02),
        xaxis=dict(title="Iterasi ke-", tickmode="linear", dtick=1,
                   gridcolor="#f5f5f5"),
        yaxis=dict(title="RMS Error (%)", gridcolor="#f5f5f5", rangemode="tozero"),
        height=260,
        margin=dict(l=55, r=60, t=40, b=45),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
        showlegend=False,
    )
    return fig


# ─── Komparasi multi-panel ────────────────────────────────────────────────────

def plot_comparison_spasi(results: dict, rho_min=None, rho_max=None) -> go.Figure:
    """
    Plot komparasi pseudosection untuk variasi spasi elektroda.
    results: dict {spasi_value: ForwardResult}
    """
    spasi_vals = sorted(results.keys())
    n = len(spasi_vals)
    if n == 0:
        return go.Figure()

    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=[f"Spasi a = {s} m — {results[s].array_name}" for s in spasi_vals],
        vertical_spacing=0.08,
    )

    for idx, s in enumerate(spasi_vals):
        res = results[s]
        dp = res.datum_points
        if dp is None or len(dp) == 0:
            continue

        x, z, rho = dp[:, 0], dp[:, 1], dp[:, 2]
        log_rho = _log_scale(rho)
        vmin = np.log10(rho_min) if rho_min else log_rho.min()
        vmax = np.log10(rho_max) if rho_max else log_rho.max()

        fig.add_trace(go.Scatter(
            x=x, y=z,
            mode="markers",
            marker=dict(
                size=12, symbol="square",
                color=log_rho,
                colorscale=COLORSCALE_GEO,
                cmin=vmin, cmax=vmax,
                showscale=(idx == n - 1),
                colorbar=dict(
                    title="ρa (Ω·m)",
                    tickvals=[vmin, (vmin + vmax) / 2, vmax],
                    ticktext=[f"{10**vmin:.0f}", f"{10**((vmin+vmax)/2):.0f}", f"{10**vmax:.0f}"],
                    len=0.9 / n,
                    y=1 - (idx + 0.5) / n,
                    thickness=12,
                ),
            ),
            hovertemplate=f"a={s}m | x=%{{x:.1f}}m | z=%{{y:.2f}}m | ρa=%{{customdata:.1f}} Ω·m<extra></extra>",
            customdata=rho,
            showlegend=False,
        ), row=idx + 1, col=1)

        fig.update_yaxes(autorange="reversed", title_text="Depth (m)",
                         row=idx + 1, col=1)
        fig.update_xaxes(title_text="Jarak (m)" if idx == n - 1 else "",
                         row=idx + 1, col=1)

    fig.update_layout(
        height=280 * n,
        margin=dict(l=60, r=90, t=50, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11),
        title=dict(
            text="Komparasi Pseudosection — Variasi Spasi Elektroda",
            font=dict(size=14), x=0.02
        ),
    )
    return fig


def plot_comparison_array(results: dict, rho_min=None, rho_max=None) -> go.Figure:
    """
    Plot komparasi pseudosection untuk variasi konfigurasi.
    results: dict {array_name: ForwardResult}
    """
    array_names = list(results.keys())
    n = len(array_names)
    if n == 0:
        return go.Figure()

    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=[f"{name} — a = {results[name].electrode_spacing} m" for name in array_names],
        vertical_spacing=0.08,
    )

    for idx, name in enumerate(array_names):
        res = results[name]
        dp = res.datum_points
        if dp is None or len(dp) == 0:
            continue

        x, z, rho = dp[:, 0], dp[:, 1], dp[:, 2]
        log_rho = _log_scale(rho)
        vmin = np.log10(rho_min) if rho_min else log_rho.min()
        vmax = np.log10(rho_max) if rho_max else log_rho.max()

        fig.add_trace(go.Scatter(
            x=x, y=z,
            mode="markers",
            marker=dict(
                size=12, symbol="square",
                color=log_rho,
                colorscale=COLORSCALE_GEO,
                cmin=vmin, cmax=vmax,
                showscale=(idx == n - 1),
                colorbar=dict(
                    title="ρa (Ω·m)",
                    tickvals=[vmin, (vmin + vmax) / 2, vmax],
                    ticktext=[f"{10**vmin:.0f}", f"{10**((vmin+vmax)/2):.0f}", f"{10**vmax:.0f}"],
                    len=0.9 / n,
                    y=1 - (idx + 0.5) / n,
                    thickness=12,
                ),
            ),
            hovertemplate=f"{name} | x=%{{x:.1f}}m | z=%{{y:.2f}}m | ρa=%{{customdata:.1f}} Ω·m<extra></extra>",
            customdata=rho,
            showlegend=False,
        ), row=idx + 1, col=1)

        fig.update_yaxes(autorange="reversed", title_text="Depth (m)",
                         row=idx + 1, col=1)
        fig.update_xaxes(title_text="Jarak (m)" if idx == n - 1 else "",
                         row=idx + 1, col=1)

    fig.update_layout(
        height=280 * n,
        margin=dict(l=60, r=90, t=50, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11),
        title=dict(
            text="Komparasi Pseudosection — Variasi Konfigurasi Elektroda",
            font=dict(size=14), x=0.02
        ),
    )
    return fig


def plot_layer_bar(layer_avgs: list) -> go.Figure:
    """Bar chart rata-rata resistivitas per lapisan."""
    if not layer_avgs:
        return go.Figure()

    labels = [f"Lap. {d['layer']}\n(z={d['depth']:.1f}m)" for d in layer_avgs]
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
        title=dict(text="Rata-rata Resistivitas per Lapisan", font=dict(size=13), x=0.02),
        xaxis=dict(title="Resistivitas (Ω·m)", type="log", gridcolor="#f0f0f0"),
        yaxis=dict(autorange="reversed"),
        height=230,
        margin=dict(l=90, r=80, t=40, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
    )
    return fig
