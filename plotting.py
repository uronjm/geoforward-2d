"""
Modul visualisasi penampang geolistrik 2D menggunakan Plotly.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from forward_model import ForwardResult, classify_material


# ─── Skala warna geolistrik ──────────────────────────────────────────────────

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

SPASI_COLORS = {1: "#7F77DD", 2: "#185FA5", 3: "#0F6E56"}


def _log_scale(values: np.ndarray):
    """Konversi ke log10 untuk colorscale."""
    v = np.array(values, dtype=float)
    v = np.where(v <= 0, 0.1, v)
    return np.log10(v)


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


# ─── True resistivity section ────────────────────────────────────────────────

def plot_true_section(
    rho_matrix: np.ndarray,
    x_positions: list,
    depths: list,
    title: str = "Penampang True Resistivity 2D",
    height: int = 380,
    rho_min: float = None,
    rho_max: float = None,
) -> go.Figure:
    """Plot penampang true resistivity sebagai heatmap."""
    if rho_matrix is None or rho_matrix.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada data model", xref="paper",
                           yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    mat = rho_matrix.copy().astype(float)
    mat = np.where(mat <= 0, np.nan, mat)
    log_mat = np.where(np.isnan(mat), np.nan, np.log10(mat))

    vmin = np.log10(rho_min) if rho_min else np.nanmin(log_mat)
    vmax = np.log10(rho_max) if rho_max else np.nanmax(log_mat)

    n_ticks = 6
    tick_vals = np.linspace(vmin, vmax, n_ticks)
    tick_text = [f"{10**v:.0f}" for v in tick_vals]

    hover = [[
        f"x={x_positions[ci] if ci < len(x_positions) else ci:.1f} m, "
        f"z={depths[ri] if ri < len(depths) else ri:.1f} m, "
        f"ρ={mat[ri,ci]:.1f} Ω·m"
        for ci in range(mat.shape[1])
    ] for ri in range(mat.shape[0])]

    fig = go.Figure(go.Heatmap(
        z=log_mat,
        x=x_positions if len(x_positions) == log_mat.shape[1] else list(range(log_mat.shape[1])),
        y=depths if len(depths) == log_mat.shape[0] else list(range(log_mat.shape[0])),
        colorscale=COLORSCALE_GEO,
        zmin=vmin, zmax=vmax,
        colorbar=dict(
            title=dict(text="ρ (Ω·m)", side="right"),
            tickvals=tick_vals.tolist(),
            ticktext=tick_text,
            len=0.9,
            thickness=14,
        ),
        hovertext=hover,
        hovertemplate="%{hovertext}<extra></extra>",
        xgap=1, ygap=1,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1a1a1a"), x=0.02),
        xaxis=dict(title="Jarak (m)", gridcolor="#f0f0f0"),
        yaxis=dict(
            title="Kedalaman (m)",
            autorange="reversed",
            gridcolor="#f0f0f0",
        ),
        height=height,
        margin=dict(l=60, r=90, t=45, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
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

    # Area fill
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

    # Garis batas 5%
    fig.add_hline(
        y=5.0,
        line=dict(color="#D85A30", dash="dash", width=1.5),
        annotation_text="Batas 5%",
        annotation_position="right",
        annotation_font=dict(color="#D85A30", size=11),
    )

    # Annotasi nilai akhir
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
