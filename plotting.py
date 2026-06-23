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


# ─── Helper: interpolasi pseudosection → heatmap ─────────────────────────────

def _build_pseudo_heatmap(
    datum_points: np.ndarray,
    vmin_log: float,
    vmax_log: float,
    show_colorbar: bool = True,
    colorbar_y: float = 0.5,
    colorbar_len: float = 0.9,
) -> go.Heatmap:
    """
    Bangun trace Heatmap dari datum_points pseudosection.
    Bentuk segitiga khas pseudosection: melebar ke bawah (sesuai konvensi).
    Masking: titik di luar batas n-level sesuai tiap kedalaman di-NaN-kan.
    """
    x   = datum_points[:, 0]
    z   = datum_points[:, 1]
    rho = datum_points[:, 2]

    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())

    # Grid interpolasi halus
    n_xi = 300
    n_zi = 150
    xi = np.linspace(x_min, x_max, n_xi)
    zi = np.linspace(z_min, z_max, n_zi)
    XI, ZI = np.meshgrid(xi, zi)

    log_rho = np.log10(np.where(rho <= 0, 0.1, rho))

    if SCIPY_OK and len(x) >= 4:
        ZI_interp = griddata((x, z), log_rho, (XI, ZI), method="linear")
        ZI_nn     = griddata((x, z), log_rho, (XI, ZI), method="nearest")
        ZI_interp = np.where(np.isnan(ZI_interp), ZI_nn, ZI_interp)
    else:
        # Fallback: nearest tanpa scipy
        ZI_interp = np.full((n_zi, n_xi), np.mean(log_rho))

    # ── Masking: pseudosection melebar ke bawah (segitiga) ───────────────────
    # Pada kedalaman z, titik datum valid hanya di antara x_min_valid(z)..x_max_valid(z)
    # Kita estimasi dari data aktual: per level z, cari rentang x yang ada
    z_levels = np.unique(np.round(z, 4))
    for zi_idx, z_val in enumerate(zi):
        # Cari level z terdekat di data
        nearest_level = z_levels[np.argmin(np.abs(z_levels - z_val))]
        mask = np.abs(z - nearest_level) < (z_max - z_min) / (len(z_levels) * 2 + 1e-9)
        if mask.sum() == 0:
            ZI_interp[zi_idx, :] = np.nan
            continue
        x_left  = float(x[mask].min())
        x_right = float(x[mask].max())
        # Tambahkan sedikit buffer agar tepian tidak terpotong terlalu ketat
        dx_buf = (x_max - x_min) / (n_xi * 0.5)
        row_mask = (xi < x_left - dx_buf) | (xi > x_right + dx_buf)
        ZI_interp[zi_idx, row_mask] = np.nan

    # ── Colorbar ticks ────────────────────────────────────────────────────────
    tick_log = np.linspace(vmin_log, vmax_log, 7)
    def _nice(v):
        if v < 10:   return round(v, 1)
        if v < 100:  return round(v)
        if v < 1000: return round(v / 5) * 5
        return round(v / 50) * 50
    tick_text = [str(_nice(10**lv)) for lv in tick_log]

    return go.Heatmap(
        z=ZI_interp,
        x=xi,
        y=zi,
        colorscale=COLORSCALE_RES2DINV,
        zmin=vmin_log,
        zmax=vmax_log,
        colorbar=dict(
            title=dict(text="ρa (Ω·m)", side="right", font=dict(size=11)),
            tickvals=tick_log.tolist(),
            ticktext=tick_text,
            thickness=15,
            len=colorbar_len,
            y=colorbar_y,
            tickfont=dict(size=10),
            outlinewidth=0.5,
        ) if show_colorbar else None,
        showscale=show_colorbar,
        hovertemplate=(
            "x = %{x:.2f} m<br>"
            "Pseudo-depth = %{y:.2f} m<br>"
            "ρa ≈ %{customdata:.1f} Ω·m<extra></extra>"
        ),
        customdata=np.power(10, np.where(np.isnan(ZI_interp), 0, ZI_interp)),
        zsmooth="best",
        connectgaps=False,
    )


def _pseudo_outline_traces(datum_points: np.ndarray):
    """
    Buat dua polygon putih kiri-kanan + border untuk membentuk segitiga pseudosection.
    Mengembalikan list of go.Scatter traces.
    """
    x   = datum_points[:, 0]
    z   = datum_points[:, 1]

    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())

    z_levels = np.unique(np.round(z, 4))
    # Kumpulkan batas kiri dan kanan per z
    left_pts  = []   # (z, x_left)
    right_pts = []   # (z, x_right)
    for zl in sorted(z_levels):
        mask = np.abs(z - zl) < 1e-6
        if mask.sum() == 0:
            continue
        left_pts.append( (zl, float(x[mask].min())) )
        right_pts.append((zl, float(x[mask].max())) )

    if not left_pts:
        return []

    # Sisi kiri outline (dari atas ke bawah)
    lz = [p[0] for p in left_pts]
    lx = [p[1] for p in left_pts]
    rz = [p[0] for p in right_pts]
    rx = [p[1] for p in right_pts]

    # Polygon putih KIRI: dari x_min atas → tepi kiri bawah → kembali
    poly_left_x = [x_min] + lx + [x_min, x_min]
    poly_left_z = [z_min] + lz + [z_max, z_min]

    # Polygon putih KANAN: dari x_max atas → tepi kanan bawah → kembali
    poly_right_x = [x_max] + rx[::-1] + [x_max, x_max]
    poly_right_z = [z_min] + rz[::-1] + [z_max, z_min]

    # Border outline
    border_x = [x_min] + lx + rx[::-1] + [x_max, x_min]
    border_z = [z_min] + lz + rz[::-1] + [z_min, z_min]

    traces = [
        go.Scatter(x=poly_left_x,  y=poly_left_z,  mode="lines",
                   fill="toself", fillcolor="white", line=dict(color="white", width=0),
                   hoverinfo="skip", showlegend=False),
        go.Scatter(x=poly_right_x, y=poly_right_z, mode="lines",
                   fill="toself", fillcolor="white", line=dict(color="white", width=0),
                   hoverinfo="skip", showlegend=False),
        go.Scatter(x=border_x,     y=border_z,     mode="lines",
                   line=dict(color="black", width=1.2),
                   hoverinfo="skip", showlegend=False),
    ]
    return traces


# ─── Pseudosection plot ───────────────────────────────────────────────────────

def plot_pseudosection(
    datum_points: np.ndarray,
    title: str = "Pseudosection Apparent Resistivity",
    colorbar_title: str = "ρa (Ω·m)",
    height: int = 380,
    rho_min: float = None,
    rho_max: float = None,
    show_values: bool = False,
) -> go.Figure:
    """
    Plot pseudosection sebagai penampang warna penuh (heatmap smooth),
    bergaya RES2DINV dengan bentuk segitiga pseudosection.
    datum_points: array (N, 3) → [x_mid, pseudo_depth, rho_a]
    """
    if datum_points is None or len(datum_points) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    x   = datum_points[:, 0]
    z   = datum_points[:, 1]
    rho = datum_points[:, 2]

    log_rho = np.log10(np.where(rho <= 0, 0.1, rho))
    vmin_log = float(np.log10(rho_min)) if (rho_min and rho_min > 0) else float(log_rho.min())
    vmax_log = float(np.log10(rho_max)) if (rho_max and rho_max > 0) else float(log_rho.max())

    fig = go.Figure()

    # 1. Heatmap utama
    fig.add_trace(_build_pseudo_heatmap(datum_points, vmin_log, vmax_log))

    # 2. Overlay putih + border outline segitiga
    for tr in _pseudo_outline_traces(datum_points):
        fig.add_trace(tr)

    # 3. Opsional: label nilai di titik datum
    if show_values:
        fig.add_trace(go.Scatter(
            x=x, y=z,
            mode="text",
            text=[f"{r:.0f}" for r in rho],
            textfont=dict(size=7, color="white"),
            hoverinfo="skip",
            showlegend=False,
        ))

    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())
    dx = x_max - x_min
    dz = z_max - z_min

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=13, family="Arial"), x=0.0),
        xaxis=dict(
            title=dict(text="Distance (m)", font=dict(size=11)),
            side="top",
            showgrid=False, zeroline=False, ticks="outside", ticklen=4,
            range=[x_min - dx * 0.02, x_max + dx * 0.02],
        ),
        yaxis=dict(
            title=dict(text="Pseudo-depth (m)", font=dict(size=11)),
            autorange="reversed",
            showgrid=False, zeroline=False, ticks="outside", ticklen=4,
            range=[z_max + dz * 0.05, z_min - dz * 0.05],
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


# ─── Helper: satu panel pseudosection heatmap untuk subplots ─────────────────

def _add_pseudo_panel(
    fig: go.Figure,
    dp: np.ndarray,
    row: int,
    n_total: int,
    vmin_log: float,
    vmax_log: float,
    label: str,
):
    """Tambahkan satu panel pseudosection heatmap ke subplots figure."""
    x   = dp[:, 0]
    z   = dp[:, 1]
    rho = dp[:, 2]

    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())

    n_xi, n_zi = 300, 150
    xi = np.linspace(x_min, x_max, n_xi)
    zi = np.linspace(z_min, z_max, n_zi)
    XI, ZI = np.meshgrid(xi, zi)

    log_rho = np.log10(np.where(rho <= 0, 0.1, rho))

    if SCIPY_OK and len(x) >= 4:
        ZI_interp = griddata((x, z), log_rho, (XI, ZI), method="linear")
        ZI_nn     = griddata((x, z), log_rho, (XI, ZI), method="nearest")
        ZI_interp = np.where(np.isnan(ZI_interp), ZI_nn, ZI_interp)
    else:
        ZI_interp = np.full((n_zi, n_xi), np.mean(log_rho))

    # Masking per level z
    z_levels = np.unique(np.round(z, 4))
    for zi_idx, z_val in enumerate(zi):
        nearest = z_levels[np.argmin(np.abs(z_levels - z_val))]
        mask_pts = np.abs(z - nearest) < (z_max - z_min) / (len(z_levels) * 2 + 1e-9)
        if mask_pts.sum() == 0:
            ZI_interp[zi_idx, :] = np.nan
            continue
        x_left  = float(x[mask_pts].min())
        x_right = float(x[mask_pts].max())
        dx_buf  = (x_max - x_min) / (n_xi * 0.5)
        row_mask = (xi < x_left - dx_buf) | (xi > x_right + dx_buf)
        ZI_interp[zi_idx, row_mask] = np.nan

    # Colorbar hanya pada panel terakhir
    is_last = (row == n_total)
    tick_log  = np.linspace(vmin_log, vmax_log, 7)
    def _nice(v):
        if v < 10:   return round(v, 1)
        if v < 100:  return round(v)
        if v < 1000: return round(v / 5) * 5
        return round(v / 50) * 50
    tick_text = [str(_nice(10**lv)) for lv in tick_log]

    fig.add_trace(go.Heatmap(
        z=ZI_interp, x=xi, y=zi,
        colorscale=COLORSCALE_RES2DINV,
        zmin=vmin_log, zmax=vmax_log,
        showscale=is_last,
        colorbar=dict(
            title=dict(text="ρa (Ω·m)", side="right", font=dict(size=10)),
            tickvals=tick_log.tolist(), ticktext=tick_text,
            thickness=14, len=0.9, tickfont=dict(size=9),
        ) if is_last else None,
        hovertemplate=(
            f"{label}<br>x=%{{x:.2f}} m | z=%{{y:.2f}} m<br>"
            "ρa ≈ %{customdata:.1f} Ω·m<extra></extra>"
        ),
        customdata=np.power(10, np.where(np.isnan(ZI_interp), 0, ZI_interp)),
        zsmooth="best",
        connectgaps=False,
    ), row=row, col=1)

    # Overlay putih kiri-kanan
    for tr in _pseudo_outline_traces(dp):
        fig.add_trace(tr, row=row, col=1)

    fig.update_yaxes(
        autorange="reversed", title_text="Pseudo-depth (m)",
        showgrid=False, zeroline=False, ticks="outside", ticklen=3,
        row=row, col=1
    )
    fig.update_xaxes(
        title_text="Distance (m)" if row == n_total else "",
        side="top" if row == 1 else "bottom",
        showgrid=False, zeroline=False, ticks="outside", ticklen=3,
        row=row, col=1
    )


# ─── Komparasi multi-panel ────────────────────────────────────────────────────

def plot_comparison_spasi(results: dict, rho_min=None, rho_max=None) -> go.Figure:
    """
    Plot komparasi pseudosection heatmap untuk variasi spasi elektroda.
    results: dict {spasi_value: ForwardResult}
    """
    spasi_vals = sorted(results.keys())
    n = len(spasi_vals)
    if n == 0:
        return go.Figure()

    # Hitung vmin/vmax global dari semua data
    all_rho = np.concatenate([results[s].datum_points[:, 2]
                               for s in spasi_vals
                               if results[s].datum_points is not None and len(results[s].datum_points) > 0])
    vmin_log = float(np.log10(rho_min)) if (rho_min and rho_min > 0) else float(np.log10(np.maximum(all_rho.min(), 0.1)))
    vmax_log = float(np.log10(rho_max)) if (rho_max and rho_max > 0) else float(np.log10(np.maximum(all_rho.max(), 0.1)))

    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=[f"Spasi a = {s} m  —  {results[s].array_name}" for s in spasi_vals],
        vertical_spacing=0.10,
    )

    for idx, s in enumerate(spasi_vals):
        dp = results[s].datum_points
        if dp is None or len(dp) == 0:
            continue
        _add_pseudo_panel(fig, dp, row=idx + 1, n_total=n,
                          vmin_log=vmin_log, vmax_log=vmax_log,
                          label=f"a={s}m")

    fig.update_layout(
        height=300 * n,
        margin=dict(l=65, r=90, t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=11),
        title=dict(text="<b>Komparasi Pseudosection — Variasi Spasi Elektroda</b>",
                   font=dict(size=13), x=0.0),
        showlegend=False,
    )
    return fig


def plot_comparison_array(results: dict, rho_min=None, rho_max=None) -> go.Figure:
    """
    Plot komparasi pseudosection heatmap untuk variasi konfigurasi.
    results: dict {array_name: ForwardResult}
    """
    array_names = list(results.keys())
    n = len(array_names)
    if n == 0:
        return go.Figure()

    all_rho = np.concatenate([results[nm].datum_points[:, 2]
                               for nm in array_names
                               if results[nm].datum_points is not None and len(results[nm].datum_points) > 0])
    vmin_log = float(np.log10(rho_min)) if (rho_min and rho_min > 0) else float(np.log10(np.maximum(all_rho.min(), 0.1)))
    vmax_log = float(np.log10(rho_max)) if (rho_max and rho_max > 0) else float(np.log10(np.maximum(all_rho.max(), 0.1)))

    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=[f"{nm}  —  a = {results[nm].electrode_spacing} m" for nm in array_names],
        vertical_spacing=0.10,
    )

    for idx, nm in enumerate(array_names):
        dp = results[nm].datum_points
        if dp is None or len(dp) == 0:
            continue
        _add_pseudo_panel(fig, dp, row=idx + 1, n_total=n,
                          vmin_log=vmin_log, vmax_log=vmax_log,
                          label=nm)

    fig.update_layout(
        height=300 * n,
        margin=dict(l=65, r=90, t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=11),
        title=dict(text="<b>Komparasi Pseudosection — Variasi Konfigurasi Elektroda</b>",
                   font=dict(size=13), x=0.0),
        showlegend=False,
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
