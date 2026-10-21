"""
Interactive (Plotly) drawing of an extension profile and its distinguished points.
"""

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ...exceptions import ditException

__all__ = ("plot_extension_profile",)

# Overlays are drawn this far above their true height so WebGL does not bury
# bottom-face points inside the coplanar surface; hover text shows true values.
_LIFT = 0.008

_COLORS = {
    "Gács–Körner": "#d62728",
    "Wyner": "#1f77b4",
    "exact": "#9467bd",
    "functional": "#8c564b",
    "MSS": "#7f7f7f",
    "entanglement": "#2ca02c",
    "Kamath": "#ff7f0e",
    "relaxed Wyner": "#17becf",
    "β-common": "#e377c2",
    "information bottleneck": "#984ea3",
    "privacy funnel": "#31a354",
    "excess functional information": "#e7298a",
    "Körner graph entropy": "#66a61e",
    "interaction informations": "#a6761d",
}


def plot_extension_profile(profile, features, show_upper=True, show_gray_wyner_rates=True, height=850):
    """
    Draw an extension profile :cite:`matveev2026beyond` and its distinguished points.

    The lower boundary is shaded by height (it includes the flat bottom face
    :math:`I[X:Y|W] = 0`); the upper boundary, the inner bound
    :math:`\\mathrm{Ext}_{\\min}` and the Shannon outer bound
    :math:`\\mathrm{Ext}_{\\max}` are drawn for context. Every feature is a legend
    entry that can be toggled, and hovering shows its coordinates and, optionally,
    its Gray-Wyner rates :math:`(R_0, R_1, R_2) = (I[XY:W], H[X|W], H[Y|W])`
    :cite:`gray1974source`.

    Parameters
    ----------
    profile : ExtensionProfile
        The profile (for its entropies).
    features : dict
        The output of `ExtensionProfile.compute`.
    show_upper : bool
        Whether to draw the sampled upper boundary.
    show_gray_wyner_rates : bool
        Whether hover text includes the Gray-Wyner rates.
    height : int
        Figure height in pixels.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The interactive figure.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as err:  # pragma: no cover
        msg = "plot_extension_profile requires plotly (pip install 'dit[optional]')."
        raise ditException(msg) from err

    h_xy = profile.h_xy

    def r0(p):
        return h_xy - p[0] - p[1] + p[2]

    def hover(name, p, extra=""):
        text = f"<b>{name}</b><br>H[X|W] = {p[0]:.3f}<br>H[Y|W] = {p[1]:.3f}<br>I[X:Y|W] = {p[2]:.3f}{extra}"
        if show_gray_wyner_rates:
            text += f"<br>Gray–Wyner (R0, R1, R2) = ({r0(p):.3f}, {p[0]:.3f}, {p[1]:.3f})"
        return text

    def line(pts, name, color, width=6, dash=None, text=None, group=None, showlegend=True):
        pts = np.atleast_2d(np.asarray(pts, float))
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2] + _LIFT,
            mode="lines",
            name=name,
            legendgroup=group or name,
            showlegend=showlegend,
            line={"color": color, "width": width, "dash": dash},
            text=text,
            hoverinfo="text" if text is not None else "skip",
        )

    def markers(pts, name, color, symbol, texts, size=7, group=None, showlegend=True):
        pts = np.atleast_2d(np.asarray(pts, float))
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2] + _LIFT,
            mode="markers",
            name=name,
            legendgroup=group or name,
            showlegend=showlegend,
            marker={"color": color, "symbol": symbol, "size": size, "line": {"color": "black", "width": 1}},
            text=texts,
            hoverinfo="text",
        )

    traces = []
    probes = features["probes"]
    top = probes["W = const"]

    # The body: one hull over every attained point, split into downward and upward faces.
    extra = [*probes.values(), features["Gács–Körner"], features["functional"], features["MSS"]]
    for key in ("relaxed Wyner",):
        extra += list(features.get(key, []))
    for key in (
        "information bottleneck",
        "privacy funnel",
        "Kamath",
        "excess functional information",
        "Körner graph entropy",
    ):
        for arc in features.get(key, []):
            extra += list(np.atleast_2d(arc))
    pts = np.vstack([features["lower"], *([features["upper"]] if show_upper else []), np.array(extra)])
    try:
        hull = ConvexHull(pts)
    except QhullError:  # a body of dimension < 3 (e.g. X = Y) has no surface to draw
        hull = None
    lower = (
        np.zeros((0, 3), int)
        if hull is None
        else np.array([s for s, e in zip(hull.simplices, hull.equations, strict=True) if e[2] < -1e-6]).reshape(-1, 3)
    )
    upper = (
        np.zeros((0, 3), int)
        if hull is None
        else np.array([s for s, e in zip(hull.simplices, hull.equations, strict=True) if e[2] >= -1e-6]).reshape(-1, 3)
    )
    if len(lower):
        tri = pts[lower]
        flip = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2] < 0
        lower[flip] = lower[flip][:, [0, 2, 1]]  # consistent winding, so every face is lit from the same side
        custom = np.c_[[r0(p) for p in pts], pts[:, :2]]
        traces.append(
            go.Mesh3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                i=lower[:, 0],
                j=lower[:, 1],
                k=lower[:, 2],
                intensity=pts[:, 2],
                colorscale=[[0, "#c6dbef"], [1, "#2171b5"]],
                cmin=0,
                cmax=max(top[2], 1e-9),
                showscale=False,
                opacity=0.6,
                name="lower boundary of Ext",
                showlegend=True,
                lighting={"ambient": 0.9, "diffuse": 0.2, "specular": 0.0},
                customdata=custom,
                hovertemplate="H[X|W] = %{x:.3f}<br>H[Y|W] = %{y:.3f}<br>I[X:Y|W] = %{z:.3f}"
                "<br>Gray–Wyner (R0, R1, R2) = (%{customdata[0]:.3f}, %{customdata[1]:.3f}, "
                "%{customdata[2]:.3f})<extra>lower boundary</extra>",
            )
        )
    if show_upper and len(upper):
        traces.append(
            go.Mesh3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                i=upper[:, 0],
                j=upper[:, 1],
                k=upper[:, 2],
                color="#fdae6b",
                opacity=0.3,
                flatshading=True,
                name="upper boundary of Ext",
                showlegend=True,
                hoverinfo="x+y+z",
                lighting={"ambient": 0.8, "diffuse": 0.4},
            )
        )

    # Ext_min (tetrahedron of the probes) and Ext_max (Shannon wireframe).
    tet = np.array([probes["W = XY"], probes["W = X"], probes["W = Y"], top])
    traces.append(
        go.Mesh3d(
            x=tet[:, 0],
            y=tet[:, 1],
            z=tet[:, 2],
            i=[0, 0, 0, 1],
            j=[1, 1, 2, 2],
            k=[2, 3, 3, 3],
            color="#636363",
            opacity=0.12,
            name="Ext_min",
            legendgroup="ext_min",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    edges = []
    for a in range(4):
        for b in range(a + 1, 4):
            edges += [tet[a], tet[b], [np.nan] * 3]
    traces.append(
        line(
            np.asarray(edges) - _LIFT, "Ext_min (W = XY, X, Y, const)", "#252525", width=4, dash="dash", group="ext_min"
        )
    )
    wire = []
    for a, b in features["ext_max"][1]:
        wire += [a, b, [np.nan] * 3]
    traces.append(line(np.asarray(wire) - _LIFT, "Ext_max (Shannon bound)", "gray", width=2, dash="dot"))

    # Edges of the body: Gács-Körner (both leakages vanish) and Kamath (bottom face meets one leakage face).
    gk = features["Gács–Körner"]
    traces.append(
        line([gk, top], "Gács–Körner edge", _COLORS["Gács–Körner"], width=10, text=["W = common part", "W = const"])
    )
    kam, kv = features["Kamath"], features["Kamath values"]
    traces.append(
        line(
            [probes["W = Y"], kam[0]],
            "Kamath edges",
            _COLORS["Kamath"],
            width=10,
            group="kamath",
            text=["W = Y", "W = MSS of Y for X"],
        )
    )
    traces.append(
        line(
            [probes["W = X"], kam[1]],
            "Kamath edges",
            _COLORS["Kamath"],
            width=10,
            group="kamath",
            showlegend=False,
            text=["W = X", "W = MSS of X for Y"],
        )
    )

    # Bottom-face rim between the two Kamath endpoints.
    base = pts[np.abs(pts[:, 2]) < 1e-9]
    try:
        ring = base[ConvexHull(base[:, :2]).vertices] if len(base) >= 3 else None
    except QhullError:  # the bottom face is a point or a segment
        ring = None
    if ring is not None:
        rim = ring[ring[:, 0] + ring[:, 1] > min(kam[0][:2].sum(), kam[1][:2].sum()) + 1e-6]
        rim = np.vstack([kam[0], rim[np.argsort(rim[:, 1] - rim[:, 0])], kam[1]])
        traces.append(line(rim, "bottom-face rim", "#08519c", width=6, text=[hover("bottom-face rim", p) for p in rim]))

    # Curves.
    if "relaxed Wyner" in features:
        rw = features["relaxed Wyner"]
        traces.append(
            line(
                rw,
                "relaxed Wyner",
                _COLORS["relaxed Wyner"],
                width=7,
                text=[hover("relaxed Wyner", p, f"<br>γ = residual = {p[2]:.3f}") for p in rw],
            )
        )
    if "β-common" in features:
        betas, bc = features["β-common"]
        traces.append(
            line(
                bc,
                "β-common",
                _COLORS["β-common"],
                width=7,
                dash="dash",
                text=[hover("β-common", p, f"<br>β = {b:.2f}") for p, b in zip(bc, betas, strict=True)],
            )
        )
    for key, label in (("information bottleneck", "information bottleneck"), ("privacy funnel", "privacy funnel")):
        for n, (arc, tag) in enumerate(zip(features.get(key, []), ("X → Y", "Y → X"), strict=False)):
            traces.append(
                line(
                    arc,
                    f"{label} (both zero-leakage faces)",
                    _COLORS[key],
                    width=6,
                    group=key,
                    showlegend=n == 0,
                    text=[hover(f"{label} {tag}", p) for p in arc],
                )
            )

    # Points.
    def add(name, p, symbol, extra="", size=7):
        traces.append(markers(p, name, _COLORS[name], symbol, [hover(name, p, extra)], size=size))

    add("Gács–Körner", gk, "circle", f"<br>K = {r0(gk):.3f}")
    if "Wyner" in features:
        add("Wyner", features["Wyner"], "square", f"<br>C = {r0(features['Wyner']):.3f}")
    if "exact" in features:
        add("exact", features["exact"], "diamond", "<br>(a realizer of the exact common information)")
    add("functional", features["functional"], "diamond-open", f"<br>F = {r0(features['functional']):.3f}")
    add("MSS", features["MSS"], "cross", f"<br>M = {r0(features['MSS']):.3f}")
    add("entanglement", features["entanglement"], "circle", f"<br>E = {features['entanglement value']:.3f}", size=10)
    traces.append(
        markers(
            np.array(kam),
            "Kamath / necessary conditional entropy",
            _COLORS["Kamath"],
            "square-open",
            [
                hover(
                    "W = MSS of Y for X",
                    kam[0],
                    f"<br>G(Y→X) = {kv['G Y -> X']:.3f}<br>H(Y†X) = {kv['necessary conditional entropy Y -> X']:.3f}",
                ),
                hover(
                    "W = MSS of X for Y",
                    kam[1],
                    f"<br>G(X→Y) = {kv['G X -> Y']:.3f}<br>H(X†Y) = {kv['necessary conditional entropy X -> Y']:.3f}",
                ),
            ],
            size=9,
            group="kamath",
        )
    )
    efi, efv = features["excess functional information"], features["excess functional information values"]
    if efi:
        traces.append(
            markers(
                np.array(efi),
                "excess functional information Ψ",
                _COLORS["excess functional information"],
                "diamond",
                [
                    hover(f"Ψ({tag}) corner", p, f"<br>Ψ = {efv[tag]:.4f}")
                    for p, tag in zip(efi, ("X -> Y", "Y -> X"), strict=False)
                ],
                size=9,
            )
        )
    ko, kov = features["Körner graph entropy"], features["Körner graph entropy values"]
    if ko:
        traces.append(
            markers(
                np.array(ko),
                "Körner graph entropy",
                _COLORS["Körner graph entropy"],
                "square",
                [
                    hover(f"Körner graph entropy H_K({tag})", p, f"<br>H_K = {kov[tag]:.3f}")
                    for p, tag in zip(ko, ("X", "Y"), strict=False)
                ],
                size=8,
            )
        )
    ii, iiv = features["interaction informations"], features["interaction information values"]
    add("interaction informations", ii, "x", "".join(f"<br>{k}: {v:.3f}" for k, v in iiv.items()), size=9)

    corner_pts = np.array(list(probes.values()))
    traces.append(
        go.Scatter3d(
            x=corner_pts[:, 0],
            y=corner_pts[:, 1],
            z=corner_pts[:, 2] + _LIFT,
            mode="markers+text",
            text=list(probes),
            textposition="top center",
            marker={"size": 3, "color": "black"},
            name="probe W's",
            hoverinfo="text",
        )
    )

    views = {
        "3D": {"x": 1.4, "y": -1.6, "z": 0.8},
        "bottom face, tilted": {"x": 0.3, "y": -0.9, "z": 2.0},
        "Gács–Körner edge end-on": {"x": 1.4, "y": 1.4, "z": 1.4},
    }
    buttons = [
        {
            "label": k,
            "method": "relayout",
            "args": [
                {
                    "scene.camera.eye": v,
                    "scene.camera.up": {"x": 0, "y": 0, "z": 1},
                    "scene.camera.center": {"x": 0, "y": 0, "z": 0},
                }
            ],
        }
        for k, v in views.items()
    ]
    zmax = max(float(pts[:, 2].max()), top[2])
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Extension profile  (H[XY] = {h_xy:.3f}, I[X:Y] = {profile.mutual_information:.3f})",
        height=height,
        scene={
            "xaxis_title": "H[X|W]",
            "yaxis_title": "H[Y|W]",
            "zaxis_title": "I[X:Y|W]",
            "xaxis_range": [0, profile.h_x * 1.01],
            "yaxis_range": [0, profile.h_y * 1.01],
            "zaxis_range": [0, zmax * 1.01 + 1e-9],
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": profile.h_y / max(profile.h_x, 1e-9), "z": 0.9},
            "camera": {"eye": views["3D"], "projection": {"type": "perspective"}},
        },
        legend={"itemsizing": "constant", "orientation": "h", "x": 0.0, "y": 0.0, "yanchor": "top"},
        updatemenus=[
            {"type": "buttons", "direction": "down", "x": 1.0, "y": 1.0, "xanchor": "right", "buttons": buttons}
        ],
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return fig
