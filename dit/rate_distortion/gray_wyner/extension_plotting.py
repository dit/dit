"""
Interactive (Plotly) drawing of an extension profile and its distinguished points.
"""

from contextlib import suppress

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ...exceptions import ditException

__all__ = ("plot_extension_profile", "plot_mutual_information_region", "plot_tension_region")

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
    "perfect privacy": "#006d2c",
    "channel synthesis": "#6a51a3",
}


def _extension_traces(go, profile, features, show_upper, show_gray_wyner_rates):
    """The traces of the extension-profile figure, and every attained point drawn (for hulls)."""
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
    extra += list(features.get("perfect privacy", []))
    extra += [p for k, p in features["interaction informations"].items() if k != "maximal"]
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
    color = _COLORS["interaction informations"]
    for key, label, symbol, constraint in (
        ("maximal", "maximal interaction information G_NNI", "x", "no constraint"),
        ("asymmetric private X -> Y", "asymmetric private interaction information G_PNI", "diamond-open", "U ⊥ X"),
        ("asymmetric private Y -> X", "asymmetric private interaction information G_PNI", "diamond-open", "U ⊥ Y"),
        ("symmetric private", "symmetric private interaction information G_PPI", "square-open", "U ⊥ X, U ⊥ Y"),
    ):
        traces.append(
            markers(
                ii[key],
                label,
                color,
                symbol,
                [hover(f"{label} ({key})", ii[key], f"<br>max I[X:Y|U] − I[X:Y], {constraint}<br>= {iiv[key]:.3f}")],
                size=10,
                group=label,
                showlegend=key != "asymmetric private Y -> X",
            )
        )
    if efi:
        for n, (a, b) in enumerate(
            zip(efi, (ii["asymmetric private X -> Y"], ii["asymmetric private Y -> X"]), strict=False)
        ):
            traces.append(
                line(
                    [a, b],
                    "edge U ⊥ X, H[Y|X,U] = 0 (Ψ ↔ G_PNI)",
                    color,
                    width=5,
                    dash="dot",
                    group="pni-edge",
                    showlegend=n == 0,
                    text=["Ψ end (min H[Y|U])", "G_PNI end (max H[Y|U])"],
                )
            )

    if "perfect privacy" in features:
        pp, ppv = features["perfect privacy"], features["perfect privacy values"]
        traces.append(
            markers(
                np.array(pp),
                "max rate for perfect privacy G_R*",
                _COLORS["perfect privacy"],
                "circle-open",
                [
                    hover(f"perfect privacy ({tag})", p, f"<br>G_R* = {ppv[tag]:.3f}")
                    for p, tag in zip(pp, ("X -> Y", "Y -> X"), strict=True)
                ],
                size=10,
            )
        )

    if "channel synthesis" in features:
        ts, curves = features["channel synthesis"]
        for n, ((values, pts_cs), tag) in enumerate(zip(curves, ("X → Y", "Y → X"), strict=True)):
            traces.append(
                line(
                    pts_cs,
                    "channel synthesis C(t)",
                    _COLORS["channel synthesis"],
                    width=9,
                    group="cs",
                    showlegend=n == 0,
                    text=[
                        hover(f"channel synthesis {tag}", p, f"<br>common randomness t = {t:.3f}<br>C(t) = {c:.3f}")
                        for p, t, c in zip(pts_cs, ts, values, strict=True)
                    ],
                )
            )

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

    return traces, pts


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

    traces, pts = _extension_traces(go, profile, features, show_upper, show_gray_wyner_rates)
    h_xy = profile.h_xy
    top = features["probes"]["W = const"]

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


def _import_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as err:  # pragma: no cover
        msg = "Plotting the extension profile requires plotly (pip install 'dit[optional]')."
        raise ditException(msg) from err
    return go


def _mapped(go, traces, fmap, labels, keep_meshes=True, rename=None):
    """Copies of `traces` with coordinates sent through the affine map `fmap`.

    Hover shows the new coordinates (at their true height: overlays carry `_LIFT` in z,
    and every map used here moves its third coordinate one-for-one with z) followed by
    the original description.
    """
    import copy

    rename = rename or {}
    out = []
    for t in traces:
        if isinstance(t, go.Mesh3d) and not keep_meshes:
            continue
        m = copy.deepcopy(t)
        a, b, c = fmap(np.asarray(t.x, float), np.asarray(t.y, float), np.asarray(t.z, float))
        m.x, m.y, m.z = a, b, c
        lift = _LIFT if isinstance(t, go.Scatter3d) else 0.0
        head = f"{labels[0]} = %{{customdata[0]:.3f}}<br>{labels[1]} = %{{customdata[1]:.3f}}<br>{labels[2]} = %{{customdata[2]:.3f}}"
        if isinstance(t, go.Scatter3d) and t.text is not None:
            m.customdata = np.c_[a, b, c - lift]
            m.hovertemplate = head + "<br>%{text}<extra></extra>"
        elif isinstance(t, go.Mesh3d) and t.hovertemplate:
            m.customdata = np.c_[a, b, c]
            m.hovertemplate = head + f"<extra>{rename.get(t.name, t.name)}</extra>"
        if t.name in rename:
            m.name = rename[t.name]
        out.append(m)
    return out


def _plane(go, corners, name, color):
    c = np.asarray(corners, float)
    n = len(c)
    return go.Mesh3d(
        x=c[:, 0],
        y=c[:, 1],
        z=c[:, 2],
        i=[0] * (n - 2),
        j=list(range(1, n - 1)),
        k=list(range(2, n)),
        color=color,
        opacity=0.18,
        name=name,
        showlegend=True,
        hoverinfo="name",
        visible="legendonly",
    )


def plot_mutual_information_region(profile, features, show_upper=True, height=850):
    """
    Draw the mutual information region of :cite:`li2018extended`,

    .. math::

        \\mathscr{I}_{XY} = \\{(I[X:U],\\ I[Y:U],\\ I[XY:U])\\},

    with every distinguished point. It is the extension profile under the invertible
    affine map :math:`(H[X] - x,\\ H[Y] - y,\\ H[XY] - x - y + z)`, so every trace of
    `plot_extension_profile` is mapped across. The planes :math:`v_X + v_Y - v_{XY} =
    I[X:Y]` (the image of the bottom face), :math:`v_Y = v_{XY}` (:math:`X - Y - U`) and
    :math:`v_X = v_{XY}` (:math:`Y - X - U`) can be toggled from the legend.

    Parameters
    ----------
    profile : ExtensionProfile
        The profile.
    features : dict
        The output of `ExtensionProfile.compute`.
    show_upper : bool
        Whether to draw the image of the sampled upper boundary.
    height : int
        Figure height in pixels.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The interactive figure.
    """
    go = _import_plotly()
    traces, _ = _extension_traces(go, profile, features, show_upper, show_gray_wyner_rates=False)
    h_x, h_y, h_xy, i = profile.h_x, profile.h_y, profile.h_xy, profile.mutual_information

    def fmap(x, y, z):
        return h_x - x, h_y - y, h_xy - x - y + z

    mi = _mapped(
        go,
        traces,
        fmap,
        ("I[X:U]", "I[Y:U]", "I[XY:U]"),
        rename={
            "lower boundary of Ext": "image of Ext's lower boundary",
            "upper boundary of Ext": "image of Ext's upper boundary",
        },
    )
    mi += [
        _plane(
            go,
            [(i, 0, 0), (h_x, 0, h_x - i), (h_x, h_y, h_xy), (0, h_y, h_y - i), (0, i, 0)],
            "plane v_X + v_Y − v_XY = I (bottom face)",
            "#3182bd",
        ),
        _plane(go, [(0, 0, 0), (h_x, 0, 0), (h_x, h_y, h_y), (0, h_y, h_y)], "plane v_Y = v_XY (X − Y − U)", "#984ea3"),
        _plane(go, [(0, 0, 0), (0, h_y, 0), (h_x, h_y, h_x), (h_x, 0, h_x)], "plane v_X = v_XY (Y − X − U)", "#984ea3"),
    ]
    fig = go.Figure(mi)
    fig.update_layout(
        title=f"Mutual information region  (H[XY] = {h_xy:.3f}, I[X:Y] = {i:.3f})",
        height=height,
        scene={
            "xaxis_title": "I[X:U]",
            "yaxis_title": "I[Y:U]",
            "zaxis_title": "I[XY:U]",
            "xaxis_range": [0, h_x * 1.01],
            "yaxis_range": [0, h_y * 1.01],
            "zaxis_range": [0, h_xy * 1.01],
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": h_y / max(h_x, 1e-9), "z": 1.2},
            "camera": {"eye": {"x": -1.6, "y": -1.5, "z": 0.9}},
        },
        legend={"itemsizing": "constant", "orientation": "h", "x": 0.0, "y": 0.0, "yanchor": "top"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return fig


def plot_tension_region(profile, features, height=850):
    """
    Draw the region of tension of :cite:`prabhakaran2014assisted`, in the coordinates
    :math:`\\tau = (I[Y:W|X],\\ I[X:W|Y],\\ I[X:Y|W])` (the two leakages and the residual),
    i.e. :math:`(H[Y|X] - y + z,\\ H[X|Y] - x + z,\\ z)` in extension-profile coordinates.

    The region is the image of the extension profile plus everything above it in
    :math:`\\tau`-space, so only directions :math:`\\alpha, \\beta \\ge 0`,
    :math:`\\alpha + \\beta \\le 1` of the shape function reach its boundary. The surface
    drawn is that minimal boundary, clipped to :math:`\\tau \\le (H[Y|X], H[X|Y], I[X:Y])`;
    every feature of `plot_extension_profile` is mapped across, so those off the surface
    are the ones the region of tension does not see. Its axis intercepts are marked:
    :math:`\\tau_3 = I[X:Y] - K[X:Y]` (the Gács-Körner point) and
    :math:`\\tau_1, \\tau_2` the necessary conditional entropies (the Kamath points).

    Parameters
    ----------
    profile : ExtensionProfile
        The profile.
    features : dict
        The output of `ExtensionProfile.compute`.
    height : int
        Figure height in pixels.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The interactive figure.
    """
    go = _import_plotly()
    traces, pts = _extension_traces(go, profile, features, show_upper=True, show_gray_wyner_rates=False)
    i = profile.mutual_information

    def fmap(x, y, z):
        return profile.h_y_given_x - y + z, profile.h_x_given_y - x + z, z

    box = np.array([profile.h_y_given_x, profile.h_x_given_y, i])
    t_traces = []
    tau = np.c_[fmap(pts[:, 0], pts[:, 1], pts[:, 2])]
    tau = tau[np.all(tau <= box + 1e-9, axis=1)]
    walls = []
    for axis in range(3):
        w = tau.copy()
        w[:, axis] = box[axis]
        walls.append(w)
    region = np.vstack([tau, *walls, box])
    faces = np.zeros((0, 3), int)
    if np.all(box > 1e-9):  # otherwise (e.g. X = Y) the region is lower-dimensional: no surface to draw
        with suppress(QhullError):
            hull = ConvexHull(region)
            faces = np.array(
                [s for s, e in zip(hull.simplices, hull.equations, strict=True) if np.all(e[:3] <= 1e-9)]
            ).reshape(-1, 3)
    if len(faces):
        t_traces.append(
            go.Mesh3d(
                x=region[:, 0],
                y=region[:, 1],
                z=region[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=region[:, 2],
                colorscale=[[0, "#e5f5e0"], [1, "#31a354"]],
                showscale=False,
                opacity=0.55,
                lighting={"ambient": 0.9, "diffuse": 0.2, "specular": 0.0},
                name="region of tension (lower boundary)",
                showlegend=True,
                hovertemplate="τ1 = I[Y:W|X] = %{x:.3f}<br>τ2 = I[X:W|Y] = %{y:.3f}"
                "<br>τ3 = I[X:Y|W] = %{z:.3f}<extra>region of tension</extra>",
            )
        )
    t_traces += _mapped(go, traces, fmap, ("τ1", "τ2", "τ3"), keep_meshes=False)

    kv = features["Kamath values"]
    gk = features["Gács–Körner"]
    k = profile.h_xy - gk[0] - gk[1] + gk[2]
    intercepts = {
        "τ3: I − K": [0, 0, i - k],
        "τ1: H(Y † X)": [kv["necessary conditional entropy Y -> X"], 0, 0],
        "τ2: H(X † Y)": [0, kv["necessary conditional entropy X -> Y"], 0],
    }
    ipts = np.array(list(intercepts.values()), float)
    t_traces.append(
        go.Scatter3d(
            x=ipts[:, 0],
            y=ipts[:, 1],
            z=ipts[:, 2] + _LIFT,
            mode="markers+text",
            text=[f"{name} = {max(v):.3f}" for name, v in intercepts.items()],
            textposition="top center",
            marker={"size": 6, "color": "black", "symbol": "diamond"},
            name="axis intercepts",
            hoverinfo="text",
        )
    )

    fig = go.Figure(t_traces)
    fig.update_layout(
        title="Region of tension",
        height=height,
        scene={
            "xaxis_title": "τ1 = I[Y:W|X]",
            "yaxis_title": "τ2 = I[X:W|Y]",
            "zaxis_title": "τ3 = I[X:Y|W]",
            "xaxis_range": [0, box[0] * 1.02 + 1e-9],
            "yaxis_range": [0, box[1] * 1.02 + 1e-9],
            "zaxis_range": [0, box[2] * 1.02 + 1e-9],
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": box[1] / max(box[0], 1e-9), "z": 0.7},
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 0.9}},
        },
        legend={"itemsizing": "constant", "orientation": "h", "x": 0.0, "y": 0.0, "yanchor": "top"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return fig
