"""
One Plotly visual system for the whole app.

Before this, the three figures were three different apps: Plotly Express
defaults for two of them, a partial `update_layout` on the third, so type
sizes, legend positions and margins all disagreed between tabs.

The template is built from the ACTIVE palette, so charts follow the viewer's
light/dark theme like everything else. Register once per run, then pass
`theme=None` to `st.plotly_chart` so Streamlit does not overwrite it.

Two rules govern colour here:

  1. Series colours come from `palette()["series"]` and NEVER from the status
     colours. Drawing the second selected component in amber and the third in
     red made draw order look like a verdict.
  2. A filled region always means a threshold has meaning there. Zones come
     from `zone_margin` / `zone_breach`; nothing is filled for looks.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from ui.theme import palette

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
MONO = 'ui-monospace, "Cascadia Mono", Consolas, "SF Mono", monospace'

TEMPLATE_NAME = "astra"

# Marker shape carries status as well as colour, so the state survives
# greyscale print and colour-blind viewing — same rule as the results table.
SYMBOL = {"PASS": "circle", "REVIEW": "diamond", "FLAG (HIGH RISK)": "x"}


def _rgba(hex_colour: str, alpha: float) -> str:
    """#RRGGBB -> rgba() at the given alpha."""
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _pill(colour: str) -> dict:
    """Annotation styling: the label sits on its own ground, not on the data.

    Reference-line labels previously rendered as bare text directly over
    gridlines and points, which is the single most unfinished-looking thing a
    chart can do. A page-coloured plate with padding separates them.
    """
    p = palette()
    return dict(font=dict(family=MONO, size=10, color=colour),
                bgcolor=p["surface"], borderpad=3)


def register() -> str:
    """Build and register the template against the current theme. Idempotent."""
    p = palette()
    grid = _rgba(p["muted"], 0.16)

    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, size=12, color=p["ink"]),
        paper_bgcolor="rgba(0,0,0,0)",     # inherit the page, both themes
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=p["series"],              # series ramp, never status colours
        title=dict(font=dict(size=13, color=p["ink"]), x=0, xanchor="left"),
        margin=dict(l=64, r=28, t=44, b=52),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=grid, linewidth=1,
                   ticks="outside", ticklen=4, tickcolor=grid,
                   tickfont=dict(family=MONO, size=11, color=p["muted"]),
                   title=dict(font=dict(size=12, color=p["muted"]))),
        # Dotted, not solid: the grid is a reading aid, not content.
        yaxis=dict(showgrid=True, gridcolor=grid, griddash="dot", zeroline=False,
                   showline=False,
                   tickfont=dict(family=MONO, size=11, color=p["muted"]),
                   title=dict(font=dict(size=12, color=p["muted"]))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=p["muted"])),
        # Themed hover. Plotly's default is an unstyled white box that looks
        # foreign on the dark scheme.
        hoverlabel=dict(font=dict(family=MONO, size=11, color=p["ink"]),
                        bgcolor=p["raised"], bordercolor=p["rule"]),
        hovermode="closest",
    )
    pio.templates[TEMPLATE_NAME] = t
    pio.templates.default = TEMPLATE_NAME
    return TEMPLATE_NAME


def lot_distribution(df, value_col="Value_24h", unit=""):
    """One box per LOT over the full lot population, with components overlaid.

    The previous chart passed `color="Screening_Status"` to px.box, which split
    each lot into a separate box per outcome — 11 boxes for 5 lots, one of them
    drawn from a single data point. That destroys the very baseline the box is
    meant to establish (the lot median/MAD the Z-scores are computed against)
    and reads as "most parts are failing" when 167 of 200 pass.

    Here the box IS the lot population; colour annotates points on top of it.
    The median is drawn as an explicit accent rule, because the caption claims
    the box shows the baseline and a hairline box does not actually show it.
    """
    p = palette()
    lots = sorted(df["Lot_ID"].unique())
    fig = go.Figure()

    for i, lot in enumerate(lots):
        ys = df.loc[df["Lot_ID"] == lot, value_col]
        # Quartiles and fences are precomputed rather than left to Plotly.
        # With `boxpoints=False` Plotly runs the whiskers out to the data
        # min/max, so the anomalies themselves stretched the whisker to 42 uA
        # and the box appeared to CONTAIN the parts it is supposed to be the
        # baseline for. Explicit 1.5 x IQR fences keep the box to the nominal
        # population, leaving the flagged parts visibly outside it.
        q1, med, q3 = (float(v) for v in np.percentile(ys, [25, 50, 75]))
        iqr = q3 - q1
        lo_f = max(float(ys.min()), q1 - 1.5 * iqr)
        hi_f = min(float(ys.max()), q3 + 1.5 * iqr)
        fig.add_trace(go.Box(
            x=[i], q1=[q1], median=[med], q3=[q3],
            lowerfence=[lo_f], upperfence=[hi_f],
            width=0.55, boxpoints=False, showlegend=False,
            fillcolor=_rgba(p["muted"], 0.16),
            line=dict(color=p["muted"], width=1.25), hoverinfo="skip",
        ))
        # Median as a shape, not a trace: it must not enter the legend.
        fig.add_shape(type="line", x0=i - 0.275, x1=i + 0.275, y0=med, y1=med,
                      line=dict(color=p["accent"], width=2), layer="above")

    rng = np.random.default_rng(0)          # fixed: jitter must not move between reruns
    from ui.theme import status_colors, status_order
    colours = status_colors()
    for status in status_order():
        sub = df[df["Screening_Status"] == status]
        if sub.empty:
            continue
        jit = rng.uniform(-0.18, 0.18, len(sub))
        nominal = status == "PASS"
        fig.add_trace(go.Scatter(
            x=[lots.index(l) + j for l, j in zip(sub["Lot_ID"], jit)],
            y=sub[value_col], mode="markers", name=status,
            marker=dict(size=6 if nominal else 9, color=colours[status],
                        symbol=SYMBOL.get(status, "circle"),
                        opacity=0.45 if nominal else 0.95,
                        # Page-coloured stroke separates overlapping points.
                        # This was rgba(0,0,0,0) — i.e. no separation at all.
                        line=dict(width=1, color=p["surface"])),
            customdata=np.stack([sub["Component_ID"], sub["Value_24h_Robust_Z"],
                                 sub["Value_24h_Lot_Median"]], axis=-1),
            hovertemplate=("<b>%{customdata[0]}</b><br>"
                           f"24 h: %{{y:.2f}} {unit}<br>"
                           f"lot median: %{{customdata[2]:.2f}} {unit}<br>"
                           "peer Z: %{customdata[1]:+.2f}<extra></extra>"),
        ))

    fig.update_xaxes(tickmode="array", tickvals=list(range(len(lots))),
                     ticktext=lots, title_text="Manufacturing lot")
    fig.update_yaxes(title_text=f"24 h measurement{f' ({unit})' if unit else ''}")
    fig.update_layout(height=430, boxmode="overlay")
    return fig


def failure_mode_quadrant(df, z_threshold, unit=""):
    """Baseline deviation vs drift rate — 'born bad' against 'going bad'.

    Replaces a scatter of Value_24h against its own robust Z, which measured
    r = 1.000000 within every lot: the Z is an affine transform of the value
    inside a lot, so that chart plotted a quantity against itself.

    These two axes are genuinely orthogonal on this data — peer anomalies are
    pure x-signal (0h Z 18.8-33.8, drift indistinguishable from normal), drift
    defects are pure y-signal (0h Z -1.4-3.0, drift 0.255-0.300). It is the one
    chart that shows WHY there are two detection modules.

    Only the z-threshold rule is drawn, because only it exists in the engine.
    No horizontal drift boundary is invented, and the rule is a LINE, not a
    shaded zone: the flagged side runs from Z = 3 to Z = 34, so a band covering
    it filled about 70% of the canvas in alarm colour while nearly all of that
    area was empty. The zone treatment earns its place on the trajectory chart,
    where the regions are narrow and the data crosses them; here it inverted
    figure and ground.
    """
    p = palette()
    from ui.theme import status_colors, status_order
    colours = status_colors()
    fig = go.Figure()

    for status in status_order():
        sub = df[df["Screening_Status"] == status]
        if sub.empty:
            continue
        nominal = status == "PASS"
        fig.add_trace(go.Scatter(
            x=sub["Value_0h_Robust_Z"], y=sub["Predicted_Drift_Rate_per_hr"],
            mode="markers", name=status,
            marker=dict(size=6 if nominal else 10, color=colours[status],
                        symbol=SYMBOL.get(status, "circle"),
                        opacity=0.45 if nominal else 0.95,
                        line=dict(width=1, color=p["surface"])),
            customdata=np.stack([sub["Component_ID"], sub["Lot_ID"]], axis=-1),
            hovertemplate=("<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                           "0 h peer Z: %{x:+.2f}<br>"
                           f"drift: %{{y:.3f}} {unit}/h<extra></extra>"),
        ))

    fig.add_vline(x=z_threshold, line_dash="dot", line_width=1, line_color=p["muted"],
                  annotation_text=f"|Z| = {z_threshold:.1f}",
                  annotation_font=dict(family=MONO, size=10, color=p["muted"]),
                  annotation_bgcolor=p["surface"], annotation_borderpad=3)
    # Each label sits beside the cluster it names — right of the top-left group,
    # above the bottom-right one. Parking both in the empty middle band (with
    # arrows pointing outward) left them describing nothing in particular; a
    # label adjacent to its own cluster needs no arrow at all.
    for x, y, txt in [(0.23, 0.94, "GOING BAD · normal baseline, fast drift"),
                      (0.52, 0.20, "BORN BAD · off-baseline, normal drift")]:
        fig.add_annotation(x=x, y=y, xref="paper", yref="paper", showarrow=False,
                           xanchor="left", text=txt, **_pill(p["muted"]))

    fig.update_xaxes(title_text="Peer deviation at 0 h (robust Z vs lot)",
                     # x = 0 is "exactly at the lot median" — a real landmark.
                     zeroline=True, zerolinecolor=_rgba(p["muted"], 0.30),
                     zerolinewidth=1)
    fig.update_yaxes(title_text=f"Predicted drift rate{f' ({unit}/h)' if unit else ''}")
    fig.update_layout(height=430)
    return fig


def trajectories(df, component_ids, limit, safety_ratio, unit=""):
    """Measured trajectory plus predicted 168 h, one legend entry per component.

    Previously each component produced two independently-named traces —
    "C1052 (Actual)" and "C1052 (Predicted 168h)" — so five components filled
    the legend with ten entries and consumed roughly half the figure width.
    legendgroup ties the pair together; solid is measured, dashed is projected.

    The safety margin and datasheet limit are drawn as ZONES, not as two thin
    dotted lines. A viewer should not have to infer "between these two rules
    is the margin band": clear below, margin between, breach above.
    """
    p = palette()
    fig = go.Figure()
    # Series ramp only. This previously read [accent, review, flag, ...], so
    # the second and third selected components came out in the amber and red
    # reserved for REVIEW and FLAG — draw order looked like a verdict.
    series = p["series"]

    plotted, ys_all = [], []
    for i, cid in enumerate(component_ids):
        match = df[df["Component_ID"] == cid]
        if match.empty:
            continue
        row = match.iloc[0]

        xs, ys = [0, 24], [row["Value_0h"], row["Value_24h"]]
        for h, col in ((96, "Value_96h"), (168, "Value_168h")):
            if col in row.index and pd.notna(row.get(col)):
                xs.append(h)
                ys.append(row[col])
        plotted.append((cid, row, series[i % len(series)], xs, ys))
        ys_all += [float(v) for v in ys] + [float(row["Predicted_Value_168h"])]

    # Zones are drawn first and sit below the data. The y-axis is pinned so the
    # breach band reaches the top of the plot instead of stopping at the data.
    margin_y = limit * safety_ratio
    lo = min(ys_all + [margin_y]) if ys_all else 0.0
    hi = max(ys_all + [limit]) if ys_all else limit
    pad = (hi - lo) * 0.08 or 1.0
    ylo, yhi = lo - pad, hi + pad
    fig.add_hrect(y0=margin_y, y1=limit, fillcolor=p["zone_margin"],
                  line_width=0, layer="below")
    fig.add_hrect(y0=limit, y1=yhi, fillcolor=p["zone_breach"],
                  line_width=0, layer="below")

    for cid, row, colour, xs, ys in plotted:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=str(cid),
            legendgroup=str(cid), line=dict(width=2, color=colour),
            marker=dict(size=6, color=colour,
                        line=dict(width=1, color=p["surface"])),
            hovertemplate=f"<b>{cid}</b> measured<br>%{{x}} h: %{{y:.2f}} {unit}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[24, 168], y=[row["Value_24h"], row["Predicted_Value_168h"]],
            mode="lines+markers", name=str(cid), legendgroup=str(cid),
            showlegend=False,                      # shares the entry above
            line=dict(width=2, dash="dash", color=colour),
            marker=dict(size=6, symbol="diamond-open", color=colour),
            hovertemplate=f"<b>{cid}</b> projected<br>168 h: %{{y:.2f}} {unit}<extra></extra>",
        ))

    # The 168 h endpoint IS the decision. Drawn last so it sits on top, ringed
    # in the page colour so it reads as a terminus rather than another sample.
    for cid, row, colour, _xs, _ys in plotted:
        fig.add_trace(go.Scatter(
            x=[168], y=[float(row["Predicted_Value_168h"])], mode="markers",
            legendgroup=str(cid), showlegend=False, hoverinfo="skip",
            marker=dict(size=13, symbol="diamond", color=colour,
                        line=dict(width=2.5, color=p["surface"])),
        ))

    # Labels derived from the data, not a literal. The previous annotation read
    # "Datasheet Limit (50 µA)" regardless of what the file actually contained.
    fig.add_hline(y=limit, line_color=p["flag"], line_dash="dot", line_width=1.5,
                  annotation_text=f"DATASHEET LIMIT · {limit:g} {unit}".strip(),
                  annotation_position="top left",
                  annotation_font=dict(family=MONO, size=10, color=p["flag"]),
                  annotation_bgcolor=p["surface"], annotation_borderpad=3)
    fig.add_hline(y=margin_y, line_color=p["review"], line_dash="dot", line_width=1,
                  annotation_text=f"SAFETY MARGIN {safety_ratio:.0%} · {margin_y:g} {unit}".strip(),
                  annotation_position="bottom left",
                  annotation_font=dict(family=MONO, size=10, color=p["review"]),
                  annotation_bgcolor=p["surface"], annotation_borderpad=3)

    fig.update_xaxes(tickmode="array", tickvals=[0, 24, 96, 168],
                     ticktext=["0 h", "24 h", "96 h", "168 h"],
                     title_text="Burn-in hours")
    fig.update_yaxes(title_text=f"Parameter value{f' ({unit})' if unit else ''}",
                     range=[ylo, yhi])
    fig.update_layout(height=460, hovermode="x unified")
    return fig
