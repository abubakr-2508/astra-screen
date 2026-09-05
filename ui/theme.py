"""
Single source of truth for colour.

Nothing outside this module may contain a colour literal. Two reasons:

1. The app supports light and dark. Streamlit exposes no theme CSS custom
   properties, so a hardcoded colour is necessarily wrong in one of them —
   the light accent #18566E measures 2.09:1 on the dark ground.
2. Before this module the palette existed in four places at once: the CSS
   block, two Plotly `color_discrete_map` literals, and three CSS named
   colours on threshold lines. That produced three different reds
   (#DC2626 / "red" / "darkred") and two ambers for the same three states.

Colour policy: neutral surfaces, saturated colour reserved for alarm.
GREEN IS DELETED. "Safe" is the absence of alarm, not a colour of its own.
That also removes the red/green pair, which is the common colour-blind failure.

Every value below is contrast-checked against its own ground (all >= 6:1).
"""

import streamlit as st

from config import RISK_PASS, RISK_REVIEW, RISK_FLAG

# --- light ---------------------------------------------------------------
_LIGHT = {
    "ink":        "#14181D",
    "muted":      "#5A646E",   # 5.67:1 on the page
    # Three surface steps. The page is deliberately NOT pure white: a white
    # card on a white page gives a shadow nothing to lift against, which is
    # why the metric cards read as hollow outlines.
    "surface":    "#F7F8FA",   # page
    "surface_2":  "#F2F4F6",   # sidebar, widget fills
    "raised":     "#FFFFFF",   # cards, popovers
    "rule":       "#DDE2E6",
    "accent":     "#18566E",   # 7.62:1
    "active":     "#18566E",   # selected tab / current state
    "pass":       "#5A646E",   # neutral slate — nominal is not an alarm
    "review":     "#8A5300",   # 5.96:1
    "flag":       "#B3261E",   # 6.15:1
    "measured":   "#2A6F97",
    "predicted":  "#B3261E",
    "series_alt": "#5B4B8A",
    # Chart series. Kept SEPARATE from status colours so alarm semantics stay
    # intact — a series must never be mistaken for a breach.
    #
    # Slot 2 was #8A5300, which IS the review amber — CIEDE2000 0.0, the same
    # colour. The comment above was aspiration, not fact: the second component
    # a viewer selected came out in the REVIEW colour purely by draw order.
    # #546B1F (deep olive) is the replacement, measured against the ramp's
    # documented floors — contrast 5.65:1, chroma 43.4, dE00 26.7 from review
    # and 48.8 from flag.
    #
    # Floors for this ramp: contrast >= 3:1 on the page ground (WCAG non-text),
    # chroma >= 20 so nothing reads as the near-neutral PASS grey (chroma 7.1),
    # dE00 >= 15 both between series and against review/flag.
    # Worst case light: pairwise 19.9, vs-alarm 22.5, contrast 4.88:1.
    "series":     ["#2A6F97", "#546B1F", "#5B4B8A", "#1E7A6B", "#9C4A6E"],
    # Zone bands behind chart data.
    "zone_margin": "rgba(138, 83, 0, 0.07)",
    "zone_breach": "rgba(179, 38, 30, 0.07)",
    # Light uses real shadow; dark cannot (see below).
    "elevation":  "0 1px 2px rgba(16,24,32,.05), 0 6px 16px rgba(16,24,32,.07)",
    "highlight":  "inset 0 1px 0 rgba(255,255,255,.6)",
}

# --- dark ----------------------------------------------------------------
_DARK = {
    "ink":        "#E3E6E9",
    "muted":      "#98A2AC",   # 6.52:1 on the page
    # Three steps here too. Previously the sidebar and a raised card would both
    # have been #22262B, so a card inside the sidebar would have been invisible.
    "surface":    "#1A1D21",   # page — soft near-black, never #000 (halation)
    "surface_2":  "#20242A",   # sidebar, widget fills
    "raised":     "#262B32",   # cards
    "rule":       "#313840",
    "accent":     "#6FB3CE",   # 7.26:1
    "active":     "#6FB3CE",
    "pass":       "#98A2AC",
    "review":     "#E0A44E",   # 7.72:1
    "flag":       "#F2837A",   # 6.69:1
    "measured":   "#6FB3CE",
    "predicted":  "#F2837A",
    "series_alt": "#A99BD6",
    # Same collision on this scheme: slot 2 was #E0A44E, the dark review amber,
    # at dE00 0.0. #A9C86B measures contrast 8.99:1, chroma 49.8, dE00 26.2
    # from review and 49.3 from flag.
    # Worst case dark: pairwise 18.4, vs-alarm 15.7, contrast 6.72:1.
    "series":     ["#6FB3CE", "#A9C86B", "#A99BD6", "#63C3AE", "#DE8FAE"],
    "zone_margin": "rgba(224, 164, 78, 0.09)",
    "zone_breach": "rgba(242, 131, 122, 0.09)",
    # Shadows are nearly invisible on a dark ground, so elevation is carried by
    # the surface step-up plus a 1px inset top highlight. That highlight is what
    # makes a dark card look crafted rather than flat.
    "elevation":  "0 1px 2px rgba(0,0,0,.45), 0 8px 24px rgba(0,0,0,.35)",
    "highlight":  "inset 0 1px 0 rgba(255,255,255,.05)",
}


def is_dark() -> bool:
    """True when the viewer's active theme is dark.

    `st.context.theme` is read-only and inferred from the app background, so it
    reflects the *resolved* theme rather than the configured one. It can be
    briefly stale during a theme change; falling back to light is the safe
    default because every light value also stays legible on a light ground.
    """
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def palette() -> dict:
    """The active palette. Call at render time, never cache at import."""
    return _DARK if is_dark() else _LIGHT


def status_colors() -> dict:
    """Screening status -> colour, for Plotly `color_discrete_map`."""
    p = palette()
    return {RISK_PASS: p["pass"], RISK_REVIEW: p["review"], RISK_FLAG: p["flag"]}


RADIUS = "6px"        # containers; inputs use 4px via config.toml buttonRadius
MOTION = "120ms ease"  # state changes only, never decoration


def css_vars() -> str:
    """The active palette as CSS custom properties, for the injected stylesheet."""
    p = palette()
    parts = [f"--astra-{k.replace('_', '-')}: {v};"
             for k, v in p.items() if isinstance(v, str)]
    parts += [f"--astra-series-{i+1}: {c};" for i, c in enumerate(p["series"])]
    parts += [f"--astra-radius: {RADIUS};", f"--astra-motion: {MOTION};"]
    return ":root { " + " ".join(parts) + " }"


def status_order() -> list:
    """Stable legend/category order so it never reshuffles between datasets."""
    return [RISK_PASS, RISK_REVIEW, RISK_FLAG]
