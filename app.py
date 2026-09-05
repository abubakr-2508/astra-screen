"""
AI-Driven Anomaly Detection in Component Burn-In & Screening
Interactive QA Inspector Dashboard (Streamlit Web App)
ISRO Problem Statement ID: 26170
"""

import datetime

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from config import (BASE_DIR, DATA_DIR, ROBUST_Z_SCORE_THRESHOLD,
                    SAFETY_SLOPE_MARGIN_RATIO, RISK_PASS, RISK_REVIEW, RISK_FLAG)
from data.synthetic_generator import generate_synthetic_burnin_data
from pipeline import run_full_screening_pipeline
from models.drift_predictor import DriftPredictor168h
from engine.audit_logger import get_audit_history, log_screening_run
from ui.theme import palette, status_colors, status_order, is_dark, css_vars
from ui import viz
from ui.brand import lockup

# Page Configuration — must be the first Streamlit command in the script.
st.set_page_config(
    page_title="Astra Screen",
    page_icon=str(BASE_DIR / "assets" / "favicon.png"),
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palette resolved per run against the viewer's active theme, AFTER
# set_page_config (reading st.context counts as a Streamlit command).
PAL = palette()
STATUS_COLORS = status_colors()
STATUS_ORDER = status_order()
viz.register()   # one Plotly visual system, built from the active palette

# Modern Custom CSS Styling
st.markdown("""
<style>
    /* --- Streamlit chrome ---
       Hide only the Deploy button and the hamburger menu. Do NOT hide
       [data-testid="stToolbar"] or the header: stExpandSidebarButton is a
       child of the toolbar and is the only control that re-opens a collapsed
       sidebar. Zeroing the header height is equally fatal — it is display:flex,
       so its children collapse to 0x0 while still reporting visibility:visible.
       The header is position:absolute, so its natural 60px costs no layout
       space and can simply stay. (`footer` and `stDecoration` do not exist in
       Streamlit 1.63; the usual hide-the-footer rules are omitted on purpose.)
       The 96px container padding was the real cost of the fold — that is
       reduced below, reclaiming ~156px on a 768px projector. --- */
    [data-testid="stHeader"] {
        background: transparent;
        pointer-events: none;      /* don't swallow clicks meant for content */
    }
    /* Hide the two actual eyesores ONLY. Do NOT hide [data-testid="stToolbar"]:
       stExpandSidebarButton is a child of it, and that button is the only way
       to re-open a collapsed sidebar. Hiding the toolbar strands the user. */
    [data-testid="stAppDeployButton"],
    [data-testid="stMainMenu"] { display: none; }
    [data-testid="stExpandSidebarButton"] { pointer-events: auto; }
    .stMainBlockContainer {
        padding-top: 1.75rem;
        padding-bottom: 3rem;
        max-width: 1700px;
        margin: 0 auto;
    }

    /* No `color` here on purpose. Streamlit exposes no theme CSS variables, so a
       hardcoded colour is wrong in one of the two schemes: #1E293B scored 1.16:1
       on the dark ground and #64748B 3.55:1. Inheriting `textColor` from the theme
       makes both schemes correct automatically. */
    /* --- header lockup: mark + title + subtitle as one aligned unit --- */
    .lockup-head {
        display: flex;
        align-items: center;
        gap: 0.95rem;
        margin-bottom: 0.9rem;
    }
    .lockup-head .mark { flex: none; display: block; border-radius: var(--astra-radius); }
    .lockup-txt { min-width: 0; }

    /* A wordmark, not a heading. At 2.2rem/-0.022em this was the same species
       as the section h3s (1.75rem) and only 1.26x their size, which is why it
       read as "the biggest heading" instead of "the application". Tighter
       tracking is what makes type read as a mark; the h3s drop further below,
       taking the ratio to 1.63x. */
    .main-header {
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: -0.030em;
        line-height: 1.1;
        margin-bottom: 0.15rem;
    }

    /* Section headings were h3 at 1.75rem — near enough to the product name to
       compete with it. */
    [data-testid="stHeading"] h3 { font-size: 1.35rem; letter-spacing: -0.01em; }

    /* --- small uppercase label: gives real hierarchy under an st.subheader.
           Replaces four "### <emoji> Heading" blocks that rendered as h3 and
           were visually indistinguishable from the h2 above them. --- */
    .card-label {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 0.75rem;
        font-family: ui-monospace, "Cascadia Mono", Consolas, "SF Mono", monospace;
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        opacity: 0.60;
        margin: 0.1rem 0 0.45rem;
    }

    /* --- tab bar as an instrument mode selector.
           Targeted via the ARIA role rather than data-baseweb: Streamlit 1.63
           dropped the baseweb attribute on tabs, and role="tab" is a web
           standard, so it survives framework churn.

           MEASURED PROBLEM: the hierarchy was inverted. The selected tab sat at
           7.26:1 (accent) while the four unselected ones sat at 13.50:1 (full
           ink) — the active tab was the QUIETEST thing in the bar, which is why
           it hid. Fixed by giving active the strongest text plus the only
           colour, and dropping inactive to muted. --- */
    [role="tab"] p {
        font-family: ui-monospace, "Cascadia Mono", Consolas, "SF Mono", monospace;
        font-size: 0.8125rem;          /* 13px; labels shortened to pay for it */
        font-weight: 500;
        letter-spacing: 0.08em;
        color: var(--astra-muted);
        transition: color var(--astra-motion);
    }
    [role="tab"]:hover p { color: var(--astra-ink); }
    [role="tab"][aria-selected="true"] p {
        color: var(--astra-ink);
        font-weight: 700;
    }
    /* MEASURED: the tablist had `border-bottom: 0px none` and every tab had
       `padding: 0`, so the active indicator floated in empty space and the five
       labels were words in a row, not tabs. A tab reads as a tab because its
       indicator BREAKS a rule that runs the width of the bar — so the bar gets
       the rule, and the tab is pulled down 1px to sit on it. */
    [role="tablist"] {
        border-bottom: 1px solid var(--astra-rule);
        gap: 2px;
    }
    [role="tab"] {
        padding: 0 14px;
        margin-bottom: -1px;
        border-radius: var(--astra-radius) var(--astra-radius) 0 0;
        transition: background var(--astra-motion);
    }
    [role="tab"]:hover { background: var(--astra-surface-2); }
    /* The accent lives in the underline: strongest text + the only colour in
       the bar both land on the selected tab. */
    [role="tab"][aria-selected="true"] {
        background: var(--astra-raised);
        box-shadow: inset 0 -2px 0 0 var(--astra-accent);
    }
    [data-testid="stTabPanel"] { padding-top: 0.85rem; }
    .sub-header {
        font-size: 0.9rem;
        opacity: 0.62;
        max-width: 78ch;
        margin-bottom: 0;
    }

    /* --- provenance strip: run stamp, dataset, N, active thresholds.
           No colour literals — currentColor + opacity keeps it correct in
           both schemes. --- */
    .provenance {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
        font-family: ui-monospace, "Cascadia Mono", Consolas, "SF Mono", monospace;
        font-size: 0.66rem;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        opacity: 0.60;
        padding: 7px 0 9px;
        /* Only a bottom rule. With a border above as well the strip read as a
           separate band floating under the title; with one rule beneath, the
           lockup and the run state close as a single masthead. */
        border-bottom: 1px solid currentColor;
        border-color: color-mix(in srgb, currentColor 16%, transparent);
        margin-bottom: 1.35rem;
    }
    .provenance .hint { opacity: 0.65; letter-spacing: 0.04em; }
    .provenance .unit { text-transform: none; }

    /* Closing note. Replaces a centred repeat of the product name, which the
       masthead subtitle already ends with. */
    .footer-note {
        margin-top: 2.2rem;
        padding-top: 0.85rem;
        border-top: 1px solid currentColor;
        border-color: color-mix(in srgb, currentColor 12%, transparent);
        font-family: ui-monospace, "Cascadia Mono", Consolas, "SF Mono", monospace;
        font-size: 0.66rem;
        letter-spacing: 0.06em;
        /* Measured at opacity .55 this came out 3.90:1 — a real AA failure for
           text this size. The muted token is the app's established secondary
           tier and lands at 5.67:1 / 5.49:1, so it is used here rather than a
           hand-picked opacity. */
        color: var(--astra-muted);
    }

    /* --- sidebar: tighter rhythm. 223px of the 689px sidebar was Streamlit's
           default block gap — nearly a third of the column spent on air. --- */
    [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { gap: 0.5rem; }
    /* Once sidebar content reaches the column height, Streamlit's flex layout
       SHRINKS items to fit rather than scrolling — a wrapper ended up 110px
       around 126px of content, so the lot rows overlapped the next label.
       Refusing to shrink makes it scroll instead, which is the correct
       behaviour: never squash content into other content. */
    [data-testid="stSidebarUserContent"] [data-testid="stElementContainer"],
    [data-testid="stSidebarUserContent"] [data-testid="stMarkdown"] {
        flex-shrink: 0;
        min-height: fit-content;
    }

    /* --- per-lot flag rate --- */
    .lots { display: flex; flex-direction: column; gap: 3px; }
    .lotrow { display: flex; align-items: center; gap: 8px;
              font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
              font-size: 0.68rem; }
    .lotrow .lot { opacity: 0.72; min-width: 3.4em; letter-spacing: 0.04em; }
    .lotrow .bar { flex: 1; height: 5px; border-radius: 3px;
                   background: color-mix(in srgb, currentColor 12%, transparent);
                   overflow: hidden; }
    .lotrow .bar i { display: block; height: 100%; background: var(--astra-flag); }
    .lotrow .num { min-width: 3.4em; text-align: right; font-variant-numeric: tabular-nums; }
    .lotrow .num .of { opacity: 0.45; }
    /* Secondary text inside a card-label, so it cannot become a trailing
       element that overflows its own markdown wrapper. */
    .card-label .lbl-hint {
        font-weight: 400;
        letter-spacing: 0.04em;
        text-transform: none;
        opacity: 0.72;
    }

    /* --- decision legend: glyph + colour + meaning --- */
    .card-label.ctx-gap { margin-top: 0.9rem; }
    .lgd-list { padding-bottom: 10px; }
    .lgd { display: flex; align-items: flex-start; gap: 8px; margin-bottom: 3px; }
    .lgd .g { font-family: ui-monospace, Consolas, monospace; font-weight: 700;
              font-size: 0.78rem; width: 1.1em; text-align: center; flex: none; }
    .lgd .t { font-size: 0.66rem; line-height: 1.32; opacity: 0.8; }
    .lgd .t b { display: block; font-size: 0.62rem; letter-spacing: 0.09em;
                text-transform: uppercase; opacity: 0.95; }
    .lgd-flag   .g, .lgd-flag   .t b { color: var(--astra-flag); }
    .lgd-review .g, .lgd-review .t b { color: var(--astra-review); }
    .lgd-pass   .g, .lgd-pass   .t b { color: var(--astra-muted); }
    .provenance .sys { font-weight: 700; letter-spacing: 0.14em; }
    .provenance .sep {
        width: 1px; height: 10px; display: inline-block;
        background: currentColor; opacity: 0.30;
    }

    /* --- the verdict: one dominant number, one sentence, one breakdown --- */
    .verdict {
        display: flex;
        align-items: flex-start;
        gap: 22px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 1.1rem;
    }
    .vnum {
        font-family: ui-monospace, "Cascadia Mono", Consolas, "SF Mono", monospace;
        font-size: 3.4rem;
        font-weight: 600;
        line-height: 0.95;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.02em;
        color: var(--astra-flag);
    }
    .vnum.calm { color: inherit; opacity: 0.55; }   /* zero is never an alarm */
    .vtxt { font-size: 1.05rem; line-height: 1.35; padding-top: 0.35rem; }
    .vsub .unit { text-transform: none; }
    /* These four counts are the ONLY things that move when the operator
       changes a threshold, and they were the quietest marks on the card:
       0.66rem at opacity .62 measured 4.98:1 (light) / 5.35:1 (dark) against
       the card, below the headline's own 6.54:1 at five times the size. The
       app was responding and the response was styled into a footnote.

       Colour, not opacity, carries the hierarchy now — `opacity` on this
       element would cap every child, so raising the numerals inside it is
       impossible while it is set. Labels sit at muted (6.03:1 / 5.49:1),
       the numerals at full ink (17.82:1 / 11.37:1). Both theme-aware. */
    .vsub {
        font-family: ui-monospace, "Cascadia Mono", Consolas, "SF Mono", monospace;
        text-transform: uppercase;
        font-size: 0.66rem;
        letter-spacing: 0.08em;
        color: var(--astra-muted);
        margin-top: 0.5rem;
        line-height: 1.7;
    }
    .vsub .n {
        color: var(--astra-ink);
        font-weight: 700;
        font-size: 0.78rem;
        font-variant-numeric: tabular-nums;
    }
    /* --- spec-sheet rows: label left, value right in mono with tabular
           figures so digits align down the column. Replaces ragged
           st.write("**Label**: value") pairs. --- */
    .spec {
        position: relative;
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 1.2rem;
        padding: 6px 0;
    }
    /* The divider fades toward the value column instead of ruling the row
       edge to edge. Absolutely positioned rather than a border-image because
       `.spec` is a flex container — an in-flow pseudo-element would become a
       flex item and land between the label and its value. */
    .spec::after {
        content: "";
        position: absolute;
        left: 0; right: 0; bottom: 0;
        height: 1px;
        background: linear-gradient(to right,
                    color-mix(in srgb, currentColor 16%, transparent),
                    transparent);
    }
    .spec .k {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        opacity: 0.58;
        white-space: nowrap;
    }
    .spec .v {
        font-family: ui-monospace, "Cascadia Mono", Consolas, "SF Mono", monospace;
        font-variant-numeric: tabular-nums;
        font-size: 0.86rem;
        text-align: right;
    }
    .spec .v .u {
        opacity: 0.50;
        font-size: 0.72rem;
        margin-left: 0.35em;
    }

    /* --- evidence cards, ruled by severity. Every reason used to render as
           an identical blue st.info regardless of what it said. --- */
    .ev {
        border-left: 3px solid currentColor;
        padding: 0.6rem 0.85rem;
        margin-bottom: 0.55rem;
        background: color-mix(in srgb, currentColor 4%, transparent);
    }
    .ev-h {
        font-family: ui-monospace, "Cascadia Mono", Consolas, "SF Mono", monospace;
        font-size: 0.64rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .ev-b { font-size: 0.88rem; line-height: 1.5; opacity: 0.88; }
    .ev-crit { border-left-color: var(--astra-flag); }
    .ev-crit .ev-h { color: var(--astra-flag); }
    .ev-warn { border-left-color: var(--astra-review); }
    .ev-warn .ev-h { color: var(--astra-review); }
    .ev-ok   { border-left-color: var(--astra-muted); }
    .ev-ok   .ev-h { color: var(--astra-muted); }

    /* NOTE: .metric-card, .status-pass, .status-review and .status-flag were
       defined here and applied to no element — 28 lines of dead style that
       advertised a status-chip system the running app never had. Their six
       colour literals had been separately re-typed into the Plotly maps
       below, which is how the app ended up with three different reds.
       Colour now lives only in ui/theme.py. */
</style>
""", unsafe_allow_html=True)

# The block above is a plain string, so palette values are injected separately
# as CSS custom properties rather than escaping every brace in an f-string.
# The full token set as CSS custom properties, so every later rule can use
# them instead of inventing its own values.
st.markdown(f"<style>{css_vars()}</style>", unsafe_allow_html=True)

# First consumer of the tokens, and the most visible complaint: st.metric had
# background rgba(0,0,0,0) — a 1px outline on the page, with no fill, which is
# exactly why the cards read as hollow rather than raised.
st.markdown("""
<style>
    /* `.card` styles exactly one element, the verdict card, and that card is a
       READOUT — nothing about it is clickable. It previously also carried a
       `transition: box-shadow, border-color` with no `:hover` rule anywhere to
       fire it, and a `[data-testid="stMetric"]` selector that matches nothing
       (st.metric is never called). Both are gone: hover feedback on a display
       element is decoration impersonating an affordance, which is the exact
       thing this UI is being stripped of. State goes on the buttons below. */
    .card {
        background: var(--astra-raised);
        border: 1px solid var(--astra-rule);
        border-radius: var(--astra-radius);
        box-shadow: var(--astra-elevation), var(--astra-highlight);
        padding: 1rem 1.15rem;
    }

    /* Buttons are the only surface here a cursor can act on, so they are the
       only place a hover state belongs. Measured before this: background
       #252A31, border #313840, box-shadow none, and `transition: all` with no
       duration — every change instantaneous.

       The border carries the signal, not the background: rest and raised
       differ by #252A31 vs #262B32, which is invisible. Accent against the
       sidebar ground measures ~7.4:1 light and ~7:1 dark, well past the 3:1
       floor for a graphical object. On press the lift collapses, which is
       what makes a button feel like a button. */
    /* `transform` is deliberately NOT in this transition list. Measured with
       the press sampled frame by frame: with a 120ms ramp the button had not
       moved at all at mousedown, so a normal quick click never showed the
       travel — a press has to answer instantly or it reads as lag. Hover and
       release still ease; the 1px depression does not. */
    [data-testid="stBaseButton-secondary"] {
        transition: border-color var(--astra-motion),
                    box-shadow var(--astra-motion);
    }
    [data-testid="stBaseButton-secondary"]:hover {
        border-color: var(--astra-accent);
        box-shadow: var(--astra-elevation);
    }
    [data-testid="stBaseButton-secondary"]:active {
        box-shadow: none;
        transform: translateY(1px);
    }

    /* st.caption renders at FULL INK in 1.63 — 16.77:1 light, 13.50:1 dark,
       byte-identical to body text — so five sections of explanatory text
       carried the same weight as the content they are subordinate to.
       Nothing in Streamlit's own stylesheet sets that colour, so one rule
       cascades cleanly to the inner <p>. Lands at 6.03:1 / 5.49:1. */
    [data-testid="stCaptionContainer"] { color: var(--astra-muted); }

    /* The download / fullscreen / search cluster sits at opacity 0 with
       `transition: none`, so it snaps into existence the instant the cursor
       crosses a chart or table edge. */
    [data-testid="stElementToolbar"] { transition: opacity var(--astra-motion); }

    /* ONE label system. Eight widget labels rendered in Source Sans 14px
       sentence case ("Status", "Lot", "Search component", "Peer anomaly
       threshold |Z|") while every label we authored is mono uppercase at
       0.70rem — two typographic voices on the same screen, which is most of
       why it still read as a document with widgets dropped into it. Measured
       after: "PEER ANOMALY THRESHOLD |Z|" stays on one 24px line, so the
       sidebar height is unchanged. */
    [data-testid="stWidgetLabel"] p {
        font-family: ui-monospace, "Cascadia Mono", Consolas, "SF Mono", monospace;
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        opacity: 0.60;
    }

    /* The sidebar opened with 78px of nothing above DATASET. stSidebarHeader is
       a 260x60 band that already exists there, holding only a logo spacer and
       the collapse button — so the title costs ZERO content height, which the
       sidebar cannot spare (683px of content in 720px on a small laptop).
       `margin-right: auto` keeps the collapse button at the right edge. */
    [data-testid="stSidebarHeader"] { display: flex; align-items: center; }
    [data-testid="stSidebarHeader"]::before {
        content: "RUN CONTROL";
        font-family: ui-monospace, "Cascadia Mono", Consolas, "SF Mono", monospace;
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        opacity: 0.60;
        margin-right: auto;
    }

    /* The theme toggle is chrome, not a form control, so it loses the button
       shell at rest and only shows a surface when the cursor is on it. Scoped
       by the `st-key-` class the frontend emits for keyed widgets, so the other
       three buttons keep their normal treatment. */
    .st-key-theme_toggle [data-testid="stBaseButton-secondary"] {
        background: transparent;
        border-color: transparent;
    }
    .st-key-theme_toggle [data-testid="stBaseButton-secondary"]:hover {
        background: var(--astra-surface-2);
        border-color: var(--astra-rule);
        box-shadow: none;
    }

    /* Sidebar section rules, drawn with ZERO added height. The sidebar holds
       683px of content in 720px on a small laptop — 37px of slack — and four
       labels with 8px of breathing room each would spend 32px of it and
       re-open the overlap fixed in Step 14. The hairline is positioned inside
       the gap that already exists (1.6px label margin + 8px flex gap). It
       fades out to the right so it reads as a section break, not a table rule.
       Not applied to the first label, which needs no rule above it. */
    [data-testid="stSidebar"] .card-label.sec { position: relative; }
    [data-testid="stSidebar"] .card-label.sec::before {
        content: "";
        position: absolute;
        left: 0; right: 0; top: -5px;
        height: 1px;
        background: linear-gradient(to right,
                    color-mix(in srgb, currentColor 22%, transparent),
                    transparent);
    }
</style>
""", unsafe_allow_html=True)

# --- Theme state -----------------------------------------------------------
# Lifted above the header because the toggle now lives IN the header: a
# masthead needs something on its right, or it stays a block of text.
# Streamlit has no theme-setting API, but it persists the active choice in
# localStorage under 'stActiveTheme-/-v2' ("System" | "Light" | "Dark") and
# honours it on load. Writing that key drives Streamlit's REAL theme system,
# so charts and widgets re-theme correctly — unlike a CSS overlay, which
# cannot reach Plotly's rendered backgrounds.
#
# st.markdown strips <script>, so the write happens inside a zero-height
# components iframe, which is same-origin and can reach window.parent.
#
# Streamlit 1.63 deprecates components.v1.html in favour of st.iframe, but
# st.iframe takes a src URL, not raw HTML. A data: URI would be an OPAQUE
# origin, so window.parent.localStorage would be blocked and the mechanism
# would break. components.v1.html is therefore the correct call here, and it
# only runs on an actual toggle — not on every rerun — so the deprecation
# notice appears once per theme switch rather than filling the terminal.
_THEME_KEY = "stActiveTheme-/-v2"

_pending = st.session_state.pop("_theme_apply", None)
if _pending:
    components.html(
        f"""<script>
              const w = window.parent;
              w.localStorage.setItem({_THEME_KEY!r}, JSON.stringify({_pending!r}));
              w.location.reload();
            </script>""",
        height=0,
    )

# Default is System. Only the alternative is offered — no three-way picker.
_dark_now = is_dark()
_target = "Light" if _dark_now else "Dark"
# Deliberately ONE control. A "follow system" reset cannot be offered
# conditionally: applying a choice reloads the page, which clears
# st.session_state, so any "has the user overridden?" flag is always False by
# the time it would be read. First load in a fresh browser is "System"; after
# that the button toggles explicitly between the two.
# Material icon rather than a text-only label: a sun/moon reads instantly and
# is a proper icon font, not emoji.
_icon = ":material/light_mode:" if _dark_now else ":material/dark_mode:"

# Header — a MASTHEAD, not a heading. The title measured 2.2rem against 1.75rem
# section headings: same family, same weight, 1.26x apart, so it read as "the
# biggest heading on the page" rather than "the application". It is now tighter
# and smaller while the headings shrink further, and the row carries chrome on
# the right, closed by the provenance rule below.
_h_left, _h_right = st.columns([5, 1], vertical_alignment="center")
with _h_left:
    st.markdown(
        lockup(
            "Astra Screen",
            "Component burn-in screening &middot; peer-relative anomaly detection "
            "&amp; 168&#8201;h drift prediction &middot; ISRO PS-26170",
        ),
        unsafe_allow_html=True,
    )
with _h_right:
    # key= gives the container an `st-key-theme_toggle` class (emitted by the
    # frontend, verified in the 1.63 bundle), which is how the CSS below strips
    # this one button back to chrome without touching the other three.
    if st.button(f"{_target} theme", icon=_icon, width="stretch",
                 key="theme_toggle",
                 help=f"Switch to the {_target.lower()} theme. "
                      "Defaults to following your system setting."):
        st.session_state["_theme_apply"] = _target
        st.rerun()

# Sidebar Configuration.
# No brand lockup here: the header lockup (Step 11) already identifies the app,
# and repeating the name ~40px away was duplication that also cost the context
# blocks their headroom.
# Dataset is the primary INPUT, so it leads. It previously rendered last, below
# the theme toggle — a cosmetic preference above the main input.
st.sidebar.markdown('<div class="card-label">Dataset</div>', unsafe_allow_html=True)
data_source_opt = st.sidebar.radio(
    "Source", ["ISRO benchmark sample", "Upload CSV"], label_visibility="collapsed"
)
_slot_upload = st.sidebar.container()

st.sidebar.markdown('<div class="card-label sec">Screening configuration</div>', unsafe_allow_html=True)
z_thresh = st.sidebar.slider("Peer anomaly threshold |Z|", 1.5, 5.0,
                             float(ROBUST_Z_SCORE_THRESHOLD), 0.1,
                             help="Robust median/MAD Z against the part's own lot. "
                                  "REVIEW at |Z| >= this; FLAG at |Z| >= this + 1.")
safety_ratio = st.sidebar.slider("Safety margin (% of limit)", 50, 95,
                                 int(SAFETY_SLOPE_MARGIN_RATIO * 100), 5) / 100.0

# Filled after the pipeline runs — containers hold their position in the
# sidebar regardless of when they are written to.
# ONE slot for both context blocks. Two adjacent st.markdown containers let
# the second ride 6px up into the first: Streamlit's markdown wrapper reported
# 110px around 126px of content, and no amount of flex-shrink guarding on the
# anonymous wrapper divs fixed it reliably. Rendering both blocks in a single
# markdown call removes the boundary rather than fighting it.
_slot_context = st.sidebar.container()

sign_off = st.sidebar.button(
    "Commit run to audit trail", width="stretch",
    help="Records this run with the thresholds above. Screening objective is to "
         "minimise false negatives: a defective component released to flight is "
         "far costlier than a good one held back.",
)


# Load / Select Dataset
@st.cache_data
def load_default_sample():
    csv_path = DATA_DIR / "sample_burnin_data.csv"
    if not csv_path.exists():
        df = generate_synthetic_burnin_data(save_path=str(csv_path))
    else:
        df = pd.read_csv(csv_path)
    return df

REQUIRED_SCHEMA = [
    ("Component_ID",    "text",   "Unique serial, e.g. C1052"),
    ("Lot_ID",          "text",   "Manufacturing lot the part belongs to"),
    ("Parameter",       "text",   "Measured parameter and unit, e.g. Leakage_Current_uA"),
    ("Value_0h",        "number", "Measurement before burn-in"),
    ("Value_24h",       "number", "Measurement at 24 hours"),
    ("Value_96h",       "number", "Measurement at 96 hours"),
    ("Value_168h",      "number", "Measurement at 168 hours — used to score the model"),
    ("Datasheet_Limit", "number", "Absolute limit for this parameter"),
]


def _template_csv() -> str:
    """A valid two-row starter file, so the required shape is copyable."""
    head = ",".join(c for c, _, _ in REQUIRED_SCHEMA)
    return (f"{head}\n"
            "C1001,LOT_A,Leakage_Current_uA,8.73,9.27,11.07,11.84,50.0\n"
            "C1002,LOT_A,Leakage_Current_uA,9.45,9.61,11.24,11.41,50.0\n")


if data_source_opt == "Upload CSV":
    uploaded_file = _slot_upload.file_uploader("Component test data (CSV)", type=["csv"])
    if uploaded_file is None:
        # A real empty state. This previously called st.stop() after a bare
        # warning, which blanked the whole page — KPIs, tabs and charts — and
        # gave no indication of what a valid file looks like.
        st.markdown('<div class="card-label">Expected format</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame(REQUIRED_SCHEMA, columns=["Column", "Type", "Description"]),
            hide_index=True, width="stretch",
        )
        st.download_button(
            "Download template CSV", _template_csv(),
            file_name="astra_screen_template.csv", mime="text/csv",
        )
        st.caption("All eight columns are required. Rows with unusable measurements are dropped "
                   "and reported; duplicate Component_IDs keep the first occurrence.")
        st.stop()
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read that file as CSV — {type(exc).__name__}: {exc}")
        st.stop()
    dataset_name = uploaded_file.name
else:
    raw_df = load_default_sample()
    dataset_name = "ISRO Benchmark Sample Dataset"

# Execute Pipeline. log_run=False: the thresholds are live controls, so an
# ungated write would append an audit row on every slider movement. Recording
# is deliberate, via the sign-off button in the sidebar.
try:
    with st.spinner("Screening components..."):
        processed_df, summary = run_full_screening_pipeline(
            raw_df,
            dataset_name=dataset_name,
            z_threshold=z_thresh,
            safety_ratio=safety_ratio,
            log_run=False,
        )
except Exception as exc:
    st.error(f"Screening could not complete — {type(exc).__name__}: {exc}")
    st.caption("Check the uploaded file against the expected format, or switch back "
               "to the benchmark dataset.")
    st.stop()

# --- Guard the error path -------------------------------------------------
# When validation clears every row the pipeline returns {"error","validation"}
# only. Reading summary["total_processed"] here raised KeyError and put a raw
# traceback on screen — on the one path the author handled deliberately.
if "screening_counts" not in summary:
    st.error(summary.get("error", "Screening produced no valid component records."))
    with st.expander("Validation detail"):
        st.json(summary.get("validation", {}))
    st.stop()

total_count = summary["total_processed"]
counts = summary["screening_counts"]
perf = summary.get("performance_metrics", {})


def _unit(df: pd.DataFrame) -> str:
    """Measurement unit from the Parameter column, e.g. Leakage_Current_uA -> µA.

    Derived rather than hardcoded: the app previously printed "µA" everywhere,
    so a propagation-delay or gain dataset was silently mislabelled.
    """
    try:
        tail = str(df["Parameter"].iloc[0]).rsplit("_", 1)[-1]
    except Exception:
        return ""
    return {"uA": "µA", "mA": "mA", "nA": "nA", "mV": "mV", "V": "V",
            "ns": "ns", "ps": "ps", "us": "µs", "ohm": "Ω"}.get(tail, tail)

@st.cache_data(show_spinner=False)
def _honest_metrics(df: pd.DataFrame):
    """In-sample vs lot-wise held-out vs naive baseline.

    The app previously displayed only the in-sample fit — the model scored on
    the rows it was trained on — which is what "R2 = 0.992" actually was.
    GroupKFold on Lot_ID rather than a random split: components from one lot
    share process history, so a random split leaks lot-level signal, and in
    deployment you screen a NEW lot using a model trained on PAST lots.

    Returns None when the data cannot support a grouped hold-out.
    Costs ~0.2 s on 200 rows, so it runs live rather than being precomputed.
    """
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if "Value_168h" not in df.columns or "Lot_ID" not in df.columns:
        return None
    d = df.dropna(subset=["Value_0h", "Value_24h", "Value_168h"])
    n_lots = d["Lot_ID"].nunique()
    if len(d) < 20 or n_lots < 2:
        return None

    predictor = DriftPredictor168h()
    predictor.fit(d)
    X, y = predictor._extract_features(d), d["Value_168h"]

    def row(label, pred):
        return {"Evaluation": label,
                "MAE": float(mean_absolute_error(y, pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
                "R²": float(r2_score(y, pred))}

    try:
        oof = cross_val_predict(clone(predictor.model), X, y,
                                cv=GroupKFold(n_splits=min(5, n_lots)),
                                groups=d["Lot_ID"])
    except Exception:
        return None

    naive = X["Value_0h"] + X["Early_Slope_per_hr"] * 168.0
    return [
        row(f"Held out by lot ({min(5, n_lots)}-fold)", oof),
        row("In-sample (training fit)", predictor.model.predict(X)),
        row("Naive linear extrapolation", naive),
    ]

unit = _unit(processed_df)
_limits = processed_df["Datasheet_Limit"]
limit_txt = f"{_limits.iloc[0]:g} {unit}".strip() if _limits.nunique() == 1 else "their datasheet limits"

# --- Provenance strip -----------------------------------------------------
# Instruments show their own state. These are the thresholds this run actually
# used, so the strip, the evidence text and the audit entry now all agree.
_entry = summary.get("audit_entry") or {}
if _entry.get("timestamp"):
    _ts = _entry["timestamp"][:16].replace("T", " ") + " · committed"
else:
    _ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + " · uncommitted"
_mae_txt = (f"{perf['mae']:g}&nbsp;{unit}".strip()
            if perf.get("has_ground_truth") and perf.get("mae") is not None else "n/a")

st.markdown(
    f'<div class="provenance">'
    f'<span class="sys">ASTRA SCREEN</span><span class="sep"></span>'
    f'<span>{_ts}</span><span class="sep"></span>'
    f'<span>{dataset_name.upper()}</span><span class="sep"></span>'
    f'<span>N&nbsp;=&nbsp;{total_count}</span><span class="sep"></span>'
    f'<span>|Z|&nbsp;&ge;&nbsp;{z_thresh:.1f}</span><span class="sep"></span>'
    f'<span>MARGIN&nbsp;{safety_ratio:.0%}</span><span class="sep"></span>'
    f'<span title="In-sample: the model is scored on the same rows it was fitted on. '
    f'Held-out figures are on the Model &amp; Audit tab.">'
    f'MAE&nbsp;<span class="unit">{_mae_txt}</span><span class="hint">&nbsp;in-sample</span></span>'
    f'</div>',
    unsafe_allow_html=True
)

# --- Verdict --------------------------------------------------------------
breach = int(processed_df["ModuleB_Datasheet_Breach_Flag"].sum())
offbase = max(counts["FLAG"] - breach, 0)
calm = "" if counts["FLAG"] else " calm"   # a zero must never render as an alarm


# --- Sidebar context, written into the slots reserved above -----------------
# Lot-level view. The original audit flagged that the app "has no lot-level
# opinion despite Lot_ID driving every statistic it computes". Honest caveat:
# on this benchmark the lots are near-uniform (flag rates 15-18%), so this
# shows little variation here — the structure is right and it would carry real
# signal on production lot data.
_lots = (processed_df.groupby("Lot_ID")
         .agg(n=("Component_ID", "size"),
              flag=("Screening_Status", lambda c: int((c == RISK_FLAG).sum())),
              med=("Value_24h", "median"))
         .reset_index().sort_values("Lot_ID"))

_rows = "".join(
    f'<div class="lotrow"><span class="lot">{r.Lot_ID}</span>'
    f'<span class="bar"><i style="width:{min(100, round(100 * r.flag / max(r.n, 1)))}%"></i></span>'
    f'<span class="num">{r.flag}<span class="of">/{r.n}</span></span></div>'
    for r in _lots.itertuples()
)
# Legend. Answers a question the UI answered nowhere: what REVIEW actually
# means (off-baseline but still inside spec).
_LEGEND = [
    (RISK_FLAG, "flag", "×", "Predicted to breach the limit"),
    (RISK_REVIEW, "review", "!", "Off-baseline, still in spec"),
    (RISK_PASS, "pass", "·", "Within the lot baseline"),
]
_legend_html = "".join(
    f'<div class="lgd lgd-{cls}"><span class="g">{glyph}</span>'
    f'<span class="t"><b>{name.split(" (")[0]}</b>{desc}</span></div>'
    for name, cls, glyph, desc in _LEGEND
)

# Both context blocks render in ONE markdown call. As two adjacent containers
# the second rode 6px up into the first: Streamlit's markdown wrapper measured
# 110px around 126px of content, and flex-shrink guards on the anonymous
# wrapper divs did not fix it reliably. One block removes the boundary.
_slot_context.markdown(
    '<div class="card-label sec">Lots'
    '<span class="lbl-hint">flagged / screened</span></div>'
    f'<div class="lots">{_rows}</div>'
    '<div class="card-label ctx-gap sec">Decisions</div>'
    f'<div class="lgd-list">{_legend_html}</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="verdict card">'
    f'<div class="vnum{calm}">{counts["FLAG"]}</div>'
    f'<div class="vtxt">of {total_count} components held back from release'
    f'<div class="vsub"><span class="n">{breach}</span> PREDICTED TO BREACH '
    f'<span class="unit">{limit_txt}</span> BY 168&#8201;H'
    f' &nbsp;·&nbsp; <span class="n">{offbase}</span> OFF-BASELINE VS THEIR LOT'
    f' &nbsp;·&nbsp; <span class="n">{counts["REVIEW"]}</span> FOR REVIEW'
    f' &nbsp;·&nbsp; <span class="n">{counts["PASS"]}</span> CLEAR</div></div></div>',
    unsafe_allow_html=True
)



# Deliberate sign-off: one run, one record, with the thresholds actually used.
if sign_off:
    entry = log_screening_run(
        dataset_name=dataset_name,
        total_components=total_count,
        flagged_count=counts["FLAG"],
        review_count=counts["REVIEW"],
        pass_count=counts["PASS"],
        metrics=perf,
        z_threshold=z_thresh,
        safety_ratio=safety_ratio,
    )
    st.toast(f"Run committed at |Z| ≥ {z_thresh:.1f}, margin {safety_ratio:.0%}")

# Dashboard Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "SCREENING",
    "INSPECTOR",
    "PEER",
    "DRIFT",
    "MODEL & AUDIT"
])

# TAB 1: Component Screening Table
with tab1:
    st.subheader("Component Screening Results")
    
    # Filter Controls. Ratios were [2,2,4], which gave the widest slot to a
    # one-line text input while the multiselects held 16-character chips.
    f_col1, f_col2, f_col3 = st.columns([3, 3, 2])
    selected_status = f_col1.multiselect(
        "Status", STATUS_ORDER, default=STATUS_ORDER)
    selected_lot = f_col2.multiselect(
        "Lot", processed_df["Lot_ID"].unique().tolist(),
        default=processed_df["Lot_ID"].unique().tolist())
    search_comp = f_col3.text_input("Search component", "")

    filtered_df = processed_df[
        processed_df["Screening_Status"].isin(selected_status) &
        processed_df["Lot_ID"].isin(selected_lot)
    ]
    if search_comp:
        filtered_df = filtered_df[filtered_df["Component_ID"].astype(str).str.contains(
            search_comp, case=False, regex=False, na=False)]

    # Status carries a glyph as well as a colour, so the state survives
    # greyscale print and colour-blind viewing.
    _TOKEN = {RISK_PASS: "·  PASS", RISK_REVIEW: "!  REVIEW", RISK_FLAG: "×  FLAG"}

    table_df = filtered_df.assign(
        Status=filtered_df["Screening_Status"].map(_TOKEN).fillna(filtered_df["Screening_Status"]),
        # Explanation_Summary is the " | " join of up to four sentences — 372
        # characters at worst. The first reason is the primary one; the full
        # reasoning belongs in the Inspector, not in a table cell.
        Evidence=filtered_df["Explanation_List"].apply(
            lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) else ""),
    ).sort_values(
        # Risk_Score has only 2 distinct values across the flagged rows, so it
        # cannot rank them on its own; predicted 168h breaks the tie and puts
        # the parts closest to breaching at the top.
        by=["Risk_Score", "Predicted_Value_168h"], ascending=[False, False]
    )

    display_cols = [
        "Component_ID", "Lot_ID", "Status", "Risk_Score",
        "Value_0h", "Value_24h", "Value_24h_Robust_Z",
        "Predicted_Value_168h", "Datasheet_Limit", "Evidence",
    ]

    _u = f" ({unit})" if unit else ""
    selection = st.dataframe(
        table_df[display_cols],
        width="stretch",
        hide_index=True,
        height=560,
        on_select="rerun",
        selection_mode="single-row",
        key="screening_table",
        column_config={
            "Component_ID": st.column_config.TextColumn("Component", width="small"),
            "Lot_ID": st.column_config.TextColumn("Lot", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Risk_Score": st.column_config.ProgressColumn(
                "Risk", min_value=0, max_value=100, format="%.0f", width="small",
                help="Composite 0-100. Note: only two distinct values occur among "
                     "flagged parts, so it separates PASS from FLAG but does not rank within FLAG."),
            "Value_0h": st.column_config.NumberColumn(f"0 h{_u}", format="%.2f", width="small"),
            "Value_24h": st.column_config.NumberColumn(f"24 h{_u}", format="%.2f", width="small"),
            "Value_24h_Robust_Z": st.column_config.NumberColumn(
                "Peer Z", format="%+.2f", width="small",
                help="Robust (median/MAD) Z-score against the part's own lot."),
            "Predicted_Value_168h": st.column_config.NumberColumn(
                f"Pred. 168 h{_u}", format="%.2f", width="small"),
            "Datasheet_Limit": st.column_config.NumberColumn(f"Limit{_u}", format="%.0f", width="small"),
            "Evidence": st.column_config.TextColumn("Primary evidence", width="large"),
        },
    )

    # Clicking a row drives the Inspector tab. Previously the Inspector had its
    # own selectbox with no relationship to what was on screen here.
    _rows = (selection.get("selection") or {}).get("rows") if isinstance(selection, dict) else None
    if _rows:
        st.session_state["inspect_cid"] = table_df.iloc[_rows[0]]["Component_ID"]

    st.caption(f"{len(filtered_df)} of {len(processed_df)} components shown. "
               "Select a row to open it in the Inspector.")

# TAB 2: Detail Inspector & Rationale
with tab2:
    st.subheader("QA Component Detail & Auditable Rationale")
    
    # Ordered worst-first so the default lands on something worth looking at.
    comp_list = processed_df.sort_values(
        by=["Risk_Score", "Predicted_Value_168h"], ascending=[False, False]
    )["Component_ID"].tolist()

    # A row click in the SCREENING tab sets this. Guarded: if the stored id is
    # no longer present (filters changed, different dataset loaded) Streamlit
    # raises on an out-of-range index, so fall back to the worst part.
    _pre = st.session_state.get("inspect_cid")
    _idx = comp_list.index(_pre) if _pre in comp_list else 0
    selected_cid = st.selectbox("Component", comp_list, index=_idx)
    
    comp_row = processed_df[processed_df["Component_ID"] == selected_cid].iloc[0]
    
    c_status = comp_row["Screening_Status"]
    c_score = comp_row["Risk_Score"]
    
    # Status Banner
    if c_status == "FLAG (HIGH RISK)":
        st.error(f"**STATUS: {c_status}** (Risk Score: {c_score}/100)")
    elif c_status == "REVIEW":
        st.warning(f"**STATUS: {c_status}** (Risk Score: {c_score}/100)")
    else:
        st.success(f"**STATUS: {c_status}** (Risk Score: {c_score}/100)")
        
    m_col1, m_col2 = st.columns(2, gap="large")


    def _spec(rows) -> str:
        """Aligned label/value rows. Values are mono + tabular so digits line up;
        the previous st.write("**Label**: value") pairs were proportional and
        ragged, so nothing in the column agreed with anything else."""
        out = []
        for label, value, sub in rows:
            u = f'<span class="u">{sub}</span>' if sub else ""
            out.append(f'<div class="spec"><span class="k">{label}</span>'
                       f'<span class="v">{value}{u}</span></div>')
        return "".join(out)

    # Parameter renders as the raw column value, e.g. "Leakage_Current_uA".
    # Show the human name and let the unit ride on the values where it belongs.
    _param = str(comp_row["Parameter"])
    _param_name = _param.rsplit("_", 1)[0].replace("_", " ") if "_" in _param else _param

    def _num(col, dp=2):
        v = comp_row.get(col)
        return "—" if v is None or pd.isna(v) else f"{float(v):.{dp}f}"

    with m_col1:
        st.markdown('<div class="card-label">Identification</div>', unsafe_allow_html=True)
        st.markdown(_spec([
            ("Component", comp_row["Component_ID"], ""),
            ("Manufacturing lot", comp_row["Lot_ID"], ""),
            ("Parameter", _param_name, ""),
        ]), unsafe_allow_html=True)

        # New block. The inspector previously showed a predicted value with no
        # measured values beside it — odd on a screen whose job is justifying
        # the decision.
        st.markdown('<div class="card-label">Measurements</div>', unsafe_allow_html=True)
        meas = [("0 h", _num("Value_0h"), unit), ("24 h", _num("Value_24h"), unit)]
        if "Value_96h" in comp_row.index and pd.notna(comp_row.get("Value_96h")):
            meas.append(("96 h", _num("Value_96h"), unit))
        meas.append(("168 h predicted", _num("Predicted_Value_168h"), unit))
        if "Value_168h" in comp_row.index and pd.notna(comp_row.get("Value_168h")):
            meas.append(("168 h measured", _num("Value_168h"), unit))
        st.markdown(_spec(meas), unsafe_allow_html=True)

        st.markdown('<div class="card-label">Limits &amp; scores</div>', unsafe_allow_html=True)
        st.markdown(_spec([
            ("Datasheet limit", _num("Datasheet_Limit", 1), unit),
            # Was hardcoded "(80%)" — wrong as soon as the slider moves, and the
            # slider now works.
            (f"Safety threshold ({safety_ratio:.0%})", _num("Safety_Threshold_168h", 1), unit),
            ("Peer Z at 24 h", f"{float(comp_row['Value_24h_Robust_Z']):+.2f}", ""),
            ("Risk score", f"{float(c_score):.0f}", "/100"),
        ]), unsafe_allow_html=True)

    with m_col2:
        st.markdown('<div class="card-label">Auditable rationale</div>', unsafe_allow_html=True)
        # Severity comes from the explainer's own prefixes rather than the
        # component status, so a nominal line inside a flagged part still reads
        # as nominal. Every reason previously rendered as an identical blue
        # st.info, so "will breach the datasheet limit" looked the same as
        # "measurements align with the lot baseline".
        CRITICAL = ("Predicted Limit Failure",)
        WARNING = ("Peer Anomaly", "Safety Margin Breach", "Abnormal Early Drift",
                   "Unsupervised Pattern Outlier")
        for exp in comp_row["Explanation_List"]:
            head, _, body = str(exp).partition(":")
            if not body:                      # nominal lines carry no "Label:" prefix
                head, body = "Nominal", exp
            sev = ("crit" if head.strip() in CRITICAL
                   else "warn" if head.strip() in WARNING else "ok")
            st.markdown(
                f'<div class="ev ev-{sev}" role="listitem">'
                f'<div class="ev-h">{head.strip()}</div>'
                f'<div class="ev-b">{body.strip()}</div></div>',
                unsafe_allow_html=True
            )

# TAB 3: Peer Lot Distribution (Module A)
with tab3:
    st.subheader("Module A — peer population analysis")

    # Side by side, not stacked. Full width gave each chart a 3.37:1 box, which
    # is why the data read as squashed; halved, they sit near 1.6:1 and the
    # scatter gets the near-square frame a scatter actually wants.
    a_col1, a_col2 = st.columns(2, gap="large")

    with a_col1:
        st.markdown('<div class="card-label">Lot distribution</div>', unsafe_allow_html=True)
        st.caption("Each box is the full lot population — the distribution the median and MAD "
                   "are computed from. The accent rule marks the lot median.")
        st.plotly_chart(viz.lot_distribution(processed_df, "Value_24h", unit),
                        width="stretch", theme=None)

    with a_col2:
        st.markdown('<div class="card-label">Failure mode</div>', unsafe_allow_html=True)
        st.caption("Parts that start off-baseline drift at a normal rate; parts that drift fast "
                   "start on-baseline. Two independent modes, hence two detectors.")
        st.plotly_chart(viz.failure_mode_quadrant(processed_df, z_thresh, unit),
                        width="stretch", theme=None)

# TAB 4: Trajectory & Drift (Module B)
with tab4:
    st.subheader("Module B — measured trajectory and 168 h projection")

    _ranked = processed_df.sort_values(
        by=["Risk_Score", "Predicted_Value_168h"], ascending=[False, False])
    selected_comp_traj = st.multiselect(
        "Components", _ranked["Component_ID"].tolist(),
        default=_ranked["Component_ID"].head(4).tolist(),
        help="Solid is measured, dashed is projected to 168 h.")

    if not selected_comp_traj:
        st.caption("Select one or more components to plot their trajectories.")
    else:
        _lim = processed_df["Datasheet_Limit"]
        st.plotly_chart(
            viz.trajectories(processed_df, selected_comp_traj,
                             float(_lim.iloc[0]), safety_ratio, unit),
            width="stretch", theme=None)
        if _lim.nunique() > 1:
            st.caption("This dataset mixes datasheet limits; the reference lines show the first.")

# TAB 5: Model Evaluation & Audit Trail
with tab5:
    st.subheader("Model Evaluation & Audit Compliance")
    
    st.markdown('<div class="card-label">Model prediction performance</div>', unsafe_allow_html=True)

    if not perf.get("has_ground_truth"):
        st.info("No 168 h ground truth in this dataset, so prediction error cannot be scored. "
                "Upload a completed burn-in run to evaluate the model.")
    else:
        honest = _honest_metrics(raw_df)
        if honest is None:
            st.caption("Not enough lots for a grouped hold-out; showing in-sample fit only.")
            st.dataframe(pd.DataFrame([{
                "Evaluation": "In-sample (training fit)", "MAE": perf["mae"],
                "RMSE": perf["rmse"], "R²": perf["r2"]}]), hide_index=True, width="stretch")
        else:
            st.dataframe(pd.DataFrame(honest), hide_index=True, width="stretch",
                column_config={
                    "Evaluation": st.column_config.TextColumn("Evaluation", width="medium"),
                    "MAE": st.column_config.NumberColumn(f"MAE{(' (' + unit + ')') if unit else ''}", format="%.3f"),
                    "RMSE": st.column_config.NumberColumn("RMSE", format="%.3f"),
                    "R²": st.column_config.NumberColumn("R²", format="%.4f"),
                })
        st.caption(
            "Held-out is grouped by **Lot_ID**, not a random split: you screen a *new* lot with a "
            "model trained on *past* lots, so a lot-wise hold-out is the only split that reflects "
            "deployment. In-sample is the same rows the model was fitted on and is shown for "
            "reference only. The naive baseline is straight-line extrapolation from 0 h and 24 h — "
            "the model has to beat it to be worth having."
        )

    st.markdown("")
    st.markdown('<div class="card-label">Audit trail</div>', unsafe_allow_html=True)
    history = get_audit_history(limit=20)
    if history:
        # Was a stack of st.code blocks: syntax-highlighted terminal output with a
        # hover copy button, presented as a compliance record.
        hist_df = pd.DataFrame([{
            "Time": e["timestamp"][:19].replace("T", " "),
            "Dataset": e["dataset"],
            "N": e["total_components"],
            "Pass": e["summary"]["PASS"],
            "Review": e["summary"]["REVIEW"],
            "Flag": e["summary"]["FLAG"],
            "|Z|": e["configured_thresholds"]["robust_z_score_threshold"],
            "Margin": float(e["configured_thresholds"]["safety_margin_ratio"]) * 100.0,
        } for e in reversed(history)])
        st.dataframe(hist_df, hide_index=True, width="stretch",
            height=min(330, 44 + 35 * len(hist_df)),
            column_config={
                "Time": st.column_config.TextColumn("Time", width="small"),
                "Dataset": st.column_config.TextColumn("Dataset", width="medium"),
                "N": st.column_config.NumberColumn("N", format="%d", width="small"),
                "Pass": st.column_config.NumberColumn("Pass", format="%d", width="small"),
                "Review": st.column_config.NumberColumn("Review", format="%d", width="small"),
                "Flag": st.column_config.NumberColumn("Flag", format="%d", width="small"),
                "|Z|": st.column_config.NumberColumn("|Z|", format="%.1f", width="small"),
                "Margin": st.column_config.NumberColumn("Safety margin", format="%.0f%%", width="small"),
            })
        st.caption("Every committed run is appended here with the thresholds it actually used. "
                   "Runs are recorded only on explicit sign-off, so adjusting a control does not "
                   "write to the trail.")
    else:
        st.caption("No runs committed yet. Adjust the thresholds and use "
                   "**Commit run to audit trail** in the sidebar to record one.")

# Footer
# The old footer repeated "Astra Screen · ISRO PS-26170" when the masthead
# subtitle already ends in PS-26170 — the same identifier twice on one page.
# What a footer can usefully carry here is the run's provenance instead.
st.markdown(
    f'<div class="footer-note">Decisions above were produced at '
    f'|Z| &ge; {z_thresh:.1f}, safety margin {safety_ratio:.0%} of the datasheet '
    f'limit. Commit a run to record them in the audit trail.</div>',
    unsafe_allow_html=True
)
