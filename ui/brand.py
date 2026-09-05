"""
Product mark and header lockup.

The mark is the product in one glyph: a drift ramp rising to cross a limit
line. It is drawn as inline SVG rather than reusing assets/favicon.png for
three reasons — the PNG bakes in a dark ground that would look wrong on the
light page, SVG takes its colours from the active theme, and it stays crisp
at any size.

Colour choice is measured, not aesthetic. Three options were compared for
ground-against-page contrast:

    fixed dark ground   16.77:1 light / 1.05:1 dark   <- vanishes on dark
    fixed accent ground  7.62:1 light / 2.09:1 dark   <- weak on dark
    ground inverts       7.62:1 light / 7.26:1 dark   <- chosen

So the ground is the accent and the ramp is the page colour: dark petrol with
a light ramp in light mode, light petrol with a dark ramp in dark mode.

The red breach segment from the favicon is deliberately omitted here. It
measures 3.20:1 (light) and 2.81:1 (dark) against the ground — under the 3:1
floor for graphical objects — and at 44px it would be about four pixels. The
meaning is carried by the geometry: a line rising through a threshold.
"""

from ui.theme import palette


def mark_svg(size: int = 44) -> str:
    """The product mark, coloured from the active theme."""
    p = palette()
    ground, stroke = p["accent"], p["surface"]
    r = round(size * 0.22, 2)
    w = max(2.0, size * 0.085)          # ramp weight
    lw = max(1.0, size * 0.045)         # limit rule weight
    pad, limit_y = size * 0.22, size * 0.38
    return (
        f'<svg class="mark" width="{size}" height="{size}" viewBox="0 0 {size} {size}" '
        f'role="img" aria-label="Astra Screen" xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="0" y="0" width="{size}" height="{size}" rx="{r}" fill="{ground}"/>'
        f'<line x1="{pad}" y1="{limit_y}" x2="{size - pad}" y2="{limit_y}" '
        f'stroke="{stroke}" stroke-width="{lw}" stroke-linecap="round" opacity="0.55"/>'
        f'<path d="M {pad} {size * 0.76} L {size - pad} {size * 0.24}" '
        f'stroke="{stroke}" stroke-width="{w}" stroke-linecap="round" fill="none"/>'
        f'</svg>'
    )


def lockup(title: str, subtitle: str, badge: str = "", size: int = 44) -> str:
    """Mark + title (+ identifier badge) + subtitle as one aligned unit.

    Replaces two stacked text divs that floated with nothing anchoring them —
    the measured cause of "the product name doesn't announce itself".

    `badge` carries an identifier rather than a description — the problem ID
    used to sit at the end of the subtitle, where it was both the wrong kind
    of information for that line and the reason the line wrapped.
    """
    badge_html = f'<span class="lockup-badge">{badge}</span>' if badge else ""
    return (
        f'<div class="lockup-head">{mark_svg(size)}'
        f'<div class="lockup-txt">'
        f'<div class="main-header">{title}{badge_html}</div>'
        f'<div class="sub-header">{subtitle}</div>'
        f'</div></div>'
    )
