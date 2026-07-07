from pathlib import Path
import subprocess
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figs"
SVG_OUT = FIG_DIR / "radler_workflow_overview.svg"
PDF_OUT = FIG_DIR / "radler_workflow_overview.pdf"
PNG_OUT = FIG_DIR / "radler_workflow_overview.png"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def rect(x, y, w, h, fill, stroke, rx=20, sw=2, klass="", extra=""):
    cls = f' class="{klass}"' if klass else ""
    return (
        f'<rect{cls} x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'
    )


def text(
    x,
    y,
    value,
    size=22,
    weight=600,
    fill="#173042",
    anchor="middle",
    extra="",
    lines=None,
    leading=1.18,
):
    if lines:
        spans = []
        start = -0.5 * (len(lines) - 1) * leading
        for i, line in enumerate(lines):
            dy = f"{start:.3f}em" if i == 0 else f"{leading:.3f}em"
            spans.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
        body = "".join(spans)
    else:
        body = esc(value)
    return (
        f'<text x="{x}" y="{y}" font-family="Avenir Next, Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" dominant-baseline="middle" {extra}>'
        f"{body}</text>"
    )


def line_arrow(x1, y1, x2, y2, stroke="#35586C", sw=3, dash="", marker="url(#arrow)", extra=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{sw}" stroke-linecap="round"{dash_attr} marker-end="{marker}" {extra}/>'
    )


def line_seg(x1, y1, x2, y2, stroke="#35586C", sw=3, dash="", extra=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{sw}" stroke-linecap="round"{dash_attr} {extra}/>'
    )


def path_arrow(path_d, stroke="#35586C", sw=3, dash="", marker="url(#arrow)", extra=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{path_d}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
        f'stroke-linecap="round" stroke-linejoin="round"{dash_attr} marker-end="{marker}" {extra}/>'
    )


def poly_arrow(points, stroke="#35586C", sw=3, dash="", marker="url(#arrow)", extra=""):
    path_d = "M " + " L ".join(f"{x} {y}" for x, y in points)
    return path_arrow(path_d, stroke=stroke, sw=sw, dash=dash, marker=marker, extra=extra)


def build_svg() -> str:
    return dedent(
        f"""\
        <svg xmlns="http://www.w3.org/2000/svg" width="1800" height="710" viewBox="0 0 1800 710">
          <defs>
            <filter id="shadow" x="-20%" y="-20%" width="140%" height="160%">
              <feDropShadow dx="0" dy="10" stdDeviation="10" flood-color="#BFCAD2" flood-opacity="0.55"/>
            </filter>
            <linearGradient id="panelTop" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stop-color="#FFF4DB"/>
              <stop offset="100%" stop-color="#FFE7B2"/>
            </linearGradient>
            <linearGradient id="panelBottom" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stop-color="#E8FBF8"/>
              <stop offset="100%" stop-color="#D3F2EE"/>
            </linearGradient>
            <linearGradient id="card" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stop-color="#FCFDFE"/>
              <stop offset="100%" stop-color="#F1F5F8"/>
            </linearGradient>
            <clipPath id="inputPreviewClip">
              <rect x="82" y="247" width="176" height="110" rx="18"/>
            </clipPath>
            <clipPath id="outputPreviewClip">
              <rect x="1532" y="247" width="176" height="110" rx="18"/>
            </clipPath>
            <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
              <path d="M0,0 L12,6 L0,12 z" fill="#35586C"/>
            </marker>
            <marker id="arrowWarm" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
              <path d="M0,0 L12,6 L0,12 z" fill="#B66A12"/>
            </marker>
          </defs>

          <rect x="0" y="0" width="1800" height="710" fill="#FFFFFF"/>
          {rect(390, 120, 980, 210, "url(#panelTop)", "#E1C27B", rx=34, sw=2)}
          {rect(390, 380, 980, 205, "url(#panelBottom)", "#8ECFC6", rx=34, sw=2)}

          {rect(445, 132, 250, 50, "#FFE3AA", "#E0B663", rx=24, sw=2)}
          {text(570, 157, "Context branch", size=27, weight=700, fill="#9A5D0D")}
          {rect(445, 394, 265, 50, "#C8EEE9", "#6EB7B0", rx=24, sw=2)}
          {text(577, 419, "Inference branch", size=27, weight=700, fill="#225E58")}

          {rect(45, 175, 250, 250, "url(#card)", "#A8B8C2", rx=36, sw=3, extra='filter="url(#shadow)"')}
          {rect(82, 247, 176, 110, "#FFFFFF", "#B3BEC8", rx=18, sw=2)}
          <image href="radler_input_real.jpg" x="82" y="247" width="176" height="110" preserveAspectRatio="xMidYMid slice" clip-path="url(#inputPreviewClip)"/>
          <rect x="82" y="247" width="176" height="110" rx="18" fill="none" stroke="#D3DCE3" stroke-width="1.5"/>
          {text(170, 120, "", lines=["Input UAV", "frame"], size=34, weight=750, fill="#173042")}
          {text(170, 455, "", lines=["same image drives", "both paths"], size=18, weight=500, fill="#5F7282")}

          {rect(430, 188, 250, 126, "url(#card)", "#E2B46D", rx=28, sw=2, extra='filter="url(#shadow)"')}
          <rect x="465" y="240" width="16" height="42" rx="4" fill="#E6A44E"/>
          <rect x="493" y="212" width="16" height="70" rx="4" fill="#E6A44E"/>
          <rect x="521" y="229" width="16" height="53" rx="4" fill="#E6A44E"/>
          <rect x="549" y="201" width="16" height="81" rx="4" fill="#E6A44E"/>
          <rect x="598" y="214" width="16" height="68" rx="4" fill="#6BB9B2"/>
          <rect x="626" y="234" width="16" height="48" rx="4" fill="#6BB9B2"/>
          {text(580, 250, "", lines=["Context", "features"], size=24, weight=730)}

          {rect(735, 188, 240, 126, "url(#card)", "#E2B46D", rx=28, sw=2, extra='filter="url(#shadow)"')}
          {text(855, 226, "", lines=["Feature scaling", "& diagnostics"], size=21, weight=730)}
          {text(855, 278, "optional redundancy check", size=15, weight=500, fill="#5F7282")}

          {rect(1030, 171, 285, 143, "#FFF9EF", "#B66A12", rx=30, sw=3, extra='filter="url(#shadow)"')}
          {text(1172, 218, "Width selector", size=28, weight=760, fill="#A7630D")}
          {rect(1070, 240, 225, 58, "#F8CB89", "#B66A12", rx=22, sw=2)}
          {rect(1090, 255, 38, 28, "#FCE6C3", "#E1B97E", rx=10, sw=1.5)}
          {rect(1140, 255, 38, 28, "#FCE6C3", "#E1B97E", rx=10, sw=1.5)}
          {rect(1190, 255, 38, 28, "#F0A44B", "#B66A12", rx=10, sw=1.5)}
          {rect(1240, 255, 38, 28, "#FCE6C3", "#E1B97E", rx=10, sw=1.5)}
          {text(1109, 269, "25", size=15, weight=700, fill="#9A5D0D")}
          {text(1159, 269, "50", size=15, weight=700, fill="#9A5D0D")}
          {text(1209, 269, "75", size=15, weight=700, fill="#FFFFFF")}
          {text(1259, 269, "100", size=15, weight=700, fill="#9A5D0D")}

          {rect(770, 432, 580, 132, "#F4FBFA", "#2E7F78", rx=30, sw=3, extra='filter="url(#shadow)"')}
          {text(1060, 460, "Slimmable segmentation backbone", size=23, weight=750, fill="#225E58")}
          <rect x="810" y="481" width="240" height="12" rx="6" fill="#61B5AD" stroke="#225E58" stroke-width="2"/>
          <rect x="810" y="503" width="330" height="12" rx="6" fill="#61B5AD" stroke="#225E58" stroke-width="2"/>
          <rect x="810" y="525" width="430" height="12" rx="6" fill="#2F8C84" stroke="#225E58" stroke-width="2"/>
          <rect x="810" y="547" width="520" height="12" rx="6" fill="#61B5AD" stroke="#225E58" stroke-width="2"/>
          {text(1030, 487, "25%", size=15, weight=700, fill="#FFFFFF", anchor="end")}
          {text(1120, 509, "50%", size=15, weight=700, fill="#FFFFFF", anchor="end")}
          {text(1220, 531, "75%", size=15, weight=700, fill="#FFFFFF", anchor="end")}
          {text(1308, 553, "100%", size=15, weight=700, fill="#FFFFFF", anchor="end")}
          {text(1035, 623, "one deployed model, multiple widths", size=18, weight=520, fill="#5F7282")}

          {rect(1485, 175, 270, 250, "url(#card)", "#8AAAC0", rx=36, sw=3, extra='filter="url(#shadow)"')}
          {rect(1532, 247, 176, 110, "#FFFFFF", "#B3BEC8", rx=18, sw=2)}
          <image href="radler_output_real.jpg" x="1532" y="247" width="176" height="110" preserveAspectRatio="xMidYMid slice" clip-path="url(#outputPreviewClip)"/>
          <rect x="1532" y="247" width="176" height="110" rx="18" fill="none" stroke="#D3DCE3" stroke-width="1.5"/>
          {text(1640, 132, "", lines=["Segmentation", "output"], size=31, weight=750, fill="#173042")}
          {text(1620, 455, "", lines=["weed mask at the", "chosen operating point"], size=18, weight=500, fill="#5F7282")}

          {line_arrow(294, 255, 390, 255)}
          {line_arrow(680, 251, 725, 251)}
          {line_arrow(975, 251, 1022, 251)}
          {poly_arrow([(293, 332), (327, 332), (327, 472), (390, 472)])}
          {line_arrow(1172, 314, 1172, 432, stroke="#B66A12", dash="10 8", marker="url(#arrowWarm)")}
          {text(1192, 374, "selected width", size=17, weight=700, fill="#B66A12", anchor="start")}
          {poly_arrow([(1350, 498), (1385, 498), (1385, 302), (1480, 302)])}

          {rect(940, 42, 430, 62, "#EFF7D9", "#ADC25A", rx=24, sw=2)}
          {text(1155, 82, "Goal: near-best IoU with lower cost", size=21, weight=760, fill="#51611B")}
        </svg>
        """
    )


def main():
    SVG_OUT.write_text(build_svg(), encoding="utf-8")
    subprocess.run(
        ["rsvg-convert", "-f", "pdf", "-o", str(PDF_OUT), str(SVG_OUT)],
        check=True,
    )
    subprocess.run(
        ["rsvg-convert", "-w", "2400", "-o", str(PNG_OUT), str(SVG_OUT)],
        check=True,
    )


if __name__ == "__main__":
    main()
