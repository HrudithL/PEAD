"""Assemble the analysis into a single formatted PDF under outputs/."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
)

from .config import Config
from . import plots, ticker_groups

_NAVY = colors.HexColor("#1f4e79")
_LIGHT = colors.HexColor("#dce6f1")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("PTitle", parent=ss["Title"], textColor=_NAVY, fontSize=24,
                          spaceAfter=6))
    ss.add(ParagraphStyle("PSub", parent=ss["Normal"], fontSize=11,
                          textColor=colors.HexColor("#555555"), spaceAfter=2))
    ss.add(ParagraphStyle("H2", parent=ss["Heading2"], textColor=_NAVY, fontSize=15,
                          spaceBefore=10, spaceAfter=6))
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=10, leading=14,
                          spaceAfter=6))
    ss.add(ParagraphStyle("Caption", parent=ss["Normal"], fontSize=8.5,
                          textColor=colors.HexColor("#666666"), spaceAfter=12,
                          alignment=1))
    ss.add(ParagraphStyle("TblCell", parent=ss["Normal"], fontSize=8.5, leading=11))
    ss.add(ParagraphStyle("Mono", parent=ss["Normal"], fontName="Courier",
                          fontSize=8.5, leading=12, textColor=_NAVY,
                          backColor=colors.HexColor("#f0f3f8"), spaceAfter=8,
                          borderPadding=4))
    ss.add(ParagraphStyle("Tiny", parent=ss["Normal"], fontSize=7, leading=9.5,
                          textColor=colors.HexColor("#444444")))
    return ss


def _fig_image(fig, tmpdir: str, name: str, width: float = 6.9 * inch) -> Image:
    path = os.path.join(tmpdir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    img = Image(path)
    aspect = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * aspect
    return img


def _universe_text(cfg: Config, firms: list[str]) -> str:
    if cfg.tickers:
        shown = ", ".join(firms[:40])
        extra = len(firms) - 40
        suffix = f" &hellip; (+{extra} more)" if extra > 0 else ""

        spec_desc = ticker_groups.describe_spec(cfg.ticker_spec) if cfg.ticker_spec else None
        if spec_desc:
            head = (f"Referenced by group name <b>{cfg.ticker_spec}</b> &mdash; "
                    f"{spec_desc}, expanded to {len(cfg.tickers):,} unique tickers "
                    f"(overlaps counted once). ")
        else:
            tag = f" (--tickers {cfg.ticker_spec})" if cfg.ticker_spec else ""
            head = f"Custom subset{tag} &mdash; "
        return f"{head}{len(firms)} firms with usable events: {shown}{suffix}"
    return (f"Full overlap of IBES quarterly-EPS firms with the price data "
            f"({len(firms):,} firms; see appendix for the full list)")


def _param_table(cfg: Config, n_events: int, firms: list[str], ss) -> Table:
    rows = [
        ["Parameter (configurable)", "Value used in this run"],
        ["Announcement years  (--start-year/--end-year)", f"{cfg.start_year} - {cfg.end_year}"],
        ["Surprise measure  (--measure)", cfg.label_measure()],
        ["Return adjustment  (--benchmark)", cfg.label_benchmark()],
        ["Surprise buckets  (--buckets)", f"{cfg.buckets} ({plots._bucket_word(cfg)})"],
        ["Event window  (--window-pre/--window-post)",
         f"[-{cfg.window_pre}, +{cfg.window_post}] trading days"],
        ["Min. analyst estimates  (--min-numest)", str(cfg.min_numest)],
        ["Ticker universe  (--tickers)",
         Paragraph(_universe_text(cfg, firms), ss["TblCell"])],
        ["Usable events", f"{n_events:,}"],
        ["Distinct firms", f"{len(firms):,}"],
    ]
    t = Table(rows, colWidths=[2.9 * inch, 3.8 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bbbbbb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _summary_table(summary: pd.DataFrame) -> Table:
    df = summary.copy()
    num_cols = [c for c in df.columns if c not in ("Bucket", "N")]
    for c in num_cols:
        df[c] = df[c].map(lambda v: f"{v:,.3f}" if "surprise" in c.lower()
                          else (f"{v:,.2f}" if isinstance(v, float) else v))
    df["N"] = df["N"].map(lambda v: f"{v:,}")

    header = list(df.columns)
    data = [header] + df.values.tolist()
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return t


def build_pdf(cfg: Config, ev: pd.DataFrame, agg: dict) -> str:
    os.makedirs(cfg.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(cfg.output_dir, f"PEAD_report_{stamp}.pdf")

    ss = _styles()
    story = []
    n_events = len(ev)
    firms = sorted(ev["oftic"].dropna().unique().tolist())
    bw = plots._bucket_word(cfg)
    bw1 = plots._bucket_word(cfg, singular=True)

    # --- Title page ---
    story.append(Spacer(1, 0.7 * inch))
    story.append(Paragraph("Post-Earnings Announcement Drift", ss["PTitle"]))
    story.append(Paragraph("An event-study of how prices keep drifting after earnings surprises",
                           ss["PSub"]))
    story.append(Paragraph(f"Generated {datetime.now():%B %d, %Y %H:%M}", ss["PSub"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(_param_table(cfg, n_events, firms, ss))
    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph("<b>Reproduce this exact run:</b>", ss["Body"]))
    story.append(Paragraph(cfg.as_cli_command(), ss["Mono"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "<b>What this measures.</b> When a company reports earnings that differ from "
        "the analyst consensus, its stock jumps on the news \u2014 but historically the "
        "price keeps drifting in the same direction for weeks afterward as information "
        "diffuses across investors. This report sorts every quarterly earnings "
        f"announcement into {cfg.buckets} {bw} by the size of its surprise, then tracks "
        "the average cumulative abnormal return (CAR) of each group around the "
        "announcement.", ss["Body"]))
    story.append(Paragraph(
        "<b>Method.</b> Earnings surprise uses the last I/B/E/S consensus before the "
        "announcement versus reported actual EPS. Day 0 is the first trading session "
        "after the news (announcements released after the close are mapped to the next "
        "day). Abnormal return is the daily stock return "
        + ("minus the SPY return" if cfg.benchmark == "spy" else "(raw, unadjusted)")
        + "; CAR is the running sum of abnormal returns across the event window. "
        "Buckets are reformed every calendar quarter and the surprise is winsorized at "
        "the 1st/99th percentiles.", ss["Body"]))

    story.append(PageBreak())

    with tempfile.TemporaryDirectory() as tmp:
        # --- Figure 1: drift fan ---
        story.append(Paragraph("1. The drift, bucket by bucket", ss["H2"]))
        story.append(Paragraph(
            "Each line is the average CAR path for one surprise bucket. If markets were "
            "perfectly efficient the lines would be flat after day 0. Instead, "
            "high-surprise firms keep rising and low-surprise firms keep falling \u2014 "
            "the post-earnings announcement drift.", ss["Body"]))
        story.append(_fig_image(plots.fig_drift_fan(agg, cfg), tmp, "fan"))
        story.append(Paragraph(
            f"Figure 1. Mean cumulative abnormal return by surprise {bw1}, "
            f"trading days [-{cfg.window_pre}, +{cfg.window_post}].", ss["Caption"]))

        story.append(_fig_image(plots.fig_announcement_zoom(agg, cfg), tmp, "zoom"))
        story.append(Paragraph(
            "Figure 2. Close-up of the announcement reaction for the lowest, middle and "
            "highest buckets.", ss["Caption"]))

        story.append(PageBreak())

        # --- Figure: terminal + long/short ---
        story.append(Paragraph("2. Magnitude of the drift", ss["H2"]))
        story.append(Paragraph(
            f"The terminal drift increases monotonically with the surprise {bw1}: the "
            "more positive the earnings surprise, the larger the subsequent gain. A "
            "long/short portfolio that buys the top bucket and shorts the bottom bucket "
            "isolates the spread.", ss["Body"]))
        story.append(_fig_image(plots.fig_terminal_bar(agg, cfg), tmp, "term"))
        story.append(Paragraph(
            f"Figure 3. CAR at +{cfg.window_post} trading days by bucket.", ss["Caption"]))
        story.append(_fig_image(plots.fig_long_short(agg, cfg), tmp, "ls"))
        story.append(Paragraph(
            "Figure 4. Top-minus-bottom CAR spread over the event window.", ss["Caption"]))

        story.append(PageBreak())

        # --- Figure: stability + distribution + counts ---
        story.append(Paragraph("3. Robustness and sample", ss["H2"]))
        story.append(_fig_image(plots.fig_annual_stability(ev, agg, cfg), tmp, "annual"))
        story.append(Paragraph(
            "Figure 5. The long/short drift year by year \u2014 whether the effect is "
            "persistent or concentrated in a few periods.", ss["Caption"]))
        story.append(_fig_image(plots.fig_counts_surprise(ev, agg, cfg), tmp, "counts"))
        story.append(Paragraph(
            "Figure 6. Sample size and average surprise per bucket (a sanity check that "
            "buckets are balanced and ordered).", ss["Caption"]))
        story.append(_fig_image(plots.fig_surprise_hist(ev, cfg), tmp, "hist"))
        story.append(Paragraph("Figure 7. Distribution of the earnings surprise measure.",
                               ss["Caption"]))

        story.append(PageBreak())

        # --- Summary table ---
        story.append(Paragraph("4. Summary statistics by bucket", ss["H2"]))
        story.append(Paragraph(
            "CAR figures are in percent. The t-statistic tests whether each bucket's "
            f"terminal (+{cfg.window_post}-day) CAR differs from zero.", ss["Body"]))
        story.append(_summary_table(agg["summary"]))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(
            f"<b>Headline result.</b> Long top {bw1} / short bottom {bw1}: "
            f"a {agg['ls_mean']*100:.2f}% spread over {cfg.window_post} trading days "
            f"(t = {agg['ls_t']:.2f}).", ss["Body"]))

        # --- Appendix: every firm in the run ---
        story.append(PageBreak())
        story.append(Paragraph(
            f"Appendix. Firms included in this run ({len(firms):,})", ss["H2"]))
        story.append(Paragraph(
            "Tickers (I/B/E/S OFTIC, matched to the price data) that contributed at "
            "least one usable quarterly announcement under the parameters above.",
            ss["Body"]))
        story.append(Paragraph(", ".join(firms), ss["Tiny"]))

        doc = SimpleDocTemplate(
            out_path, pagesize=letter,
            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
            topMargin=0.7 * inch, bottomMargin=0.7 * inch,
            title="Post-Earnings Announcement Drift", author="PEAD toolkit",
        )
        doc.build(story, onLaterPages=_footer, onFirstPage=_footer)

    return out_path


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#999999"))
    canvas.drawString(0.75 * inch, 0.4 * inch, "PEAD analysis")
    canvas.drawRightString(7.75 * inch, 0.4 * inch, f"Page {doc.page}")
    canvas.restoreState()
