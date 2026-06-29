"""The PDF deliverable (Section 10), matching the repo's reportlab convention.

Assembles fit metrics, the SHAP summary, the Fama-MacBeth coefficient table, the
top/bottom descriptive comparison, and the per-fold consistency table into
``outputs/drift_ml_report_*.pdf``. Also writes the ranked driver map to
``outputs/feature_importance.csv``.

Matplotlib figure builders are kept here (like ``pead.equities.plots``) and
embedded as PNGs; the document itself is built with reportlab. All builders are
defensive: missing/empty inputs render a "no data" placeholder rather than
raising, so the report is produced even on small samples.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
)

from .config import DriftMLConfig

_NAVY = colors.HexColor("#1f4e79")
_LIGHT = colors.HexColor("#dce6f1")
_RED = "#c0392b"
_GREEN = "#1e8449"
_BLUE = "#1f4e79"

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def _placeholder(msg: str):
    fig, ax = plt.subplots(figsize=(9, 2))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, color="0.4")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_oos_metrics(results: dict, cfg: DriftMLConfig):
    """Out-of-sample IC, R^2, and decile spread per horizon (+ classifier AUC)."""
    oos_reg = results.get("oos_reg", {}) or {}
    horizons = [h for h in results.get("horizons", ()) if h in oos_reg and oos_reg[h]]
    if not horizons:
        return _placeholder("No out-of-sample folds (need more calendar quarters).")

    ic = [oos_reg[h].get("spearman_ic", np.nan) for h in horizons]
    r2 = [oos_reg[h].get("r2", np.nan) for h in horizons]
    spread = [oos_reg[h].get("decile_spread", np.nan) for h in horizons]
    x = np.arange(len(horizons))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    w = 0.38
    a1.bar(x - w / 2, ic, w, color=_BLUE, label="Spearman IC")
    a1.bar(x + w / 2, r2, w, color=_GREEN, label="OOS R\u00b2")
    a1.axhline(0, color="0.5", lw=0.8)
    a1.set_xticks(x); a1.set_xticklabels([f"+{h}d" for h in horizons])
    a1.set_title("Out-of-sample skill by horizon")
    a1.legend(fontsize=8, frameon=False)

    a2.bar(x, spread, color=[_GREEN if s >= 0 else _RED for s in spread])
    a2.axhline(0, color="0.5", lw=0.8)
    a2.set_xticks(x); a2.set_xticklabels([f"+{h}d" for h in horizons])
    a2.set_title("Top\u2013bottom decile spread of predictions")
    fig.tight_layout()
    return fig


def fig_shap_summary(results: dict, cfg: DriftMLConfig, top_n: int = 20):
    """Signed global SHAP importance for the primary-horizon regressor."""
    shap_imp = results.get("shap_imp")
    if shap_imp is None or shap_imp.empty:
        return _placeholder("No SHAP importances (no fitted folds).")
    d = shap_imp.head(top_n).iloc[::-1]
    direction = d.get("direction")
    colors_ = [_GREEN if (pd.notna(v) and v >= 0) else _RED
               for v in (direction if direction is not None else [np.nan] * len(d))]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.32 * len(d))))
    ax.barh(d.index, d["importance"], color=colors_)
    ax.set_xlabel("mean |SHAP|  (green = raises drift, red = lowers drift)")
    ax.set_title(f"Top drivers of drift (CAR[+1,+{results.get('primary_horizon')}])")
    fig.tight_layout()
    return fig


def fig_consistency(results: dict, cfg: DriftMLConfig, top_n: int = 20):
    """Consistency score of the top drivers (stability + significance blend)."""
    table = results.get("consistency")
    if table is None or table.empty or "consistency_score" not in table.columns:
        return _placeholder("No consistency table (no fitted folds).")
    d = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.32 * len(d))))
    ax.barh(d.index, d["consistency_score"], color=_BLUE)
    ax.set_xlim(0, 1)
    ax.set_xlabel("consistency score (0\u20131): top-k stability + linear significance")
    ax.set_title("How consistently each feature ranks as a driver")
    fig.tight_layout()
    return fig


def fig_descriptive(results: dict, cfg: DriftMLConfig, top_n: int = 12):
    """Standardized top-vs-bottom decile effect sizes (Cohen's d)."""
    desc = results.get("descriptive")
    if desc is None or desc.empty or "cohens_d" not in desc.columns:
        return _placeholder("No descriptive comparison (insufficient decile data).")
    d = desc.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.34 * len(d))))
    ax.barh(d.index, d["cohens_d"],
            color=[_GREEN if v >= 0 else _RED for v in d["cohens_d"]])
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Cohen's d  (top decile minus bottom decile)")
    ax.set_title("What strong- vs weak-drift firms look like")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# document
# --------------------------------------------------------------------------- #

def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("PTitle", parent=ss["Title"], textColor=_NAVY, fontSize=23,
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
    ss.add(ParagraphStyle("Mono", parent=ss["Normal"], fontName="Courier",
                          fontSize=8.5, leading=12, textColor=_NAVY,
                          backColor=colors.HexColor("#f0f3f8"), spaceAfter=8,
                          borderPadding=4))
    return ss


def _fig_image(fig, tmpdir: str, name: str, width: float = 6.9 * inch) -> Image:
    path = os.path.join(tmpdir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    img = Image(path)
    aspect = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * aspect
    return img


def _df_table(df: pd.DataFrame, cols: list[str], cfg_round: int = 3,
              max_rows: int = 25, index_label: str = "feature") -> Table:
    show = df.copy()
    show = show[[c for c in cols if c in show.columns]].head(max_rows)
    header = [index_label] + list(show.columns)
    data = [header]
    for idx, row in show.iterrows():
        cells = [str(idx)]
        for c in show.columns:
            v = row[c]
            if isinstance(v, (int, np.integer)):
                cells.append(f"{v:,}")
            elif isinstance(v, float) or isinstance(v, np.floating):
                cells.append("" if pd.isna(v) else f"{v:,.{cfg_round}f}")
            else:
                cells.append("" if pd.isna(v) else str(v))
        data.append(cells)
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t


def write_feature_importance_csv(table: pd.DataFrame, cfg: DriftMLConfig) -> str:
    """Persist the ranked driver map to ``outputs/feature_importance.csv``."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, "feature_importance.csv")
    out = table if table is not None else pd.DataFrame()
    out.to_csv(path, index=True)
    return path


def _headline(results: dict) -> str:
    cons = results.get("consistency")
    if cons is None or cons.empty:
        return "No fitted models yet \u2014 insufficient calendar coverage for walk-forward CV."
    top = cons.head(3)
    parts = []
    for feat, row in top.iterrows():
        d = row.get("direction", np.nan)
        arrow = "\u2191" if (pd.notna(d) and d >= 0) else ("\u2193" if pd.notna(d) else "?")
        parts.append(f"<b>{feat}</b> {arrow}")
    return "Most consistent drivers of post-earnings drift: " + ", ".join(parts) + "."


def build_pdf(cfg: DriftMLConfig, results: dict) -> str:
    """Render the full PDF report and return its path."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(cfg.output_dir, f"drift_ml_report_{stamp}.pdf")

    ss = _styles()
    story = []
    df = results.get("df")
    n_events = 0 if df is None else len(df)
    n_features = len(results.get("feature_cols", []))
    ph = results.get("primary_horizon", cfg.primary_horizon)

    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("What Drives Post-Earnings Drift", ss["PTitle"]))
    story.append(Paragraph("A feature-attribution study of the cross-section of realized drift",
                           ss["PSub"]))
    story.append(Paragraph(f"Generated {datetime.now():%B %d, %Y %H:%M}", ss["PSub"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        f"<b>Sample.</b> {n_events:,} quarterly earnings events "
        f"({cfg.start_year}\u2013{cfg.end_year}), {n_features} point-in-time features. "
        f"Label = market-adjusted CAR[+1,+{ph}] (the drift, excluding the day-0 jump), "
        f"z-scored within each calendar quarter.", ss["Body"]))
    story.append(Paragraph(
        "<b>Method.</b> An interpretable Fama-MacBeth cross-sectional regression and a "
        "non-linear LightGBM model, both evaluated under purged, embargoed walk-forward "
        "cross-validation so each 60-day label cannot leak into neighbouring folds. SHAP "
        "and permutation importances are aggregated across folds to score how "
        "<i>consistently</i> each feature drives the drift.", ss["Body"]))
    story.append(Paragraph(_headline(results), ss["Body"]))
    story.append(Paragraph("<b>Reproduce:</b>", ss["Body"]))
    story.append(Paragraph(cfg.as_cli_command(), ss["Mono"]))
    story.append(PageBreak())

    with tempfile.TemporaryDirectory() as tmp:
        story.append(Paragraph("1. Out-of-sample skill", ss["H2"]))
        oos_clf = results.get("oos_clf")
        clf_txt = ""
        if oos_clf:
            clf_txt = (f" Classifier (strong-up vs down drift): ROC-AUC "
                       f"{oos_clf.get('roc_auc', float('nan')):.3f}, PR-AUC "
                       f"{oos_clf.get('pr_auc', float('nan')):.3f}.")
        story.append(Paragraph(
            "Does the model rank drift correctly out of sample, and is there an economic "
            "spread between its top and bottom predictions?" + clf_txt, ss["Body"]))
        story.append(_fig_image(fig_oos_metrics(results, cfg), tmp, "oos"))
        story.append(Paragraph(
            "Figure 1. Spearman rank IC, OOS R\u00b2, and top\u2013bottom decile spread "
            "per drift horizon.", ss["Caption"]))
        story.append(PageBreak())

        story.append(Paragraph("2. Which features drive the drift", ss["H2"]))
        story.append(Paragraph(
            "Global SHAP importance with sign: green features push drift up, red push it "
            "down. This is the non-linear analogue of the Fama-MacBeth coefficients.",
            ss["Body"]))
        story.append(_fig_image(fig_shap_summary(results, cfg), tmp, "shap"))
        story.append(Paragraph("Figure 2. Signed global feature importance (SHAP).",
                               ss["Caption"]))
        story.append(_fig_image(fig_consistency(results, cfg), tmp, "cons"))
        story.append(Paragraph(
            "Figure 3. Consistency score \u2014 how stably each feature ranks as a top "
            "driver across walk-forward folds, blended with linear significance.",
            ss["Caption"]))
        story.append(PageBreak())

        story.append(Paragraph("3. Strong vs weak drift firms", ss["H2"]))
        story.append(Paragraph(
            "Standardized difference (Cohen's d) in each feature between top-decile and "
            "bottom-decile drift events.", ss["Body"]))
        story.append(_fig_image(fig_descriptive(results, cfg), tmp, "desc"))
        story.append(Paragraph("Figure 4. Top-minus-bottom decile effect sizes.",
                               ss["Caption"]))
        story.append(PageBreak())

        story.append(Paragraph("4. Headline driver map", ss["H2"]))
        story.append(Paragraph(
            "Feature \u2192 direction \u2192 strength \u2192 consistency. <i>sign_agreement</i> "
            "flags where the linear (Fama-MacBeth) and non-linear (SHAP) signs concur.",
            ss["Body"]))
        cons = results.get("consistency")
        if cons is not None and not cons.empty:
            story.append(_df_table(cons, [
                "importance", "direction", "top_k_hit_rate", "coef", "t_stat",
                "sign_agreement", "consistency_score"]))
        else:
            story.append(Paragraph("Not available (no fitted folds).", ss["Body"]))
        story.append(Spacer(1, 0.15 * inch))

        story.append(Paragraph("5. Fama-MacBeth coefficients", ss["H2"]))
        fm = results.get("fm")
        if fm is not None and not fm.empty:
            story.append(_df_table(fm, ["coef", "t_stat", "n_quarters", "frac_positive"]))
        else:
            story.append(Paragraph("Not available.", ss["Body"]))

        doc = SimpleDocTemplate(
            out_path, pagesize=letter,
            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
            topMargin=0.7 * inch, bottomMargin=0.7 * inch,
            title="What Drives Post-Earnings Drift", author="PEAD drift-ML toolkit",
        )
        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)

    return out_path


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#999999"))
    canvas.drawString(0.75 * inch, 0.4 * inch, "Sub-sample drift attribution")
    canvas.drawRightString(7.75 * inch, 0.4 * inch, f"Page {doc.page}")
    canvas.restoreState()
