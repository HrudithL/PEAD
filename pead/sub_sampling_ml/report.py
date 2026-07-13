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
from . import descriptions as D

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


def _friendly(names) -> list[str]:
    """Map feature column names to friendly chart labels (raw name in parens)."""
    out = []
    for n in names:
        lab = D.feature_label(n)
        out.append(f"{lab}\n({n})" if lab != n else str(n))
    return out


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
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.42 * len(d))))
    ax.barh(_friendly(d.index), d["importance"], color=colors_)
    ax.set_xlabel("mean |SHAP|  (green = higher feature raises drift, red = lowers drift)")
    ax.set_title(f"Top drivers of drift (CAR[+1,+{results.get('primary_horizon')}])")
    ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    return fig


def fig_consistency(results: dict, cfg: DriftMLConfig, top_n: int = 20):
    """Consistency score of the top drivers (stability + significance blend)."""
    table = results.get("consistency")
    if table is None or table.empty or "consistency_score" not in table.columns:
        return _placeholder("No consistency table (no fitted folds).")
    d = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.42 * len(d))))
    ax.barh(_friendly(d.index), d["consistency_score"], color=_BLUE)
    ax.set_xlim(0, 1)
    ax.set_xlabel("consistency score (0\u20131): top-k stability + linear significance")
    ax.set_title("How consistently each feature ranks as a driver")
    ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    return fig


def fig_descriptive(results: dict, cfg: DriftMLConfig, top_n: int = 12):
    """Standardized top-vs-bottom decile effect sizes (Cohen's d)."""
    desc = results.get("descriptive")
    if desc is None or desc.empty or "cohens_d" not in desc.columns:
        return _placeholder("No descriptive comparison (insufficient decile data).")
    d = desc.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.46 * len(d))))
    ax.barh(_friendly(d.index), d["cohens_d"],
            color=[_GREEN if v >= 0 else _RED for v in d["cohens_d"]])
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Cohen's d  (green = strong-drift firms have more; red = less)")
    ax.set_title("What strong- vs weak-drift firms look like")
    ax.tick_params(axis="y", labelsize=7.5)
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


def write_feature_matrix(df: pd.DataFrame, cfg: DriftMLConfig) -> str:
    """Export the full event x feature x label matrix to ``outputs/event_features.csv``.

    Mirrors the cached parquet so the matrix is inspectable alongside the report
    without needing a parquet reader.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, "event_features.csv")
    (df if df is not None else pd.DataFrame()).to_csv(path, index=False)
    return path


def feature_summary(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Per-feature coverage + summary stats (count, % present, mean/std/min/med/max).

    Categorical / non-numeric features report coverage and the number of distinct
    levels; numeric features additionally report distributional stats.
    """
    n = len(df) if df is not None else 0
    rows = []
    for c in feature_cols:
        if df is None or c not in df.columns:
            continue
        s = df[c]
        present = int(s.notna().sum())
        rec = {
            "feature": c,
            "family": D.feature_family(c),
            "meaning": D.feature_desc(c),
            "n_present": present,
            "pct_present": round(100.0 * present / n, 1) if n else np.nan,
        }
        if pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors="coerce").dropna()
            if len(v):
                rec.update({
                    "mean": float(v.mean()), "std": float(v.std()),
                    "min": float(v.min()), "median": float(v.median()),
                    "max": float(v.max()), "n_levels": np.nan,
                })
        else:
            rec.update({"mean": np.nan, "std": np.nan, "min": np.nan,
                        "median": np.nan, "max": np.nan,
                        "n_levels": int(s.nunique(dropna=True))})
        rows.append(rec)
    cols = ["feature", "family", "meaning", "n_present", "pct_present",
            "mean", "std", "min", "median", "max", "n_levels"]
    out = pd.DataFrame(rows, columns=cols)
    return out.set_index("feature") if not out.empty else out


def write_feature_summary(summary: pd.DataFrame, cfg: DriftMLConfig) -> str:
    """Persist per-feature coverage/stats to ``outputs/feature_summary.csv``."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    path = os.path.join(cfg.output_dir, "feature_summary.csv")
    (summary if summary is not None else pd.DataFrame()).to_csv(path, index=True)
    return path


def _wrapped_glossary_table(pairs: list[tuple[str, ...]], headers: list[str],
                            col_widths: list[float], ss) -> Table:
    """A table whose right-hand column wraps (Paragraphs), for glossary text."""
    cell = ParagraphStyle("Cell", parent=ss["Normal"], fontSize=7.8, leading=10)
    key = ParagraphStyle("Key", parent=cell, fontName="Helvetica-Bold",
                         textColor=_NAVY)
    data = [[Paragraph(f"<b>{h}</b>", cell) for h in headers]]
    for row in pairs:
        cells = [Paragraph(str(row[0]), key)]
        cells += [Paragraph(str(x), cell) for x in row[1:]]
        data.append(cells)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _metric_glossary_flow(ss) -> list:
    """Flowables: a 'how to read this report' metric glossary."""
    rows = [(name, text) for name, text in D.METRICS.items()]
    return [_wrapped_glossary_table(rows, ["Metric", "What it means"],
                                    [1.7 * inch, 5.2 * inch], ss)]


def _feature_glossary_flow(feature_cols: list[str], ss) -> list:
    """Flowables: a feature -> family -> plain meaning table, grouped by family.

    Only features actually present in the run are documented, so the glossary
    matches the charts and driver map the reader is looking at.
    """
    present = [c for c in feature_cols if c in D.FEATURES]
    # Group by family, preserving the catalog ordering of FEATURES.
    order = list(D.FEATURES.keys())
    present.sort(key=lambda c: order.index(c) if c in order else 999)
    rows = []
    for c in present:
        meta = D.FEATURES[c]
        rows.append((c, meta["family"], meta["desc"]))
    if not rows:
        return [Paragraph("No documented features in this run.", ss["Body"])]
    return [_wrapped_glossary_table(
        rows, ["Feature", "Family", "Plain-English meaning"],
        [1.25 * inch, 1.25 * inch, 4.4 * inch], ss)]


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
    story.append(Paragraph(
        "<b>How to read this report.</b> Section A defines every metric used "
        "below; Section B explains what each input feature means. Sections 1\u20135 "
        "then present the results, each figure paired with a plain-language "
        "\u201cwhat it shows / how to read it\u201d note.", ss["Body"]))
    story.append(Paragraph("<b>Reproduce:</b>", ss["Body"]))
    story.append(Paragraph(cfg.as_cli_command(), ss["Mono"]))
    story.append(PageBreak())

    # --- Section A: how to read this report (label + metric glossary) ------- #
    story.append(Paragraph("A. How to read this report", ss["H2"]))
    story.append(Paragraph(
        f"<b>What is being predicted (\u201cthe drift\u201d).</b> {D.LABELS['drift_raw']}",
        ss["Body"]))
    story.append(Paragraph(
        f"<b>Primary target.</b> {D.LABELS['drift_z']}", ss["Body"]))
    story.append(Paragraph(
        "Two complementary models map features to this drift: an interpretable "
        "<b>Fama-MacBeth</b> linear regression (one cross-sectional fit per quarter, "
        "coefficients averaged) and a non-linear <b>LightGBM</b> tree model explained "
        "with <b>SHAP</b>. Both are validated with purged, embargoed walk-forward "
        "cross-validation so a 60-day label never leaks into a neighbouring fold.",
        ss["Body"]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("Metric glossary", ss["Body"]))
    story.extend(_metric_glossary_flow(ss))
    story.append(PageBreak())

    # --- Section B: feature glossary ---------------------------------------- #
    story.append(Paragraph("B. Feature glossary \u2014 what each input means", ss["H2"]))
    story.append(Paragraph(
        "Every feature is point-in-time: knowable strictly before trading day +1, "
        "so nothing leaks from the future. These are the inputs ranked in the SHAP "
        "and consistency charts and the Fama-MacBeth table.", ss["Body"]))
    story.extend(_feature_glossary_flow(results.get("feature_cols", []), ss))
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
        story.append(Paragraph("Figure 1. " + D.PLOTS["oos"]["shows"], ss["Caption"]))
        story.append(Paragraph("<b>How to read it.</b> " + D.PLOTS["oos"]["read"],
                               ss["Body"]))
        story.append(PageBreak())

        story.append(Paragraph("2. Which features drive the drift", ss["H2"]))
        story.append(Paragraph(D.PLOTS["shap"]["shows"], ss["Body"]))
        story.append(_fig_image(fig_shap_summary(results, cfg), tmp, "shap"))
        story.append(Paragraph("Figure 2. Signed global feature importance (SHAP).",
                               ss["Caption"]))
        story.append(Paragraph("<b>How to read it.</b> " + D.PLOTS["shap"]["read"],
                               ss["Body"]))
        story.append(PageBreak())

        story.append(Paragraph("2b. How consistent are those drivers?", ss["H2"]))
        story.append(Paragraph(D.PLOTS["cons"]["shows"], ss["Body"]))
        story.append(_fig_image(fig_consistency(results, cfg), tmp, "cons"))
        story.append(Paragraph(
            "Figure 3. Consistency score across walk-forward folds.", ss["Caption"]))
        story.append(Paragraph("<b>How to read it.</b> " + D.PLOTS["cons"]["read"],
                               ss["Body"]))
        story.append(PageBreak())

        story.append(Paragraph("3. Strong vs weak drift firms", ss["H2"]))
        story.append(Paragraph(D.PLOTS["desc"]["shows"], ss["Body"]))
        story.append(_fig_image(fig_descriptive(results, cfg), tmp, "desc"))
        story.append(Paragraph("Figure 4. Top-minus-bottom decile effect sizes.",
                               ss["Caption"]))
        story.append(Paragraph("<b>How to read it.</b> " + D.PLOTS["desc"]["read"],
                               ss["Body"]))
        story.append(PageBreak())

        story.append(Paragraph("4. Headline driver map", ss["H2"]))
        story.append(Paragraph(
            "The summary table: each feature with its strength (<i>importance</i>), "
            "<i>direction</i> (+ raises drift, \u2212 lowers it), cross-fold stability "
            "(<i>top_k_hit_rate</i>), linear effect (<i>coef</i>, <i>t_stat</i>), whether "
            "the linear and SHAP signs concur (<i>sign_agreement</i>), and the overall "
            "<i>consistency_score</i> (0\u20131). See Section A for each term.",
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

        story.append(PageBreak())
        story.append(Paragraph("6. Feature coverage & summary statistics", ss["H2"]))
        story.append(Paragraph(
            "How well-populated each feature is across the sample and its basic "
            "distribution. Low <i>pct_present</i> (e.g. WRDS fundamentals that are "
            "missing for some firm-quarters) means a feature is informative on fewer "
            "events; the full matrix is exported to <i>outputs/event_features.csv</i> "
            "and per-feature stats to <i>outputs/feature_summary.csv</i>.", ss["Body"]))
        summ = results.get("feature_summary")
        if summ is not None and not summ.empty:
            story.append(_df_table(
                summ, ["family", "n_present", "pct_present", "mean", "std",
                       "median", "n_levels"], max_rows=50))
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
