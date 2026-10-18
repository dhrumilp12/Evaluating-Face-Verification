#!/usr/bin/env python3
"""Build the Project 2 written report from verified, measured JSON outputs."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/.matplotlib"))

LABELS = {
    "original": "Original", "resolution_56": "Resolution 56", "resolution_28": "Resolution 28",
    "resolution_14": "Resolution 14", "blur_0.7": "Blur 0.7", "blur_1.4": "Blur 1.4",
    "blur_2.1": "Blur 2.1", "brightness_0.75": "Brightness 0.75", "brightness_0.5": "Brightness 0.50",
    "brightness_0.25": "Brightness 0.25",
}


def read(path):
    return json.loads((ROOT / path).read_text())


def percent(value, digits=3):
    return f"{100 * value:.{digits}f}"


def plots(result, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    methods = {"magface_cosine": "MagFace cosine", "qmagface": "QMagFace"}
    lookup = {(r["condition"], r["method"]): r for r in result["aggregates"]}
    families = [(["original", "resolution_56", "resolution_28", "resolution_14"], [112, 56, 28, 14], "Resolution (pixels)"),
                (["original", "blur_0.7", "blur_1.4", "blur_2.1"], [0, .7, 1.4, 2.1], "Gaussian blur sigma (pixels)"),
                (["original", "brightness_0.75", "brightness_0.5", "brightness_0.25"], [1, .75, .5, .25], "Brightness multiplier")]
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.4), sharey=True)
    for ax, (conditions, x, label) in zip(axes, families):
        for method, name in methods.items():
            ax.plot(x, [100 * lookup[(c, method)]["mean_accuracy"] for c in conditions], marker="o", label=name)
        ax.set(xlabel=label, xticks=x)
        if x[0] > x[-1]:
            ax.invert_xaxis()
        ax.grid(alpha=.22)
    axes[0].set_ylabel("Mean fold accuracy (%)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "report_accuracy.png", dpi=220)
    plt.close(fig)
    # Pair-weighting is explicit; this is not an independent-image sample.
    quality = {}
    for condition in LABELS:
        entries = [r for r in result["distributions"] if r["condition"] == condition and r["field"] == "probe_quality"]
        quality[condition] = sum(r["mean"] * r["count"] for r in entries) / sum(r["count"] for r in entries)
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 3.2), sharey=True)
    for ax, (conditions, x, label) in zip(axes, families):
        ax.plot(x, [quality[c] for c in conditions], marker="o", color="#087e8b")
        ax.set(xlabel=label, xticks=x)
        if x[0] > x[-1]:
            ax.invert_xaxis()
        ax.grid(alpha=.22)
    axes[0].set_ylabel("Mean raw probe norm")
    fig.tight_layout()
    fig.savefig(directory / "report_quality.png", dpi=220)
    plt.close(fig)
    return quality


class Report:
    def __init__(self):
        import matplotlib
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        fontdir = Path(matplotlib.get_data_path()) / "fonts/ttf"
        for name, filename in (("DejaVu", "DejaVuSans.ttf"), ("DejaVuBold", "DejaVuSans-Bold.ttf"),
                               ("DejaVuMono", "DejaVuSansMono.ttf")):
            pdfmetrics.registerFont(TTFont(name, str(fontdir / filename)))
        pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVuBold")
        self.colors = colors
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle("ReportTitle", fontName="DejaVuBold", fontSize=21, leading=26,
                                       textColor=colors.HexColor("#123a56"), spaceAfter=12))
        self.styles.add(ParagraphStyle("ReportHeading", fontName="DejaVuBold", fontSize=13, leading=18,
                                       textColor=colors.HexColor("#123a56"), spaceBefore=15, spaceAfter=7, keepWithNext=True))
        self.styles.add(ParagraphStyle("ReportBody", fontName="DejaVu", fontSize=9.4, leading=14.2, spaceAfter=8))
        self.styles.add(ParagraphStyle("ReportSmall", fontName="DejaVu", fontSize=7.6, leading=10.2, spaceAfter=5))
        self.styles.add(ParagraphStyle("ReportCell", fontName="DejaVu", fontSize=7.7, leading=10.4))
        self.styles.add(ParagraphStyle("ReportCode", fontName="DejaVuMono", fontSize=7.3, leading=10.4,
                                       backColor=colors.HexColor("#f0f4f7"), borderPadding=8, spaceAfter=10))
        self.flow, self.markdown = [], []

    def title(self, value):
        from reportlab.platypus import Paragraph
        self.flow.append(Paragraph(escape(value), self.styles["ReportTitle"]))
        self.markdown.append("# " + value + "\n")

    def heading(self, value):
        from reportlab.platypus import Paragraph
        self.flow.append(Paragraph(escape(value), self.styles["ReportHeading"]))
        self.markdown.append("## " + value + "\n")

    def paragraph(self, value, small=False):
        from reportlab.platypus import Paragraph, KeepTogether
        self.flow.append(KeepTogether([Paragraph(escape(value), self.styles["ReportSmall" if small else "ReportBody"])]))
        self.markdown.append(value + "\n")

    def table(self, headers, rows, widths=None):
        from reportlab.platypus import Paragraph, Table, TableStyle, Spacer, KeepTogether
        all_rows = [[Paragraph(escape(str(cell)), self.styles["ReportCell"]) for cell in row] for row in [headers, *rows]]
        table = Table(all_rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.colors.HexColor("#dce8ef")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [self.colors.white, self.colors.HexColor("#f5f7f9")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, 0), .5, self.colors.HexColor("#9aafbf")),
        ]))
        self.flow.append(KeepTogether([table, Spacer(1, 8)]))
        self.markdown.append("| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n" +
                             "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows) + "\n")

    def figure(self, path, caption):
        from reportlab.platypus import Image, KeepTogether, Paragraph
        from PIL import Image as PILImage
        with PILImage.open(path) as picture:
            width, height = picture.size
        self.flow.append(KeepTogether([Image(str(path), width=492, height=492 * height / width),
                                       Paragraph(escape(caption), self.styles["ReportSmall"])]))
        relative = "../../" + path.relative_to(ROOT).as_posix()
        self.markdown.append(f"![{caption}]({relative})\n")

    def code(self, value):
        from reportlab.platypus import Preformatted, KeepTogether
        self.flow.append(KeepTogether([Preformatted(value, self.styles["ReportCode"])]))
        self.markdown.append("```bash\n" + value + "\n```\n")

    def save(self):
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate
        output = ROOT / "output/pdf/project2_report.pdf"
        output.parent.mkdir(parents=True, exist_ok=True)

        def footer(canvas, document):
            canvas.saveState()
            canvas.setStrokeColor(self.colors.HexColor("#c6d3dd"))
            canvas.line(54, 43, 558, 43)
            canvas.setFont("DejaVu", 7.5)
            canvas.setFillColor(self.colors.HexColor("#476073"))
            canvas.drawString(54, 30, "Dhrumil | Biometrics Project 2 | QMagFace")
            canvas.drawRightString(558, 30, str(document.page))
            canvas.restoreState()

        document = SimpleDocTemplate(str(output), pagesize=letter, rightMargin=54, leftMargin=54,
                                     topMargin=44, bottomMargin=57, title="Quality-Aware Face Verification Under Image Degradation",
                                     author="Dhrumil", pageCompression=1)
        document.build(self.flow, onFirstPage=footer, onLaterPages=footer)
        (ROOT / "docs/project2/final_report.md").write_text("\n".join(self.markdown), encoding="utf-8")
        print(output)


def main():
    benchmark = read("results/project2/author_benchmark/summary.json")
    manifest = read("results/project2/robustness/manifest.json")
    if benchmark["status"] != "completed" or manifest["status"] != "completed":
        raise ValueError("Both measured experiments must be completed before report generation")
    result = read("results/project2/robustness/evaluation.json")
    # Verification is deliberately a prerequisite; a report must not endorse stale outputs.
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts/check_project2.py")], cwd=ROOT, check=True)
    quality = plots(result, ROOT / "results/project2/report_figures")
    rows = {(r["condition"], r["method"]): r for r in result["aggregates"]}
    paired = {r["condition"]: r for r in result["paired_changes"] if r["fold"] is None}
    a, b = rows[("original", "magface_cosine")], rows[("original", "qmagface")]
    severe_a, severe_b = rows[("resolution_14", "magface_cosine")], rows[("resolution_14", "qmagface")]
    bm = benchmark["methods"]
    improved = [LABELS[c] for c in LABELS if paired[c]["net_correct_gain"] > 0]
    regressed = [LABELS[c] for c in LABELS if paired[c]["net_correct_gain"] < 0]
    tied = [LABELS[c] for c in LABELS if paired[c]["net_correct_gain"] == 0]
    report = Report()
    report.title("Quality-Aware Face Verification Under Image Degradation")
    report.paragraph("Reproducing and Investigating QMagFace | Dhrumil | Biometrics Project 2 | October 18, 2026")
    report.heading("Abstract")
    report.paragraph(
        "This project extends an existing face-verification degradation study by reproducing QMagFace, "
        "a published comparison rule that uses raw MagFace feature magnitude as image quality. "
        f"On author-provided LFW features, the official evaluation yields {bm['magface_cosine']['accuracy_percent']:.4f}% "
        f"for cosine scoring and {bm['qmagface']['accuracy_percent']:.4f}% for QMagFace, with 5,994 of 6,000 pairs covered. "
        "The independent scorer agrees with the official implementation within 5e-15. "
        "A separate ten-condition investigation holds the network and pair set fixed, degrades only the probe, "
        "and calibrates each scoring method exclusively on original-quality training folds. "
        f"Original-image accuracy is {percent(a['mean_accuracy'])}% for cosine and {percent(b['mean_accuracy'])}% "
        f"for QMagFace, on {a['scored_pairs']:,} scored pairs. "
        f"At 14-pixel resolution, QMagFace raises accuracy from {percent(severe_a['mean_accuracy'])}% to "
        f"{percent(severe_b['mean_accuracy'])}%, but its AUC falls from {severe_a['pooled_auc']:.6f} to "
        f"{severe_b['pooled_auc']:.6f}. This is an operating-point benefit under the frozen calibration policy, "
        "not evidence of universally better score ranking. Backbone training is not reproduced.")
    report.heading("1. Problem and selected paper")
    report.paragraph(
        "The research question is whether adding image quality to comparison scoring improves face verification under "
        "resolution loss, Gaussian blur and reduced brightness relative to cosine similarity of exactly the same embeddings. "
        "The selected method is QMagFace: Simple and Accurate Quality-Aware Face Recognition, by Terhorst et al., "
        "published at IEEE/CVF WACV 2023 [1]. Its public implementation, pretrained network and feature artifacts make "
        "the comparison method independently testable. The project reproduces scoring and a benchmark protocol, "
        "then investigates controlled post-alignment degradation.")
    report.paragraph(
        "Project 1's VGGFace2-pretrained FaceNet system remains a historical reference: its recorded mean-fold accuracy "
        "was 99.167% on original images and 89.982% at 20 x 20 probe resolution. It used a different backbone and "
        "160-pixel bounding-box crops. Those numbers are not a matched estimate of the QMagFace scoring effect. "
        "The principal comparison here is MagFace cosine versus QMagFace on identical raw MagFace embeddings; "
        "Project 1 code, results and history are preserved.")
    report.heading("2. Reproduced method and implementation")
    report.paragraph(
        "For raw reference and probe embeddings e_r and e_p, let q_r and q_p be their L2 norms and s the cosine "
        "similarity of their unit-normalized versions. QMagFace applies the following comparison function [1,2]:")
    report.code("q_r = ||e_r||_2;  q_p = ||e_p||_2\ns = dot(e_r/q_r, e_p/q_p)\nscore = s + min(0, beta*s - alpha) * min(q_r, q_p)\nalpha = 0.077428;  beta = 0.125926")
    report.paragraph(
        "The coefficients remain fixed at the published QMagFace-100 values; we do not refit them on LFW or choose them "
        "per degradation. The adjusted score is not a probability and may be below -1. For a fixed cosine score "
        "below the weight cutoff, higher minimum quality causes a larger negative adjustment. The rule is therefore "
        "not a simple bonus added to every blurry image. Each scoring method needs its own decision threshold.")
    report.paragraph(
        "The audited MagFace iResNet100 checkpoint was trained on MS1MV2 [3]. It consumes aligned 112 x 112 BGR "
        "pixels divided by 255, with one forward pass and no flip aggregation. All 925 backbone entries are checked "
        "during safe weights-only loading. Raw 512-dimensional features, raw norms and normalized features are "
        "cached separately. The adapter matches the pinned author network bit for bit on two supplied test crops; "
        "a 16-image development pilot with no evaluation-image-path overlap also verified finite raw outputs and CPU/MPS agreement. "
        "These checks validate implementation compatibility, not recognition performance by themselves.")
    report.heading("3. Author-feature benchmark reproduction")
    report.table(["Method", "Paper (%)", "Measured (%)", "Gap (pp)", "FM / FNM"], [
        [name, f"{bm[key]['published_accuracy_percent']:.2f}", f"{bm[key]['accuracy_percent']:.4f}",
         f"{bm[key]['gap_percentage_points']:+.4f}", f"{bm[key]['false_matches']} / {bm[key]['false_nonmatches']}"]
        for key, name in (("magface_cosine", "MagFace cosine"), ("qmagface", "QMagFace"))], [115, 80, 95, 85, 117])
    report.paragraph(
        "Published comparison targets were checked in Table 2 of the final WACV proceedings paper. "
        "The authors' archive contains 13,227 raw LFW feature rows. Six requested pairs lack features: four genuine "
        "and two impostor. Coverage is 99.9%, leaving 2,996 genuine and 2,998 impostor pairs. The official loader "
        "excludes these pairs and then applies contiguous, unshuffled ten-fold KFold to the surviving list, so its "
        "fold membership differs from the original 6,000-pair fold assignments. Every exclusion is exported.")
    report.paragraph(
        "The official metric uses 600 thresholds spanning the global observed score minimum and maximum, "
        "accepts score > threshold and chooses the first threshold on a calibration tie. This was retained to "
        "reproduce the released evaluation, including its use of held-out score extrema to form the grid. "
        "It is distinct from the stricter calibration protocol used in the investigation. QMagFace corrects five "
        "cosine errors and introduces none in this run; ten pairs are wrong for both. The measured QMagFace result "
        "rounds to the published 99.83%, while MagFace's baseline is slightly higher than the published 99.73%. "
        "This is a close reproduction with explicit coverage/protocol qualifications, not exact recovery of every paper result.")
    report.heading("4. Controlled investigation protocol")
    report.paragraph(
        "The original LFW funneled images, ordered 6,000-pair definitions and ten predefined folds are reused. "
        "A facenet-pytorch MTCNN selects the highest-confidence face containing the image center with probability "
        "at least 0.90. Five landmarks are fitted to a fixed ArcFace 112-pixel similarity template; Pillow "
        "bilinear affine sampling uses a black border. This is a separately documented alignment pipeline and "
        "does not claim to reconstruct the authors' benchmark alignment. Alignment is performed once before "
        "degradation, so the experiment measures post-alignment recognition rather than detection robustness.")
    report.table(["Family", "Levels", "Operation"], [
        ["Original", "112 x 112", "No synthetic degradation"],
        ["Resolution", "56, 28, 14 pixels", "Bilinear downsample, then restore to 112"],
        ["Gaussian blur", "sigma 0.7, 1.4, 2.1", "Separable sampled Gaussian; radius ceil(3*sigma); reflect border"],
        ["Brightness", "0.75, 0.50, 0.25", "Multiply encoded RGB, without gamma correction or added noise"],
    ], [90, 125, 277])
    report.paragraph(
        "Each condition starts independently from the original aligned probe crop; references remain original. "
        "Blur and brightness use float64 arithmetic, clipping and nearest-even uint8 quantization. Both scorers "
        "receive identical embeddings and pairs. For each held-out fold, each method maximizes accuracy on "
        "successful original-quality pairs in the other nine folds. Threshold candidates are score midpoints "
        "and finite accept-all/reject-all endpoints; ties choose the largest threshold and acceptance uses >=. "
        "The selected threshold is fixed for all held-out degradation conditions. No degraded test labels tune "
        "thresholds, model settings or coefficients. The full grid was frozen after the development pilot and "
        "before test inference.")
    report.paragraph(
        f"Alignment succeeds for {manifest['alignment']['requested_images'] - manifest['alignment']['failed_images']:,} "
        f"of {manifest['alignment']['requested_images']:,} unique evaluation images. "
        f"Each condition requests {a['requested_pairs']:,} pairs and scores {a['scored_pairs']:,}: "
        f"{a['genuine_pairs']:,} genuine and {a['impostor_pairs']:,} impostor. "
        "Failures remain in predictions, per-label/fold coverage and reject-on-failure summaries. "
        "The following recognition rates condition on successful preprocessing; coverage is reported separately.")
    report.heading("5. Measured robustness results")
    report.table(["Condition", "Cosine accuracy (%)", "QMagFace accuracy (%)", "Delta (pp)", "Net corrected"], [
        [LABELS[c], percent(rows[(c, "magface_cosine")]["mean_accuracy"]), percent(rows[(c, "qmagface")]["mean_accuracy"]),
         f"{100 * (rows[(c, 'qmagface')]['mean_accuracy'] - rows[(c, 'magface_cosine')]['mean_accuracy']):+.3f}",
         f"{paired[c]['net_correct_gain']:+d}"] for c in LABELS], [112, 105, 105, 80, 90])
    report.paragraph(
        "Accuracy is the equally weighted mean of the ten original folds; pp means percentage points. "
        "Net corrected is the number of cosine errors corrected by QMagFace minus newly introduced errors on "
        "the same pairs. Counts and fold means have slightly different weighting because one fold has an exclusion. "
        "Fold SD and SE are saved as descriptive summaries, not confidence intervals or significance tests.", small=True)
    report.figure(ROOT / "results/project2/report_figures/report_accuracy.png",
                  "Figure 1. Fixed-original-threshold accuracy across independently degraded probes. Each family includes the same original control.")
    report.paragraph(
        "QMagFace has a positive net decision gain for " + (", ".join(improved) if improved else "none of the conditions") +
        "; a negative gain for " + (", ".join(regressed) if regressed else "none") +
        "; and equal total errors for " + (", ".join(tied) if tied else "none") + ". "
        "An equal total can still hide changed individual decisions; the exported paired-change table retains "
        "both corrected and introduced errors for every fold and condition.")
    report.table(["Condition", "Cosine FM / FNM", "QMagFace FM / FNM", "Cosine FMR / FNMR (%)", "QMagFace FMR / FNMR (%)"], [
        [LABELS[c], f"{rows[(c,'magface_cosine')]['false_matches']} / {rows[(c,'magface_cosine')]['false_nonmatches']}",
         f"{rows[(c,'qmagface')]['false_matches']} / {rows[(c,'qmagface')]['false_nonmatches']}",
         f"{percent(rows[(c,'magface_cosine')]['pooled_fmr'])} / {percent(rows[(c,'magface_cosine')]['pooled_fnmr'])}",
         f"{percent(rows[(c,'qmagface')]['pooled_fmr'])} / {percent(rows[(c,'qmagface')]['pooled_fnmr'])}"] for c in LABELS], [104, 88, 96, 102, 102])
    report.paragraph(
        "FM and FNM are pooled false matches and false nonmatches. FMR and FNMR use the pooled impostor "
        "and genuine denominators, respectively. The counts make tradeoffs visible when an accuracy improvement "
        "is partly obtained by accepting more pairs. All rates use the fixed original-calibrated thresholds.", small=True)
    for condition, name in (("resolution_14", "Severe resolution loss"), ("blur_2.1", "Severe blur"), ("brightness_0.25", "Severe brightness reduction")):
        x, y = rows[(condition, "magface_cosine")], rows[(condition, "qmagface")]
        change = paired[condition]
        report.paragraph(
            f"{name}: cosine accuracy is {percent(x['mean_accuracy'])}% and QMagFace accuracy is "
            f"{percent(y['mean_accuracy'])}%. Quality-aware scoring corrects {change['qmagface_only_correct']} "
            f"cosine errors and introduces {change['cosine_only_correct']}. Pooled FMR changes from "
            f"{percent(x['pooled_fmr'])}% to {percent(y['pooled_fmr'])}%, while FNMR changes from "
            f"{percent(x['pooled_fnmr'])}% to {percent(y['pooled_fnmr'])}%. "
            f"Relative to its own original condition, cosine loses "
            f"{100*(a['mean_accuracy']-x['mean_accuracy']):.3f} accuracy points and QMagFace loses "
            f"{100*(b['mean_accuracy']-y['mean_accuracy']):.3f} points.")
    report.figure(ROOT / "results/project2/report_figures/report_quality.png",
                  "Figure 2. Raw MagFace probe norms across degradation, averaged over scored pairs; repeated images are pair-weighted.")
    report.paragraph(
        f"The pair-weighted mean raw probe norm is {quality['original']:.3f} on original images, "
        f"{quality['resolution_14']:.3f} at 14 pixels, {quality['blur_2.1']:.3f} at blur sigma 2.1 and "
        f"{quality['brightness_0.25']:.3f} at brightness 0.25. "
        "These observations quantify how the pretrained quality estimate responds to our transformations. "
        "They do not prove that feature magnitude orders every possible real-world image degradation correctly. "
        "Reference-quality distributions remain fixed because reference images are never transformed.")
    report.table(["Condition", "Cosine AUC", "QMagFace AUC", "Cosine EER (%)", "QMagFace EER (%)"], [
        [LABELS[c], f"{rows[(c,'magface_cosine')]['pooled_auc']:.6f}", f"{rows[(c,'qmagface')]['pooled_auc']:.6f}",
         percent(rows[(c,'magface_cosine')]['pooled_eer_interpolated']),
         percent(rows[(c,'qmagface')]['pooled_eer_interpolated'])] for c in LABELS], [112, 95, 95, 95, 95])
    report.paragraph(
        "Pooled ROC AUC and interpolated EER describe score ranking across all scored pairs and threshold sweeps. "
        "Their thresholds are not used for held-out predictions. The full ROC/DET curves, genuine/impostor "
        "score distributions, raw-quality distributions and original-condition paired changes accompany the report.", small=True)
    report.paragraph(
        f"The severe-resolution result illustrates why fixed-threshold accuracy and ranking must be separated. "
        f"At 14 pixels, QMagFace gains {100*(severe_b['mean_accuracy']-severe_a['mean_accuracy']):.3f} accuracy points, "
        f"yet AUC decreases from {severe_a['pooled_auc']:.6f} to {severe_b['pooled_auc']:.6f} and EER increases "
        f"from {percent(severe_a['pooled_eer_interpolated'])}% to {percent(severe_b['pooled_eer_interpolated'])}%. "
        "Thus the gain at the original-calibrated operating point does not show globally better discrimination. "
        "For the active quality-adjustment branch and minimum norm m, the score is (1 + beta*m)*s - alpha*m. "
        "A fixed adjusted-score threshold T corresponds to a cosine cutoff (T + alpha*m)/(1 + beta*m). "
        "At the negative thresholds measured here, this cutoff decreases as quality decreases. This explains "
        "how the rule can recover genuine acceptances under degradation while allowing additional false matches; "
        "it does not recover the visual information removed by downsampling.")
    report.heading("6. Limitations and interpretation")
    report.paragraph(
        "This project reuses a pretrained backbone and fixed published coefficients; it does not reproduce "
        "MagFace training, QMagFace parameter fitting or every paper dataset. Author-feature reproduction and "
        "our aligned-image investigation have different coverage and threshold protocols, so their absolute "
        "accuracies must not be conflated. The 14-pixel condition is a relative-scale extension of Project 1's "
        "20-pixel condition, not an identical crop or whole-system comparison.")
    report.paragraph(
        "LFW contains repeated subjects and images, and the ten calibration sets overlap. The study therefore "
        "reports descriptive paired changes rather than treating folds or conditions as independent experimental "
        "replicates. A small gain of a few errors is not proof of a population-level advantage. With about 3,000 "
        "impostor comparisons, a single false match is approximately 0.033 percentage points of FMR; the sample "
        "cannot establish extreme low-FMR performance or make zero observed errors imply zero risk.")
    report.paragraph(
        "Synthetic brightness scaling is not a physical low-light camera model: it adds neither sensor noise "
        "nor exposure-dependent artifacts. Fixed alignment excludes degradation-induced detection failures. "
        "Training/evaluation identity overlap was not independently audited. No demographic fairness analysis "
        "or deployment recommendation is supported by these data. The publicly documented research artifacts "
        "are used locally; model weights, faces and large feature caches are excluded from the submission repository.")
    report.heading("7. Conclusions and reproducibility")
    report.paragraph(
        "The comparison function and author-feature benchmark have been reproduced with numerical agreement "
        "and explicit protocol differences. The robustness experiment isolates the scoring rule by holding "
        "the feature extractor and pairs fixed, with each method's separately calibrated threshold held fixed across conditions. "
        f"QMagFace improves fixed-threshold accuracy in {len(improved)} of ten conditions and ties in {len(tied)}; "
        f"its largest gain is {100*(severe_b['mean_accuracy']-severe_a['mean_accuracy']):.3f} points at 14 pixels. "
        f"However, even there it rejects {percent(severe_b['pooled_fnmr'])}% of genuine pairs, and its ranking metrics "
        "are worse than cosine. Blur shows a smaller gain, while brightness effects are only zero or one net "
        "corrected decision. Quality-aware scoring is useful for this frozen-threshold policy, but the results "
        "do not support a blanket claim that it solves image degradation or dominates cosine at every operating point.")
    report.paragraph(
        "The reference environment is Python 3.13.7, NumPy 2.5.3, PyTorch 2.8.0, torchvision 0.23.0, "
        "facenet-pytorch 2.5.3 and Pillow 11.2.1 on macOS arm64. Inference used MPS with batches of 32 and "
        "two CPU threads for alignment. A separate benchmark environment adds scipy 1.16.2 and scikit-learn "
        "1.7.2 without replacing Project 1 dependencies. The checkpoint's observed SHA-256 is "
        "cfeba792dada6f1f30d1e118aff077d493dd95dd76c77c30f57f90fd0164ad58; this is a local observed hash, "
        "not a publisher-signed checksum.")
    report.code(".venv/bin/python -m unittest discover -s tests -v\n.venv/bin/python scripts/check_project2.py\n.venv/bin/python scripts/project2_reanalyze.py\n.venv/bin/python -m src.project2_experiment \\\n  --stage run --device mps --grid full --batch-size 32")
    report.paragraph(
        "Use --device cpu on systems without MPS. The detailed acquisition and clean-environment instructions "
        "are in docs/project2/reproduce.md; the artifact audit, frozen protocol, AI-assistance record and experiment "
        "log document decisions and provenance. Machine-readable scores preserve failed attempts, original folds, "
        "both raw qualities and both scores. Predictions retain method-specific thresholds. Manifests record "
        "source/output hashes, software versions, checkpoint identity and timing. The repository "
        "checker re-evaluates saved pairs and detects corrupt, stale or mismatched outputs; optional cache "
        "checks reconstruct scores from raw features. The submission retains the actual .git history.")
    report.heading("References")
    report.paragraph("[1] P. Terhorst, M. Ihlefeld, M. Huber, N. Damer, F. Kirchbuchner, K. Raja and A. Kuijper. "
                     "QMagFace: Simple and Accurate Quality-Aware Face Recognition. IEEE/CVF WACV, 2023, pp. 3484-3494. "
                     "https://openaccess.thecvf.com/content/WACV2023/html/Terhorst_QMagFace_Simple_and_Accurate_Quality-Aware_Face_Recognition_WACV_2023_paper.html", small=True)
    report.paragraph("[2] Authors' QMagFace code and artifact links. https://github.com/pterhoer/QMagFace "
                     "Commit 395797145da1b1b263cbe7fb1564c6ea91f55b23 (accessed October 18, 2026).", small=True)
    report.paragraph("[3] Q. Meng, S. Zhao, Z. Huang and F. Zhou. MagFace: A Universal Representation for Face "
                     "Recognition and Quality Assessment. IEEE/CVF CVPR, 2021. https://github.com/IrvingMeng/MagFace "
                     "Commit 99bae614ac2643b9694bf18e0c5645272ff6acfa.", small=True)
    report.paragraph("[4] Labeled Faces in the Wild, funneled images and official pair files, distributed through "
                     "the scikit-learn/figshare mirror. Source URLs and verified dataset hashes are recorded in "
                     "src/lfw_dataset.py and docs/dataset_research.md.", small=True)
    report.save()
    names = ["docs/project2/final_report.md", "output/pdf/project2_report.pdf",
             "results/project2/report_figures/report_accuracy.png", "results/project2/report_figures/report_quality.png"]
    inputs = ["results/project2/author_benchmark/summary.json", "results/project2/robustness/manifest.json",
              "results/project2/robustness/evaluation.json", "scripts/build_project2_report.py"]
    digest = lambda name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    (ROOT / "results/project2/report_manifest.json").write_text(json.dumps({
        "status": "generated", "input_sha256": {name: digest(name) for name in inputs},
        "output_sha256": {name: digest(name) for name in names},
        "scope": "Written report and two scientific figures built from verified measured outputs; visual PDF review recorded separately",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
