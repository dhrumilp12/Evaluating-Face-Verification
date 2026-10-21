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
            canvas.drawString(54, 30, "Dhrumil Patel | Biometrics Project 2 | QMagFace")
            canvas.drawRightString(558, 30, str(document.page))
            canvas.restoreState()

        document = SimpleDocTemplate(str(output), pagesize=letter, rightMargin=54, leftMargin=54,
                                     topMargin=44, bottomMargin=57, title="Quality-Aware Face Verification Under Image Degradation",
                                     author="Dhrumil Patel", pageCompression=1)
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
    tied = [LABELS[c] for c in LABELS if paired[c]["net_correct_gain"] == 0]
    report = Report()
    report.title("Quality-Aware Face Verification Under Image Degradation")
    report.paragraph("Reproducing and Investigating QMagFace | Dhrumil Patel | Biometrics Project 2 | October 21, 2026")
    report.heading("Abstract")
    report.paragraph(
        "This study tests whether QMagFace improves face verification when images lose resolution, become blurred "
        "or become darker. QMagFace adjusts cosine similarity using raw MagFace feature magnitude as an image-quality estimate. "
        f"On the authors' released LFW features, accuracy is {bm['magface_cosine']['accuracy_percent']:.4f}% "
        f"for cosine scoring and {bm['qmagface']['accuracy_percent']:.4f}% for QMagFace, using 5,994 of 6,000 pairs. "
        "The independent scorer agrees with the official implementation within 5e-15. "
        "A separate experiment tests ten conditions using the same network and pairs for both scorers. "
        "Only probe images are degraded, and thresholds are calibrated on original-quality training folds. "
        f"Original-image accuracy is {percent(a['mean_accuracy'])}% for cosine and {percent(b['mean_accuracy'])}% "
        f"for QMagFace, on {a['scored_pairs']:,} scored pairs. "
        f"At 14-pixel resolution, QMagFace raises accuracy from {percent(severe_a['mean_accuracy'])}% to "
        f"{percent(severe_b['mean_accuracy'])}%, but its AUC falls from {severe_a['pooled_auc']:.6f} to "
        f"{severe_b['pooled_auc']:.6f}. QMagFace therefore helps at the original-image decision thresholds, "
        "even though overall score separation worsens under severe resolution loss. The study uses a pretrained network.")
    report.heading("1. Problem and selected paper")
    report.paragraph(
        "Project 1 showed that image degradation can substantially reduce face-verification accuracy. "
        "Project 2 asks whether using image quality in the comparison score can reduce that loss. "
        "The selected paper is QMagFace: Simple and Accurate Quality-Aware Face Recognition, by Terhorst et al. "
        "(IEEE/CVF WACV 2023) [1]. Its released code, pretrained model and features allow the scoring method "
        "to be reproduced without retraining the network. The work first reproduces the authors' LFW evaluation, "
        "then compares QMagFace with cosine similarity under controlled degradation.")
    report.paragraph(
        "The earlier VGGFace2-pretrained FaceNet system achieved mean-fold accuracy of 99.167% on original "
        "images and 89.982% at 20 x 20 probe resolution. Those results provide context, but its network and "
        "160-pixel bounding-box crops differ from this study. To isolate the effect of the scoring rule, "
        "the main comparison uses cosine and QMagFace on identical raw MagFace embeddings. Project 1's "
        "code, results and Git history remain in the repository.")
    report.heading("2. Reproduced method and implementation")
    report.paragraph(
        "For raw reference and probe embeddings e_r and e_p, let q_r and q_p be their L2 norms and s the cosine "
        "similarity of their unit-normalized versions. QMagFace applies the following comparison function [1,2]:")
    report.code("q_r = ||e_r||_2;  q_p = ||e_p||_2\ns = dot(e_r/q_r, e_p/q_p)\nscore = s + min(0, beta*s - alpha) * min(q_r, q_p)\nalpha = 0.077428;  beta = 0.125926")
    report.paragraph(
        "The published QMagFace-100 coefficients are used throughout, without refitting on LFW or by degradation "
        "condition. The adjustment is nonpositive and can make the score less than -1; the result is a score, "
        "not a probability. Below the weight cutoff, a higher minimum quality produces a larger subtraction "
        "at the same cosine similarity. Cosine and QMagFace therefore need separate decision thresholds.")
    report.paragraph(
        "The MagFace iResNet100 checkpoint was trained on MS1MV2 [3]. Input images are aligned to 112 x 112 "
        "pixels, converted to BGR and divided by 255. Inference uses one forward pass without flipped-image "
        "aggregation. Loading checks all 925 backbone entries with weights-only deserialization. The cache "
        "stores raw 512-dimensional features, their norms and normalized features separately. "
        "The adapter produced bitwise-identical output to the pinned author network on two test crops. "
        "A 16-image development pilot, with no image-path overlap with evaluation, also checked finite outputs "
        "and CPU/MPS agreement before the recognition experiments.")
    report.heading("3. Author-feature benchmark reproduction")
    report.table(["Method", "Paper (%)", "Measured (%)", "Gap (pp)", "FM / FNM"], [
        [name, f"{bm[key]['published_accuracy_percent']:.2f}", f"{bm[key]['accuracy_percent']:.4f}",
         f"{bm[key]['gap_percentage_points']:+.4f}", f"{bm[key]['false_matches']} / {bm[key]['false_nonmatches']}"]
        for key, name in (("magface_cosine", "MagFace cosine"), ("qmagface", "QMagFace"))], [115, 80, 95, 85, 117])
    report.paragraph(
        "The paper accuracies above come from Table 2 of the final WACV proceedings paper. "
        "The authors' archive contains 13,227 raw LFW feature rows. Six requested pairs lack features: four genuine "
        "and two impostor. Coverage is 99.9%, leaving 2,996 genuine and 2,998 impostor pairs. The official loader "
        "excludes these pairs and then applies contiguous, unshuffled ten-fold KFold to the surviving list, so its "
        "fold membership differs from the original 6,000-pair assignments. The saved exclusion file lists all six pairs.")
    report.paragraph(
        "The released evaluator tests 600 thresholds between the global score minimum and maximum. It accepts "
        "score > threshold and takes the first threshold when calibration accuracies tie. Reproduction retains "
        "these conventions, including the use of held-out score extrema to construct the grid. The controlled "
        "experiment below uses a different calibration protocol. QMagFace corrects five cosine errors and "
        "introduces none; both methods misclassify ten pairs. Its measured accuracy rounds to the published "
        "99.83%, while the cosine baseline is slightly above the paper's 99.73%. The result is close to the "
        "published benchmark, subject to the six missing pairs and the evaluator conventions described here.")
    report.heading("4. Controlled investigation protocol")
    report.paragraph(
        "The investigation uses the original LFW funneled images, ordered 6,000-pair list and ten predefined folds. "
        "A facenet-pytorch MTCNN selects the highest-confidence face containing the image center with probability "
        "at least 0.90. Five landmarks are fitted to a fixed ArcFace 112-pixel similarity template; Pillow "
        "bilinear affine sampling uses a black border. These crops may differ from those used for the authors' "
        "released features. Alignment runs once on each original image, before degradation, so detection "
        "failures caused by degraded input are outside this experiment.")
    report.table(["Family", "Levels", "Operation"], [
        ["Original", "112 x 112", "No synthetic degradation"],
        ["Resolution", "56, 28, 14 pixels", "Bilinear downsample, then restore to 112"],
        ["Gaussian blur", "sigma 0.7, 1.4, 2.1", "Separable sampled Gaussian; radius ceil(3*sigma); reflect border"],
        ["Brightness", "0.75, 0.50, 0.25", "Multiply encoded RGB, without gamma correction or added noise"],
    ], [90, 125, 277])
    report.paragraph(
        "Each transformation starts from the original aligned probe crop; reference images stay unchanged. "
        "Blur and brightness use float64 arithmetic, clipping and nearest-even uint8 quantization. The two "
        "scorers use the same embeddings and pairs. For each held-out fold, each method chooses its threshold "
        "by maximizing accuracy on successful original-quality pairs from the other nine folds. Candidates "
        "are score midpoints and finite accept-all/reject-all endpoints; ties select the largest threshold, "
        "and acceptance uses >=. That threshold is then fixed for every condition in the held-out fold. "
        "Test scores and labels do not select thresholds, model settings or coefficients. The condition grid "
        "was finalized after the development pilot, before test inference.")
    report.paragraph(
        f"Alignment succeeds for {manifest['alignment']['requested_images'] - manifest['alignment']['failed_images']:,} "
        f"of {manifest['alignment']['requested_images']:,} unique evaluation images. "
        f"Each condition requests {a['requested_pairs']:,} pairs and scores {a['scored_pairs']:,}: "
        f"{a['genuine_pairs']:,} genuine and {a['impostor_pairs']:,} impostor. "
        "Failed attempts remain in the prediction exports, coverage counts and separate reject-on-failure "
        "summaries. The recognition rates below use only successfully preprocessed pairs.")
    report.heading("5. Measured robustness results")
    report.table(["Condition", "Cosine accuracy (%)", "QMagFace accuracy (%)", "Delta (pp)", "Net corrected"], [
        [LABELS[c], percent(rows[(c, "magface_cosine")]["mean_accuracy"]), percent(rows[(c, "qmagface")]["mean_accuracy"]),
         f"{100 * (rows[(c, 'qmagface')]['mean_accuracy'] - rows[(c, 'magface_cosine')]['mean_accuracy']):+.3f}",
         f"{paired[c]['net_correct_gain']:+d}"] for c in LABELS], [112, 105, 105, 80, 90])
    report.paragraph(
        "Accuracy is the mean of the ten original folds, weighted equally; pp denotes percentage points. "
        "Net corrected counts cosine errors fixed by QMagFace minus new errors on the same pairs. "
        "The counts and fold means weight pairs slightly differently because one fold has an exclusion. "
        "Saved fold SD and SE describe variation across folds; they are not confidence intervals or significance tests.", small=True)
    report.figure(ROOT / "results/project2/report_figures/report_accuracy.png",
                  "Figure 1. Accuracy under probe degradation, using thresholds calibrated on original images. Each family includes the same original control.")
    report.paragraph(
        f"QMagFace reduces the total error count in {len(improved)} conditions and ties in {len(tied)}. "
        "The largest improvement occurs at 14-pixel resolution. Total errors alone can hide cases where "
        "one method fixes some decisions but gets others wrong, so the paired-change exports record both "
        "corrected and introduced errors for every fold and condition.")
    report.table(["Condition", "Cosine FM / FNM", "QMagFace FM / FNM", "Cosine FMR / FNMR (%)", "QMagFace FMR / FNMR (%)"], [
        [LABELS[c], f"{rows[(c,'magface_cosine')]['false_matches']} / {rows[(c,'magface_cosine')]['false_nonmatches']}",
         f"{rows[(c,'qmagface')]['false_matches']} / {rows[(c,'qmagface')]['false_nonmatches']}",
         f"{percent(rows[(c,'magface_cosine')]['pooled_fmr'])} / {percent(rows[(c,'magface_cosine')]['pooled_fnmr'])}",
         f"{percent(rows[(c,'qmagface')]['pooled_fmr'])} / {percent(rows[(c,'qmagface')]['pooled_fnmr'])}"] for c in LABELS], [104, 88, 96, 102, 102])
    report.paragraph(
        "FM and FNM count false matches and false nonmatches across all scored pairs. FMR divides FM by "
        "the impostor count; FNMR divides FNM by the genuine count. All decisions use thresholds calibrated "
        "on original images. A reduction in false nonmatches can come with an increase in false matches.", small=True)
    change = paired["resolution_14"]
    report.paragraph(
        f"At 14-pixel resolution, accuracy rises from {percent(severe_a['mean_accuracy'])}% with cosine to "
        f"{percent(severe_b['mean_accuracy'])}% with QMagFace. QMagFace corrects {change['qmagface_only_correct']} "
        f"cosine errors but introduces {change['cosine_only_correct']} new errors. False nonmatches fall from "
        f"{severe_a['false_nonmatches']:,} to {severe_b['false_nonmatches']:,}, while false matches rise from "
        f"{severe_a['false_matches']} to {severe_b['false_matches']}. Both methods remain far below their "
        f"original-image accuracy: cosine loses {100*(a['mean_accuracy']-severe_a['mean_accuracy']):.3f} "
        f"percentage points and QMagFace loses {100*(b['mean_accuracy']-severe_b['mean_accuracy']):.3f}.")
    x, y = rows[("blur_2.1", "magface_cosine")], rows[("blur_2.1", "qmagface")]
    change = paired["blur_2.1"]
    report.paragraph(
        f"The strongest blur has a smaller effect. At sigma 2.1, cosine accuracy is {percent(x['mean_accuracy'])}% "
        f"and QMagFace accuracy is {percent(y['mean_accuracy'])}%. QMagFace fixes {change['qmagface_only_correct']} "
        f"cosine errors and adds {change['cosine_only_correct']}. FNMR drops from {percent(x['pooled_fnmr'])}% "
        f"to {percent(y['pooled_fnmr'])}%, with FMR unchanged at {percent(y['pooled_fmr'])}%. The losses from "
        f"original-image accuracy are {100*(a['mean_accuracy']-x['mean_accuracy']):.3f} and "
        f"{100*(b['mean_accuracy']-y['mean_accuracy']):.3f} percentage points, respectively.")
    x, y = rows[("brightness_0.25", "magface_cosine")], rows[("brightness_0.25", "qmagface")]
    change = paired["brightness_0.25"]
    report.paragraph(
        f"Reducing brightness to 25% leaves accuracy close to the original condition: "
        f"{percent(x['mean_accuracy'])}% for cosine and {percent(y['mean_accuracy'])}% for QMagFace. "
        f"QMagFace corrects {change['qmagface_only_correct']} errors and introduces {change['cosine_only_correct']}, "
        "a net difference of one decision. The false-nonmatch count is unchanged between methods, while "
        f"FMR falls from {percent(x['pooled_fmr'])}% to {percent(y['pooled_fmr'])}%. Each method loses "
        f"{100*(b['mean_accuracy']-y['mean_accuracy']):.3f} percentage points from its own original accuracy.")
    report.figure(ROOT / "results/project2/report_figures/report_quality.png",
                  "Figure 2. Mean raw MagFace probe norms under degradation. Each scored pair contributes once, so repeated images receive more weight.")
    report.paragraph(
        f"The pair-weighted mean raw probe norm is {quality['original']:.3f} on original images, "
        f"{quality['resolution_14']:.3f} at 14 pixels, {quality['blur_2.1']:.3f} at blur sigma 2.1 and "
        f"{quality['brightness_0.25']:.3f} at brightness 0.25. "
        "The mean norm decreases as degradation becomes stronger within each tested family. This describes "
        "the average response; individual images need not follow the same pattern, and the result does not "
        "establish a quality ranking for all real-world degradations. Reference-quality distributions stay "
        "fixed because reference images are unchanged.")
    report.table(["Condition", "Cosine AUC", "QMagFace AUC", "Cosine EER (%)", "QMagFace EER (%)"], [
        [LABELS[c], f"{rows[(c,'magface_cosine')]['pooled_auc']:.6f}", f"{rows[(c,'qmagface')]['pooled_auc']:.6f}",
         percent(rows[(c,'magface_cosine')]['pooled_eer_interpolated']),
         percent(rows[(c,'qmagface')]['pooled_eer_interpolated'])] for c in LABELS], [112, 95, 95, 95, 95])
    report.paragraph(
        "Pooled ROC AUC and interpolated EER summarize performance across thresholds using all scored pairs. "
        "These threshold sweeps do not select the thresholds used for held-out predictions. Full ROC/DET curves, "
        "score and quality distributions, and paired changes from the original condition are saved with the results.", small=True)
    report.paragraph(
        f"The 14-pixel result needs careful interpretation. QMagFace gains "
        f"{100*(severe_b['mean_accuracy']-severe_a['mean_accuracy']):.3f} accuracy points, "
        f"yet AUC decreases from {severe_a['pooled_auc']:.6f} to {severe_b['pooled_auc']:.6f} and EER increases "
        f"from {percent(severe_a['pooled_eer_interpolated'])}% to {percent(severe_b['pooled_eer_interpolated'])}%. "
        "The improvement is specific to the decision thresholds calibrated on original images. When the "
        "quality adjustment is active and m is the minimum feature norm, the score becomes "
        "(1 + beta*m)*s - alpha*m. A threshold T on this score is equivalent to a cosine cutoff of "
        "(T + alpha*m)/(1 + beta*m). At the negative thresholds measured here, that cutoff falls as quality "
        "decreases. More genuine pairs can therefore be accepted, at the cost of possible additional false "
        "matches. This explains the accuracy gain without implying that downsampling's lost information "
        "has been recovered.")
    report.heading("6. Limitations and interpretation")
    report.paragraph(
        "The reproduction covers the scoring function and the LFW benchmark with released features. It does "
        "not repeat MagFace training, QMagFace parameter fitting or experiments on the paper's other datasets. "
        "Coverage and threshold selection differ between the author-feature benchmark and the local image "
        "experiment, so their absolute accuracies should be interpreted separately. The 14-pixel condition "
        "preserves the relative resolution scale of Project 1's 20-pixel condition, but the crops differ.")
    report.paragraph(
        "LFW repeats subjects and images, and the calibration sets overlap. Fold and condition results are "
        "therefore dependent; the paired changes reported here are descriptive. Differences of a few errors "
        "do not establish a general advantage. With about 3,000 impostor comparisons, one false match changes "
        "FMR by approximately 0.033 percentage points. This sample is too small to assess very low false-match "
        "rates reliably, and observing zero false matches does not establish zero risk.")
    report.paragraph(
        "Multiplying pixel values simulates dimming but omits sensor noise and exposure artifacts from real "
        "low-light capture. Fixed alignment also leaves out detection failures caused by degradation. "
        "Training/evaluation identity overlap and demographic fairness were not audited, so the study does "
        "not establish deployment suitability. Downloaded research artifacts are used locally; the submission "
        "excludes face images, model weights and large feature caches.")
    report.heading("7. Conclusions and reproducibility")
    report.paragraph(
        "The independent scorer agrees numerically with the authors' implementation, and the LFW result "
        "closely matches the published QMagFace accuracy under the released evaluator's conventions. "
        "The controlled experiment compares scoring rules on the same features and pairs, keeping each "
        "method's original-image thresholds fixed across degradations. "
        f"QMagFace improves fixed-threshold accuracy in {len(improved)} of ten conditions and ties in {len(tied)}; "
        f"its largest gain is {100*(severe_b['mean_accuracy']-severe_a['mean_accuracy']):.3f} points at 14 pixels. "
        f"However, even there it rejects {percent(severe_b['pooled_fnmr'])}% of genuine pairs, and its ranking metrics "
        "are worse than cosine. The blur benefit is smaller, and brightness changes produce only zero or one "
        "net corrected decision. The main finding is that QMagFace can reduce false rejections when thresholds "
        "set on original images are reused on degraded images. Severe resolution loss remains a substantial "
        "problem, and the results do not show an advantage at every decision threshold.")
    report.paragraph(
        "The reference environment is Python 3.13.7, NumPy 2.5.3, PyTorch 2.8.0, torchvision 0.23.0, "
        "facenet-pytorch 2.5.3 and Pillow 11.2.1 on macOS arm64. Inference used MPS with batches of 32 and "
        "two CPU threads for alignment. A separate benchmark environment adds scipy 1.16.2 and scikit-learn "
        "1.7.2 while keeping Project 1 dependencies intact. The downloaded checkpoint's SHA-256 is "
        "cfeba792dada6f1f30d1e118aff077d493dd95dd76c77c30f57f90fd0164ad58. This locally recorded hash "
        "identifies the file used; it is not a publisher-signed checksum.")
    report.code(".venv/bin/python -m unittest discover -s tests -v\n.venv/bin/python scripts/check_project2.py\n.venv/bin/python scripts/project2_reanalyze.py\n.venv/bin/python -m src.project2_experiment \\\n  --stage run --device mps --grid full --batch-size 32")
    report.paragraph(
        "Use --device cpu on systems without MPS. Setup and download instructions are in "
        "docs/project2/reproduce.md. The artifact audit, protocol, AI-assistance record and experiment log "
        "explain how the work was carried out. Saved scores include failed attempts, original folds, both "
        "qualities and both scores; predictions include the threshold used for each decision. Manifests "
        "record file hashes, software versions, checkpoint identity and timing. The checker recomputes "
        "metrics from saved pairs, with optional checks that reconstruct scores from raw features. "
        "The submission includes the full Git history.")
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
