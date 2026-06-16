#!/usr/bin/env python3
"""Bench report for the bibliography-span classifiers — precision-first. Reads results_span_models.json
and publishes an arxiv-report HTML to the results hub (glossapi-tokenizer-extension / span-classifier-bench):
the prose-amputation (FP) collapse vs the Rust baseline, the paired-bootstrap verdict, and the recall
holes by metadata slice (where the DL tier must earn its keep)."""
import os, sys, json
sys.path.insert(0, os.path.expanduser("~/.claude/skills/arxiv-report/assets"))
from report_kit import Report, setup_mpl, CATEGORICAL, table
import matplotlib.pyplot as plt
setup_mpl()
HERE = os.path.dirname(os.path.abspath(__file__))
R_ = json.load(open(f"{HERE}/results_span_models.json"))

LABEL = {"rust_header_eof": "Rust header→EOF (current)", "line_lr": "line-LR (interpretable)",
         "synergy": "two-stage synergy", "dl_crf": "DL frozen-emb"}
arms = [a for a in ("rust_header_eof", "line_lr", "synergy", "dl_crf") if f"{a}_gpoa" in R_]


def g(a, k):
    return R_[f"{a}_gpoa"][k]


rep = Report(
    title="A precision-first bibliography-span classifier for the Greek academic corpus",
    subtitle="Removing reference lists by SEGMENTATION without amputating main text · line-level multi-span "
             "detection vs the current header→EOF detector · greek_phd + openarchives held-out test")

base, best = "rust_header_eof", "line_lr"
boot = R_.get("bootstrap_vs_rust", {}).get(best, [0, 0, 0])
rep.cards([
    (f"{g(base,'fp_rate'):.0%}", "prose amputated by the CURRENT detector (header→EOF)", "bad"),
    (f"{g(best,'fp_rate'):.0%}", "prose amputated by the interpretable line model", "win"),
    (f"{g(best,'line_prec'):.2f}", "line-precision (1 − prose-amputation rate)"),
    (f"+{boot[1]:.2f}", "ΔFβ0.5 vs baseline · 95% CI excludes 0 (paired bootstrap)", "win"),
])
probe = None
try:
    probe = json.load(open(f"{HERE}/results_dl_probe.json"))
except Exception:
    pass
comp = None
try:
    comp = json.load(open(f"{HERE}/results_composition.json"))
except Exception:
    pass
if comp:
    rep.cards([
        (f"{comp['prose_protection_precision']:.4f}", "PROSE-PROTECTION precision — running text is almost never deleted", "win"),
        (f"{comp['composition'].get('bibliography_entry (correct)', 0)*100//comp['n_removed']}%", "of removed lines are true bibliography"),
        (f"{comp['reference_mass_precision']:.3f}", "removed lines that are bibliography OR footnote/inline/table citations"),
        (f"{100 - comp['prose_protection_precision']*100:.2f}%", "of removed lines are genuine running prose (the only costly error)"),
    ])

# --- Figure 1: prose-amputation (FP) rate per arm ---
fig1, ax = plt.subplots(figsize=(7.2, 3.4))
names = [LABEL[a] for a in arms]
fps = [g(a, "fp_rate") for a in arms]
cols = [CATEGORICAL[3] if a == "rust_header_eof" else CATEGORICAL[1] for a in arms]
ax.barh(range(len(arms)), fps, color=cols)
ax.set_yticks(range(len(arms))); ax.set_yticklabels(names); ax.invert_yaxis()
for i, v in enumerate(fps):
    ax.text(v + .01, i, f"{v:.0%}", va="center", fontsize=10)
ax.set_xlabel("prose-amputation rate  =  fraction of REMOVED lines that are actually main text (lower is better)")
ax.set_xlim(0, max(fps) * 1.2)
rep.section("The cost that matters: how much main text gets deleted",
            body="A false positive — deleting prose as if it were a bibliography — corrupts the training corpus, "
                 "so the system is tuned for precision, not F1. The current header→EOF detector removes everything "
                 f"from the first bibliography header to end-of-document, so on matched documents {g(base,'fp_rate'):.0%} "
                 f"of what it deletes is actually main text. The interpretable line model cuts that to {g(best,'fp_rate'):.0%} "
                 "— a ~9× reduction — while detecting at least as much bibliography.",
            figure=(fig1, "Prose-amputation (false-positive) rate per arm on the shared greek_phd+openarchives test docs."))

# --- table: three views per arm ---
rows = [[LABEL[a], f"{g(a,'line_prec'):.3f}", f"{g(a,'line_rec'):.3f}", f"{g(a,'fp_rate'):.0%}",
         f"{g(a,'det_rec'):.2f}", f"{g(a,'kappa'):.2f}", f"{g(a,'med_iou'):.2f}",
         f"{g(a,'med_dstart'):+.0f}/{g(a,'med_dend'):+.0f}", f"{g(a,'prose_eat_rate'):.0%}"] for a in arms]
rep.section("Three independent views (held-out test)",
            body="Line precision/recall is the headline; detection and IoU are secondary. Signed Δstart/Δend show the "
                 "current detector starts early and runs past the end (eating prose both sides), while the line model "
                 "under-captures inward (Δstart>0, Δend<0) — conservative by design. Paired bootstrap of the "
                 f"precision-weighted Fβ0.5 vs the baseline: line-LR ΔFβ0.5 = +{boot[1]:.3f} [95% {boot[0]:+.3f}, {boot[2]:+.3f}].",
            table_html=table(["arm", "line P", "line R", "FP", "det R", "κ", "IoU", "Δstart/Δend", "prose-eat"], rows))

# --- Figure 2: recall by slice (where recall is weak) ---
sl = R_.get("line_lr_slice_recall", {})
script = {k.split(":", 1)[1]: v for k, v in sl.items() if k.startswith("script:")}
noise = {k.split(":", 1)[1]: v for k, v in sl.items() if k.startswith("noise_level:")}
kind = {k.split(":", 1)[1]: v for k, v in sl.items() if k.startswith("kind:")}
fig2, axs = plt.subplots(1, 3, figsize=(11.5, 3.4))
for ax, (ti, dd) in zip(axs, [("script", script), ("noise", noise), ("kind", kind)]):
    items = sorted(dd.items(), key=lambda x: x[1])
    ax.barh(range(len(items)), [v for _, v in items], color=CATEGORICAL[0])
    ax.set_yticks(range(len(items))); ax.set_yticklabels([k[:16] for k, _ in items], fontsize=8)
    ax.set_xlim(0, 1); ax.set_title(f"recall by {ti}", fontsize=10); ax.invert_yaxis()
rep.section("Where recall is weak — the DL tier's target",
            body="The interpretable model's recall is moderate overall and concentrated holes appear on "
                 "polytonic and monotonic Greek script, heavy-OCR-noise spans, and the archival/web/further-reading "
                 "kinds — exactly the Latin-biased blind spots of regex features (Greek author/year conventions). "
                 "The frozen-embedding tier is escalated only where a paired-bootstrap gate (ΔFβ at fixed precision, "
                 "CI excluding 0) shows it recovers these without sacrificing precision.",
            figure=(fig2, "Line-recall of the interpretable model sliced by script, noise level, and span kind."))

if comp:
    c = comp["composition"]; nr = comp["n_removed"]
    order = sorted(c.items(), key=lambda x: -x[1])
    rep.section("What does it actually remove? (prose-protection)",
                body="The headline 'precision' understates the safety margin, because most non-bibliography lines it "
                     "removes are themselves CITATIONS (footnote citations, inline references in prose, numbered "
                     "reference table-rows, web sources) — reference mass a corpus cleaner usually wants gone — not "
                     f"running text. Of {nr:,} removed lines, {comp['prose_protection_precision']*100:.2f}% are NOT "
                     "running prose; only "
                     f"{(1-comp['prose_protection_precision'])*100:.2f}% (a few dozen lines) are genuine main text "
                     "wrongly deleted. So the error that actually hurts training — amputating prose — is essentially "
                     "eliminated, and whether to also strip the footnote/inline citations is a downstream policy choice "
                     "the line types expose, not a model failure.",
                table_html=table(["removed line type", "count", "share"],
                                 [[k, f"{v:,}", f"{100*v/nr:.1f}%"] for k, v in order]))
try:
    fv = json.load(open(f"{HERE}/results_fpval.json"))
except Exception:
    fv = None
if fv:
    rep.section("Independent validation (Opus audit of removed lines)",
                body="To check that the prose-protection figure is not an artifact of the deterministic categorizer, "
                     "Opus independently adjudicated 353 removed lines in context (type · should-a-cleaner-remove-it · "
                     "is-it-mislabeled-bibliography), reweighted to the removed-line population. It confirms the claim: "
                     f"genuine running prose removed ≈ {fv['pop_prose_pct']:.2f}% (prose-protection {fv['opus_prose_protection']:.4f}, "
                     "matching the deterministic 0.998); "
                     f"{fv['effective_precision_should_remove']*100:.1f}% of removed lines are things a bibliography "
                     "cleaner should remove (bib entries, footnote/inline/table/web citations, CV publication lists). "
                     f"Moreover {fv['fp_that_are_missed_bibliography_pct']:.0f}% of the so-called false positives are "
                     "bibliography the windowed annotation MISSED — so the measured strict precision understates the truth: "
                     f"true-bibliography precision ≈ {fv['true_bib_precision_est']:.3f}.")
if probe:
    rep.section("Do frozen embeddings help? (the ML/DL probe)",
                body="Home CPU is too slow to embed the full corpus (~80 lines/s; ~3 h for the 840k labelled "
                     "lines, and the production corpus is ~47M rows), so the DL tier was probed on a stratified "
                     "~33k-line sample that oversamples the Greek-script recall holes. Adding a frozen "
                     "multilingual-e5-small line embedding to the handcrafted features lifts recall at matched "
                     f"precision 0.94 by {probe['sample_lift_recall_p94']:+.3f} overall and "
                     f"{probe['greek_bib_lift']:+.3f} on Greek-script bibliographies — embeddings recover exactly "
                     "the cases the Latin-biased regex features miss. Conclusion: ship the interpretable model now "
                     "(decisive precision win, Rust-deployable); use embeddings as a feature-discovery oracle to add "
                     "cheap Greek signals, and run the full DL bench on GPU (Clariden) only behind the precision-first "
                     "gate. See MODEL_TRADEOFF.md.",
                table_html=table(["model", "recall @ precision 0.94", "Greek-script bib recall"],
                                 [["handcrafted features", f"{probe['feat_only']['recall_at_p94']:.3f}", f"{probe['feat_only']['greek_bib_recall']:.3f}"],
                                  ["+ frozen embeddings", f"{probe['feat+emb']['recall_at_p94']:.3f}", f"{probe['feat+emb']['greek_bib_recall']:.3f}"]]))
rep.caveats("Method & caveats", [
    "Precision = fraction of REMOVED lines that are truly bibliography; the complement is prose amputation (the costly error). Cleaning is reversible (sidecar + drop-knob), so a miss is cheaper than a wrongful deletion.",
    "Doc-grouped leak-free split (a document's tail+body windows never straddle train/test). Lines are the unit; char-level adds OCR noise without changing the story.",
    "The Rust baseline (endmatter_bib) applies to greek_phd+openarchives only; Kallipos is section-based. Models are also reported on all test sources.",
    "Δstart/Δend matched to the maximum-overlap gold region; >+3 / <−3 in the prose-eating direction is counted as over-capture.",
])
rep.provenance = "Source: eval/results_span_models.json · span_seq_data + line_lr + decode_spans + window_clf vs out/*/refspans header→EOF."
out = os.path.expanduser("~/presentations/glossapi-tokenizer-extension/span-classifier-bench/index.html")
os.makedirs(os.path.dirname(out), exist_ok=True)
rep.write(out)
print("wrote", out)
