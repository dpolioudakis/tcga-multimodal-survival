# Suggestions

## 1. Switch to a Cox proportional hazards formulation instead of binary 5-year classification

**The problem:** The current label definition throws away most of the cohort. Of the ~1,200 BRCA patients available pre-filtering, only n=290 survive the "exclude censored-before-5-years" rule, and only n=203 of those are in the training split. The README and PLANNING.md both flag "sample size" as *the* bottleneck limiting the deep learning models — but a large share of that bottleneck is self-inflicted by the label definition, not the underlying data.

**The fix:** Model time-to-event directly with a Cox proportional hazards model (or a Cox neural net / DeepSurv variant for the multimodal fusion arms) instead of binarizing at 5 years. This uses every patient's follow-up time, including those censored early — no one gets dropped just for having <5 years of observation.

**Why this is the highest-leverage change:**
- Could roughly triple the usable cohort size (n≈203 → n≈600+ for training), which directly attacks the constraint the project itself identifies as dominant.
- More statistically appropriate for survival data than an arbitrary binary cutoff — avoids discarding partial information from censored patients.
- Concordance index (C-index) replaces AUC/AP as the primary metric, and Kaplan–Meier/log-rank analysis (already in the pipeline) becomes a natural byproduct rather than a post-hoc add-on.
- With more usable samples, the "does deep learning ever pay off here" question the project is explicitly trying to answer becomes a fairer test — right now the deep models may be underperforming partly due to an avoidable sample-size handicap rather than an inherent architecture limitation.

**Scope:** This is a bigger change than a bug fix — it touches label construction (01/02), the loss function for every model (06–08), and the evaluation notebook (09). Worth scoping as a follow-up experiment/branch rather than a drop-in patch.
