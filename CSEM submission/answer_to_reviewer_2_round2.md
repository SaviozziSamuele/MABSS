# Response to Reviewer #2 (Round 2) — Manuscript CSEM-D-25-02344R1

We thank the reviewer for the careful re-reading of the revision and for identifying several points
of internal consistency that needed reconciling. We address each in turn.

## Major Points

### 1. On the revised Sharpe ratio (0.27 → 0.65)

We appreciate the reviewer pressing us for more specificity here, and we owe two clarifications.

**What we can and cannot reconstruct.** The correction from the original submission's SPY/UCB Sharpe
ratio of 0.27 was made during an earlier revision pass, prior to the code refactor and audit trail we
built for this round of review. We are not able to forensically recover the exact original bug from
that snapshot — the repository's preserved history begins after that fix was already applied, so we
cannot produce a diff against the pre-fix code the way we can for issues found during this round's
audit (below). What we can say with confidence: the fix combined (a) a corrected return-computation
step in the prediction pipeline, and (b) the introduction of the variance-aware UCB/Thompson Sampling
policies described in our response to Comment 1, which materially changed the reported numbers
because the original 0.27 figure was measured under a Softmax-only comparison.

**A separate, independently-found correction.** During this revision we also identified and fixed an
unrelated bug: our annualization factor was hardcoded at 252 trading days/year for every asset. For
SPY this makes negligible difference (SPY trades ~251.5 bars/year in our sample, essentially 252 —
Sharpe changes by well under 0.1%), but it materially inflates the Sharpe ratio for BTC-USD (~365
bars/year, no exchange holidays) and EURUSD=X (~260 bars/year). We disclose this separately because it
is a distinct issue from the pct-change fix above, and because it is fully reproducible from our
current codebase and test suite.

**Resolving our own earlier inconsistency.** We also note, on re-reading our first-round response,
that we stated two different "resolved" Sharpe values (0.65 and 0.60) in the same section — an error
on our part. For the record, Table 1 as it stands in this revision reports SPY/UCB Sharpe as **0.66**
(heterogeneous pool) and **0.69** (homogeneous pool).

**Disentangling the driver.** The reviewer also asks us to separate the contribution of the
calculation fix from the contribution of policy choice. We can do this partially using numbers already
in Table 1: comparing Softmax (the policy used in the original submission) to UCB under the *current,
corrected* pipeline isolates the policy-choice contribution alone. For SPY, homogeneous pool: Softmax
Sharpe = 0.66, UCB Sharpe = 0.69 — a modest but real gap, smaller than the full 0.27→0.65 jump, which
confirms the calculation fix (not recoverable in exact magnitude, as noted above) was the larger
contributor. We now report a formal Diebold-Mariano test of whether this and other CMAB-vs-baseline
gaps are statistically distinguishable from zero (see our response to your Comment/Additional Point 2
below and new Section 4.4).

**Did the same error affect other original-submission statistics, including the L1/L2/BCE reward-
function comparison?** That comparison table no longer appears in the current manuscript at all — we
address this directly under Additional Point 1 below, since it is the same underlying issue (a shift
in the paper's central framing, not a residual of the pct-change bug).

### 2. On the consistency of the trading strategy (long-only vs. long–short)

The reviewer is correct, and we apologize for the confusion: our previous response's language
("zero-beta Long/Short framework," "exploratory short positions," "Long/Short CMAB") was simply wrong
and does not describe our actual implementation. The strategy has always been, and remains, long/cash
exactly as defined in Section \ref{sec:pred} of the manuscript: $S_t \in \{0,1\}$, with $S_t=0$
liquidating fully into a zero-return cash position — never a short position. We have not changed the
manuscript's implementation description, since it was already correct; we retract the erroneous
framing from our prior response letter. This also closes out the third candidate driver the reviewer
names at the end of this point: a short-selling/"position-timing" mechanism was never a real
contributor to the Sharpe-ratio improvement, since the strategy never took short positions in the
first place — it was an artifact of the prior letter's incorrect description, not an actual third
driver alongside the calculation fix and policy choice discussed under Major Point 1.

### 3. On the exclusion of a risk metric from the context vector

The reviewer is correct on the theoretical point, and we have corrected the manuscript's wording
accordingly: in a disjoint (per-arm) linear model such as ours, a feature shared across arms is *not*
automatically non-discriminatory, since each arm learns its own weight vector and could in principle
assign that shared input a different coefficient. Our original phrasing ("merely introduce a global,
non-discriminatory bias term") overstated this. We have revised Section 3.4 to instead ground the
design choice in the fact that such a feature reflects a property of the *asset*, not of the specific
model producing the forecast, and therefore carries no genuinely arm-specific information -- a
narrower and more defensible claim than the one we originally made, and one that motivated us to test
it empirically rather than rely on the argument alone.

We have run the empirical comparison the reviewer suggests: appending a shared, rolling
realized-volatility feature (a 20-day trailing standard deviation of the target's own return series,
identical across every arm at a given timestep) to the context vector for SPY, homogeneous pool, all
three policies (5 seeds each, compared against a variance-matched 5-seed subsample of the real,
already-published 25-seed baseline).

The result is more informative than a simple confirmation: it is **policy-dependent, not uniformly
neutral**. For Softmax the feature is essentially inert (Sharpe 0.682 -> 0.679, within 5-seed noise).
For Thompson Sampling it is mildly positive (0.666 -> 0.686). For LinUCB -- the policy we identify as
the strongest point-estimate performer in Table 1 -- it is clearly detrimental (Sharpe 0.686 -> 0.642,
a 6.4% relative decline, with annualized return falling correspondingly). We offer a plausible
mechanism in the manuscript: LinUCB's exploration bonus is a function of each arm's estimated context
covariance, and a fourth, non-discriminating dimension enlarges that estimate without contributing
signal, inflating exploration noise -- a class of harm confidence-bound methods are structurally more
exposed to than Softmax (whose exploration scale depends on logit magnitude, not context covariance)
or Thompson Sampling.

We therefore read this ablation as **strengthening** rather than merely confirming our theoretical
argument: excluding the shared risk feature is not just well-motivated in principle but empirically
consequential for our best-performing policy. We have added a one-sentence pointer to Section 3.4 of
the manuscript, alongside the existing theoretical argument, with the full empirical comparison
reported in new Appendix C. We are transparent about scope: this check uses 5 seeds on SPY only
(not the full 25-seed, 5-asset design), given the wall-clock cost of the bandit stage at this scale on
local hardware; extending it is left to the broader retraining pass discussed in our Conclusion.

## Responses That Are Largely Satisfactory

**Comment 1 (UCB "dominant").** We agree the margins are narrow for several assets and have softened
this language (now "the point-estimate leader" rather than "definitively dominant"). We now tie the
claim directly to the formal significance results in new Section 4.4: across all 30 configurations
underlying Table 1 (5 tickers x 3 policies x 2 pool types), a paired Diebold-Mariano test finds
**zero** configurations remain significant at alpha=0.05 after either Holm (family-wise) or
Benjamini-Hochberg (false-discovery-rate) correction. Only one configuration has a raw, uncorrected
p<0.05 (GC=F/Thompson/heterogeneous, p=0.018; Holm-corrected p=0.551). A separate, less conservative
block-bootstrap Sharpe-difference confidence interval excludes zero for 2 of 30 configurations
(BTC-USD/Thompson/homogeneous and GC=F/Thompson/heterogeneous), both favoring CMAB. 20 of the 30
point estimates favor CMAB overall (the 10 that do not are concentrated in TLT, the asset we already
flag as an exception). We now state this plainly in the text: UCB's lead is a consistent directional
pattern across asset classes, not a set of individually confirmed per-asset effects.

**Comment 2, first part (SGA/response coefficient).** We have added a sentence to Section 2.1.2
acknowledging that the Softmax/SGA formulation's exploration scale is governed by logit magnitude
rather than an explicit uncertainty estimate, and linking this directly to our motivation for
including UCB and Thompson Sampling.

**Comment 4 (placebo test dispersion).** We have added a sentence to Section 4.3 stating explicitly
that the reported placebo-test result reflects a single synthetic realization (fixed generator seed).
Characterizing seed-to-seed dispersion requires retraining the full predictor pool per realization,
which we are deferring to the broader retraining pass discussed in our Conclusion's Future Work
paragraph (see also our response to Reviewer #3, Comment 1, regarding the compute cost of retraining).

## Additional Points

### 1. Treatment of the original central claim

We agree this shift in emphasis should be acknowledged explicitly rather than left implicit. We have
added a sentence to the Conclusion noting that earlier versions of this work emphasized the
autocorrelation structure of the reward signal in relative isolation, while the present study
foregrounds the empirically more consequential finding that architectural heterogeneity penalizes
static aggregation while benefiting CMAB-based selection. The L1/L2/BCE reward-function comparison
table from the original submission was not affected by any calculation error — it was removed because
the paper's focus shifted, not superseded by a corrected version of itself.

### 2. Statistical significance

We have added new Section 4.4, reporting a paired Diebold-Mariano test and a block-bootstrap
Sharpe-ratio-difference confidence interval for the CMAB-greedy vs. static-ensemble comparison, for
every one of the 30 configurations in Table 1, computed from the same 25 independent seeds already
used to produce the table. We apply both Holm (family-wise) and Benjamini-Hochberg (false discovery
rate) corrections across the 30 tests.

The honest result: **0 of 30 configurations remain significant after either correction at alpha=0.05.**
We report this plainly rather than searching for a more favorable test specification. 20 of the 30
point estimates favor CMAB (directionally consistent with Table 1, with the shortfall concentrated in
TLT, our already-noted exception), and a less conservative block-bootstrap Sharpe-difference CI
excludes zero for 2 of 30 configurations (both favoring CMAB) -- but neither the raw DM p-values nor
this bootstrap procedure survive family-wise or FDR correction across the full family of 30 tests. We
believe this is the right, if humbling, way to answer the question: the paper's contribution is a
consistent directional edge across diverse asset classes and architectures, not a set of individually
statistically confirmed per-asset effects, and we have revised the relevant claims in Sections 4.1 and
5 (Conclusion) to reflect this precisely.

### 3. Internal consistency of stated values

**Evaluation period.** The current manuscript states a single, consistent evaluation period ("January
2000 to March 2026," Table 1 caption) that matches our current cached data exactly (verified: our SPY
series' out-of-sample predictions run through 2026-03-16). The "January 2000 to January 2024" language
the reviewer may be recalling comes from our first-round response letter, which was written against an
earlier data pull; our corpus has since been refreshed and the manuscript's caption was updated to
match, but we had not revised the earlier response letter's wording to match — we do so now. We have
also clarified the caption itself: "January 2000" is the start of the raw data pull, while
out-of-sample testing (and therefore every number in Table 1) only begins after the initial four-year
walk-forward training window, i.e. from approximately 2004 onward — pre-empting a version of this
exact ambiguity from recurring in a future round.

**GARCH vs. EGARCH.** The placebo test described in Section 4.3 of the manuscript is, and has
consistently been, a **GARCH(1,1)** process — we have verified this directly against our synthetic-data
generator, which implements the standard symmetric GARCH(1,1) variance recursion with no
asymmetric/leverage term. The "EGARCH(1,1), Section 4.2" description that appeared in our response to
Reviewer #3 in the previous round was an error in that letter — the underlying ticker label used
internally for this dataset ("EGARCH") is a legacy naming artifact from early development and does not
reflect the actual generative model. We regret the confusion and confirm GARCH(1,1)/Section 4.3 is the
correct and only description.

---

We thank the reviewer again for the thorough second read and believe the manuscript is substantially
strengthened by these clarifications and the new empirical analyses.
