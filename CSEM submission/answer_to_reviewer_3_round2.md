# Response to Reviewer #3 (Round 2) — Manuscript CSEM-D-25-02344R1

We thank the reviewer for pressing us on these three points. In the first round we substituted
empirical/qualitative arguments for the formal analyses requested; this round, we have run the actual
tests.

## 1. Formal tests for non-stationarity / structural breaks

We have added new Appendix A, "Formal Stationarity and Structural-Break Diagnostics," reporting:

- **Augmented Dickey-Fuller (ADF) and KPSS tests** on both the raw price level and the daily return
  series for all five assets. The pattern is completely consistent across every asset: the price
  level classifies as unit-root/non-stationary (ADF fails to reject its unit-root null; KPSS strongly
  rejects its stationarity null, $p \leq 0.01$), while the daily return series classifies as
  stationary (ADF rejects the unit-root null at $p<0.001$ for every asset; KPSS fails to reject
  stationarity). This confirms our return-based modeling target is well-behaved even though the raw
  price process is not.
- **A CUSUM-of-OLS-residuals test** (Ploberger & Kramer, 1992) for mean-stability of the return series,
  and **a Quandt-Andrews Sup-F search** for the most likely single breakpoint date, with a
  parametric-bootstrap p-value (2,000 replicates, 15% trim). Neither test finds a statistically
  significant single mean-break in any of the five assets' return series at the 5% level (CUSUM
  $p \geq 0.21$; Sup-F bootstrap $p \geq 0.17$ in every case), even though the Sup-F search's
  candidate breakpoint dates land near well-known macro events (e.g. SPY: 2009-04-09, within days
  of the financial-crisis market bottom). We read this as supporting our framework's design premise:
  the non-stationarity we target is better characterized as continuous, gradual drift in
  predictability (consistent with Figure 1's reward-dispersion spikes) than as a small number of
  discrete regimes separated by sharp mean breaks — favoring an approach that continuously
  re-estimates the optimal predictor at every walk-forward window over one that would first need to
  detect and then react to isolated breakpoints.

We are explicit in the manuscript about what these tests can and cannot show: they formally establish
non-stationarity at the level of price dynamics and test for (and do not find) a discrete break in the
return series' mean, but they are not designed to, and should not be read as, direct evidence for the
qualitative "regime shifts" invoked as motivation elsewhere in the paper. The absence of a discrete
break is, if anything, more consistent with a picture of continuous, gradual drift in relative
predictability and volatility than with abrupt regime changes — which is precisely the kind of
non-stationarity our walk-forward CMAB framework is designed to track by continuously re-estimating
the optimal predictor, rather than a mechanism built to detect and react to discrete breakpoints.

We opted for ADF, KPSS, and the CUSUM/Sup-F family over Bai-Perron specifically because they are
implementable directly from packages already in our dependency set (`statsmodels`), keeping the
project's dependency surface unchanged; your comment framed ADF/KPSS/Sup-F/Bai-Perron as illustrative
alternatives rather than a mandatory set, and we believe this combination directly answers the
question asked.

## 2. Transaction-cost sensitivity

We have added new Appendix B, "Transaction-Cost Sensitivity," reporting annualized return, Sharpe
ratio, and maximum drawdown for the CMAB-greedy and static-ensemble strategies under one-way
transaction cost assumptions of 0, 5, 10, 25, and 50 basis points, applied whenever the strategy's
binary long/cash position switches (not on every arm reassignment while the position itself is
unchanged, since only a position switch corresponds to an actual trade in this single-instrument
long/cash formulation). This is computed directly from the already-cached arm-selection sequences
underlying Table 1, requiring no retraining. (An earlier draft of this letter described all three
metrics as reported while the manuscript's table in fact only showed Sharpe ratio; the manuscript's
Appendix B table now reports all three, as described below.)

Averaged across all 30 configurations, the CMAB-greedy strategy's zero-cost advantage erodes steadily
as costs rise, for both Sharpe ratio (+0.14 at 0 bps, beating the static ensemble in 20/30
configurations) and annualized return (+0.05 at 0 bps, same 20/30 win count): 19/30 configurations
still favor CMAB at 5 bps, 17/30 at 10 bps, and by 25–50 bps the average CMAB-minus-ensemble gap turns
negative for both metrics, with only 14–15/30 configurations retaining an edge. Maximum drawdown shows
a weaker, noisier pattern throughout (13–16/30 configurations favor CMAB depending on fee level, with
an essentially nil average advantage even at zero cost), since drawdown depends more on the timing of
a strategy's worst losses than on its average responsiveness.

To directly verify the mechanism, we separately computed each strategy's position-switch frequency
(the fraction of out-of-sample days the long/cash position flips) from the same cached data. Averaged
across all 30 configurations, CMAB-greedy switches position on 25.1% of days versus 19.4% for the
ensemble (~30% more often), which is what produces the extra fee events. This is, however,
policy-dependent, not a property of "the CMAB strategy" uniformly: LinUCB (24.7%) and especially
Thompson Sampling (33.9%) switch substantially more than the ensemble, while Softmax is if anything
slightly more conservative than the ensemble (16.8%). This is a genuine limitation for the more
exploratory policies, reported plainly: at realistic institutional cost levels (single-digit to
low-double-digit bps for the liquid instruments studied here) the CMAB framework retains its edge in
the clear majority of configurations, but at retail-level costs (25–50 bps) the advantage is no longer
guaranteed and should be assessed per-asset and per-policy, not assumed uniformly. Full
per-configuration results for all three metrics and the position-switch frequencies are provided as
`fee_sensitivity.csv` and `switch_rate.csv` in our GitHub repository, disaggregated by ticker, policy,
and architecture, rather than reproduced in the manuscript in full.

We continue to view directly incorporating transaction costs into the CMAB's reward function (rather
than as a post-hoc sensitivity analysis) as future work, as noted in our Conclusion — this fee-aware
reward shaping would change the bandit's learned policy itself, not just the reported metrics, and is
a substantively larger undertaking than the sensitivity analysis now included.

## 3. Statistical significance and multiple-testing correction

We appreciate the reviewer noting that our first-round response only addressed the seed-count half of
this ask. We have now added new Section 4.4, reporting a paired Diebold-Mariano test and a
block-bootstrap Sharpe-ratio-difference confidence interval, for every one of the 30 configurations
underlying Table 1, using the same 25 independent seeds. Recognizing that 30 comparisons constitute a
family for multiple-testing purposes, we apply both the Holm procedure (family-wise error rate,
conservative) and Benjamini-Hochberg (false discovery rate) corrections at $\alpha=0.05$.

**0 of 30 configurations remain significant after Holm correction; 0 of 30 after Benjamini-Hochberg**
(alpha=0.05). Only one configuration has a raw, uncorrected p<0.05 (GC=F/Thompson/heterogeneous,
p=0.018), and even that does not survive Holm correction (corrected p=0.551). A separate,
non-multiplicity-corrected block-bootstrap Sharpe-difference confidence interval excludes zero for 2
of the 30 configurations, both favoring the CMAB strategy. We report this result plainly rather than
selectively: the CMAB framework shows a consistent directional point-estimate edge (20 of 30
configurations favor CMAB, with the exceptions concentrated in TLT as already discussed in the
manuscript), but individual per-asset gaps against the static ensemble do not clear a
multiplicity-corrected significance bar given the sample sizes available (1,764 to 4,284 daily
out-of-sample observations per configuration, depending on asset) and the serial correlation present
in daily strategy returns. We have revised Section 4.1's "definitively dominant" language accordingly,
and added an explicit statement to this effect in both Section 4.4 and the Conclusion. We report the
primary test on the cross-seed-mean daily return against the deterministic static-ensemble baseline
(the same aggregation Table 1's "greedy (mean)" column already uses) rather than treating each of the
25 seeds as an independent comparison, since the seeds share an identical market realization and an
identical baseline and are therefore not independent replicates in the sense the correction procedure
assumes. As a further robustness check, Section 4.4 also reports the same DM test run independently on
each of the 25 seeds within every configuration (not fed into the multiple-testing correction, for the
reason just given): the median configuration has 0% of its seeds individually rejecting at the
uncorrected 5% level, the mean is 5.9%, and no configuration has a majority of seeds rejecting — the
per-seed evidence is consistent with, and does not contradict, the primary cross-seed-mean result.

---

We believe these additions directly close the three points raised and substantially strengthen the
manuscript's empirical rigor.
