from clause86.eval import holm_bonferroni, paired_bootstrap_ci, wilcoxon_signed_rank


def test_bootstrap_mean_is_close():
  mean, lo, hi = paired_bootstrap_ci(
    [0.1, 0.2, 0.3, 0.4], n_resamples=200, seed=1,
  )
  assert 0.0 <= lo <= mean <= hi <= 1.0
  assert abs(mean - 0.25) < 0.001


def test_wilcoxon_returns_valid_p():
  out = wilcoxon_signed_rank([1.0, 2.0, 3.0], [0.5, 1.5, 2.5])
  assert 0.0 <= out["p_value"] <= 1.0
  assert out["n"] == 3.0


def test_holm_bonferroni_corrects_in_order():
  # Holm step-down with alpha=0.05, n=3:
  #  sorted p-values 0.001, 0.04, 0.5
  #  thresholds 0.05/3=0.0167, 0.05/2=0.025, 0.05/1=0.05
  #  0.001 < 0.0167 -> reject; 0.04 > 0.025 -> stop.
  rejects = holm_bonferroni([0.001, 0.04, 0.5], alpha=0.05)
  assert rejects == [True, False, False]
  # And a configuration where two consecutive p-values both reject.
  rejects_2 = holm_bonferroni([0.001, 0.01, 0.5], alpha=0.05)
  assert rejects_2 == [True, True, False]
