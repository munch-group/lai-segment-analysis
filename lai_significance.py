"""
Significance testing for local ancestry inference (LAI) segment data.

Provides p-values for ancestry tract lengths under the admixture null model,
with corrections for length-biased sampling (inspection paradox), multiple
chromosomes, and multiple testing.
"""

import numpy as np
from scipy.stats import chi2
from scipy.interpolate import interp1d


# ── Single-segment p-value ───────────────────────────────────────────────────

def segment_pvalue(genetic_length, rate):
    """
    Gamma(2) survival function for a segment observed at a fixed position.

    Under the null, breakpoints are Poisson in genetic distance with rate λ.
    The segment covering a fixed point is length-biased: its length follows
    Gamma(2, 1/λ), giving survival function (1 + λg) exp(-λg).

    Parameters
    ----------
    genetic_length : array_like
        Segment length in Morgans.
    rate : float
        Rate parameter λ (e.g. f·t for the standard admixture model).

    Returns
    -------
    ndarray
        P-values, same shape as genetic_length.
    """
    g = np.asarray(genetic_length, dtype=float)
    lg = rate * g
    return (1 + lg) * np.exp(-lg)


# ── P-value matrix ───────────────────────────────────────────────────────────

def build_pvalue_matrix(segments, eval_positions, phys_to_gen, ancestry_rate):
    """
    Compute per-position, per-chromosome p-values for each ancestry label.

    Parameters
    ----------
    segments : list[list[(start_bp, end_bp, ancestry_label)]]
        Outer list indexes chromosomes.
    eval_positions : 1-d array
        Physical positions (bp) at which to evaluate.
    phys_to_gen : callable
        Physical (bp) -> genetic (Morgans).
    ancestry_rate : dict[str, float]
        Mapping ancestry label -> rate λ.

    Returns
    -------
    dict[str, ndarray]
        Mapping ancestry label -> (n_positions, n_chromosomes) array of
        p-values (NaN where the ancestry is absent).
    """
    eval_positions = np.asarray(eval_positions, dtype=float)
    n_pos = len(eval_positions)
    n_chrom = len(segments)
    labels = sorted(ancestry_rate.keys())

    pval_matrix = {a: np.full((n_pos, n_chrom), np.nan) for a in labels}

    for ci, ch in enumerate(segments):
        for (start, end, anc) in ch:
            if anc not in ancestry_rate:
                continue
            lam = ancestry_rate[anc]
            g = float(phys_to_gen(end) - phys_to_gen(start))
            p = segment_pvalue(g, lam)
            mask = (eval_positions >= start) & (eval_positions < end)
            pval_matrix[anc][mask, ci] = p

    return pval_matrix


# ── Combining p-values across chromosomes ────────────────────────────────────

def min_p(pval_matrix):
    """
    Min-p across chromosomes with Beta(1,k) correction.

    Uses effective sample size k_eff = number of unique p-values at each
    position, so that chromosomes sharing the same segment (identical
    breakpoints) count only once.

    Parameters
    ----------
    pval_matrix : ndarray, shape (n_pos, n_chrom)
        Per-chromosome p-values (NaN where absent).

    Returns
    -------
    ndarray, shape (n_pos,)
        Corrected p-values (NaN where no chromosome has the ancestry).
    """
    n_pos = pval_matrix.shape[0]
    result = np.full(n_pos, np.nan)
    for xi in range(n_pos):
        row = pval_matrix[xi, :]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            k = len(np.unique(valid))
            result[xi] = 1.0 - (1.0 - np.min(valid))**k
    return result


def fisher_combined(pval_matrix):
    """
    Fisher combined test across chromosomes.

    Computes -2 sum(ln p_c) ~ chi2(2k) where k is the effective number of
    independent samples (unique p-values) at each position. Duplicate
    p-values from shared segments are deduplicated before computing the
    test statistic.

    Parameters
    ----------
    pval_matrix : ndarray, shape (n_pos, n_chrom)

    Returns
    -------
    ndarray, shape (n_pos,)
    """
    eps = 1e-300
    n_pos = pval_matrix.shape[0]
    result = np.full(n_pos, np.nan)
    for xi in range(n_pos):
        row = pval_matrix[xi, :]
        valid = row[~np.isnan(row)]
        if len(valid) == 0:
            continue
        unique_p = np.unique(valid)
        k = len(unique_p)
        stat = -2 * np.sum(np.log(np.maximum(unique_p, eps)))
        result[xi] = chi2.sf(stat, df=2 * k)
    return result


def fisher_joint(pvals_a, pvals_b):
    """
    Fisher joint test across two ancestry classes.

    Combines -2(ln p_a + ln p_b) ~ chi2(4).

    Parameters
    ----------
    pvals_a, pvals_b : ndarray, shape (n_pos,)
        Per-position p-values (e.g. from min_p or fisher_combined).

    Returns
    -------
    ndarray, shape (n_pos,)
    """
    eps = 1e-300
    lp_a = np.where(np.isnan(pvals_a), 0., np.log(np.maximum(pvals_a, eps)))
    lp_b = np.where(np.isnan(pvals_b), 0., np.log(np.maximum(pvals_b, eps)))
    both = ~np.isnan(pvals_a) & ~np.isnan(pvals_b)
    stat = np.where(both, -2 * (lp_a + lp_b), np.nan)
    return np.where(both, chi2.sf(stat, df=4), np.nan)


# ── Multiple-testing correction ──────────────────────────────────────────────

def bh_correction(pvals):
    """
    Benjamini-Hochberg FDR correction. NaN-safe.

    Parameters
    ----------
    pvals : 1-d array

    Returns
    -------
    ndarray
        Adjusted p-values (same shape). NaN entries preserved.
    """
    pvals = np.asarray(pvals, dtype=float)
    adj = np.full_like(pvals, np.nan)
    valid = ~np.isnan(pvals)
    p = pvals[valid]
    n = len(p)
    if n == 0:
        return adj
    order = np.argsort(p)
    p_sorted = p[order]
    rank = np.arange(1, n + 1, dtype=float)
    adj_sorted = np.minimum(1.0, p_sorted * n / rank)
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_orig = np.empty(n)
    adj_orig[order] = adj_sorted
    adj[valid] = adj_orig
    return adj


# ── High-level entry point ───────────────────────────────────────────────────

def test_segments(segments, eval_positions, phys_to_gen, ancestry_rate, alpha=0.05):
    """
    Run the full significance testing pipeline.

    Parameters
    ----------
    segments : list[list[(start_bp, end_bp, ancestry_label)]]
    eval_positions : 1-d array of physical positions (bp).
    phys_to_gen : callable, bp -> Morgans.
    ancestry_rate : dict[str, float], ancestry label -> rate lambda.
    alpha : float, significance level.

    Returns
    -------
    dict with keys:
        pval_matrix, min_p, min_p_bonf, min_p_bh,
        fisher_combined, fisher_combined_bh,
        fisher_joint, fisher_joint_bh
    """
    eval_positions = np.asarray(eval_positions, dtype=float)
    n_eval = len(eval_positions)
    labels = sorted(ancestry_rate.keys())

    pm = build_pvalue_matrix(segments, eval_positions, phys_to_gen, ancestry_rate)

    mp = {a: min_p(pm[a]) for a in labels}
    mp_bonf = {a: np.minimum(1.0, mp[a] * n_eval) for a in labels}
    mp_bh = {a: bh_correction(mp[a]) for a in labels}

    fc = {a: fisher_combined(pm[a]) for a in labels}
    fc_bh = {a: bh_correction(fc[a]) for a in labels}

    fj = None
    fj_bh = None
    if len(labels) == 2:
        fj = fisher_joint(mp[labels[0]], mp[labels[1]])
        fj_bh = bh_correction(fj)

    return {
        "pval_matrix": pm,
        "min_p": mp,
        "min_p_bonf": mp_bonf,
        "min_p_bh": mp_bh,
        "fisher_combined": fc,
        "fisher_combined_bh": fc_bh,
        "fisher_joint": fj,
        "fisher_joint_bh": fj_bh,
    }


# ── Convenience: build interpolation function ────────────────────────────────

def make_phys_to_gen(physical_map, genetic_map):
    """
    Build a physical (bp) -> genetic (Morgans) interpolation function.

    Parameters
    ----------
    physical_map, genetic_map : 1-d arrays
    """
    return interp1d(physical_map, genetic_map,
                    bounds_error=False, fill_value='extrapolate')


# ── Simulation under the null ────────────────────────────────────────────────

def simulate_chromosome(chrom_len, gen_to_phys, phys_to_gen, ancestry_rate, rng):
    """
    Simulate a single chromosome as alternating ancestry segments.

    Breakpoints are Poisson in genetic distance, then mapped to physical
    coordinates. Segment genetic lengths ~ Exp(rate) where rate is looked
    up from ancestry_rate for the current segment's ancestry.

    Parameters
    ----------
    chrom_len : int
        Chromosome length in bp.
    gen_to_phys : callable
        Genetic (Morgans) -> physical (bp).
    phys_to_gen : callable
        Physical (bp) -> genetic (Morgans).
    ancestry_rate : dict[str, float]
        Ancestry label -> rate lambda.
    rng : numpy Generator

    Returns
    -------
    list of (start_bp, end_bp, ancestry_label) tuples covering [0, chrom_len].
    """
    labels = sorted(ancestry_rate.keys())
    gen_total = float(phys_to_gen(chrom_len))
    segs = []
    gen_pos = 0.0
    current = rng.choice(labels)

    while gen_pos < gen_total:
        lam = ancestry_rate[current]
        gen_len = rng.exponential(1.0 / lam)
        gen_end = min(gen_pos + gen_len, gen_total)
        phys_start = max(0, int(round(float(gen_to_phys(gen_pos)))))
        phys_end = min(chrom_len, int(round(float(gen_to_phys(gen_end)))))
        if phys_end > phys_start:
            segs.append((phys_start, phys_end, current))
        gen_pos = gen_end
        # Cycle to next label
        idx = labels.index(current)
        current = labels[(idx + 1) % len(labels)]

    if segs and segs[-1][1] < chrom_len:
        s, e, a = segs[-1]
        segs[-1] = (s, chrom_len, a)

    return segs


def inject_segment(segs, s0, e0, ancestry):
    """Replace all segments in [s0, e0) with a single segment of given ancestry."""
    new = []
    for (s, e, a) in segs:
        if e <= s0 or s >= e0:
            new.append((s, e, a))
        else:
            if s < s0:
                new.append((s, s0, a))
            if e > e0:
                new.append((e0, e, a))
    new.append((s0, e0, ancestry))
    new.sort()
    return new
