"""
Significance testing for local ancestry inference (LAI) segment data.

Provides p-values for ancestry tract lengths under the admixture null model,
with corrections for length-biased sampling (inspection paradox), multiple
chromosomes, and multiple testing.
"""

import numpy as np
import pandas as pd
import bitarray
from scipy.stats import chi2
from scipy.special import hyp1f1
from scipy.interpolate import interp1d
from math import lgamma, exp as _exp


# ── Coalescent effective sample size ─────────────────────────────────────────

def coalescent_k_eff(k, t_gen, n_pop):
    """
    Expected number of distinct ancestral lineages for k sampled gene copies
    after t_gen generations in a diploid population of n_pop individuals.

    Uses the spectral decomposition of the Kingman coalescent (Tavaré, 1984):

        E[A_k(τ)] = Σ_{m=1}^{k} (2m-1) · k!·(k-1)! / ((k-m)!·(k+m-1)!) · exp(-C(m,2)·τ)

    where τ = t_gen / (2·n_pop) is coalescent time.

    Parameters
    ----------
    k : int or ndarray
        Number of sampled lineages (chromosomes carrying the focal ancestry).
    t_gen : float
        Generations since admixture.
    n_pop : int
        Diploid effective population size.

    Returns
    -------
    float or ndarray
        Expected number of distinct ancestral lineages, in [1, k].
    """
    k = np.asarray(k, dtype=float)
    scalar = k.ndim == 0
    k = np.atleast_1d(k)
    tau = t_gen / (2.0 * n_pop)

    result = np.ones_like(k, dtype=float)
    for ki_idx in range(len(k)):
        ki = int(k[ki_idx])
        if ki <= 1:
            result[ki_idx] = max(ki, 0.0)
            continue
        s = 0.0
        log_k_fact = lgamma(ki + 1)
        log_km1_fact = lgamma(ki)
        for m in range(1, ki + 1):
            log_w = (np.log(2 * m - 1) + log_k_fact + log_km1_fact
                     - lgamma(ki - m + 1) - lgamma(ki + m))
            coal_rate = m * (m - 1) / 2.0
            s += _exp(log_w - coal_rate * tau)
        result[ki_idx] = s

    if scalar:
        return result[0]
    return result


def drift_prior(f0, t_gen, n_pop):
    """
    Compute Beta prior parameters informed by Wright-Fisher drift.

    Under WF diffusion, the frequency f(x) at a single locus after t
    generations of drift from initial frequency f0 has approximately:
        E[f] = f0,  Var[f] ≈ f0(1-f0)(1 - exp(-t/(2N)))

    Matching to a Beta(α₀, β₀) distribution gives α₀ = f0·c, β₀ = (1-f0)·c,
    where c = 1/(1 - exp(-t/(2N))) - 1.

    Parameters
    ----------
    f0 : float
        Initial admixture frequency of the focal ancestry.
    t_gen : float
        Generations since admixture.
    n_pop : int
        Diploid effective population size.

    Returns
    -------
    alpha0, beta0 : float
        Beta prior parameters for the focal ancestry.
    """
    tau = t_gen / (2.0 * n_pop)
    c = 1.0 / (1.0 - np.exp(-tau)) - 1.0
    return f0 * c, (1.0 - f0) * c


# ── Single-segment p-value ───────────────────────────────────────────────────

def segment_pvalue(genetic_length_left, genetic_length_right, rate):
    """
    Exponential survival p-values for left and right halves of a segment
    observed at a fixed position.

    Under the null, breakpoints are Poisson with rate λ. At a fixed
    position x inside a segment, the distances to the left and right
    breakpoints are independent Exp(λ).

    Returns
    -------
    p_left, p_right : float or ndarray
    """
    g_l = np.asarray(genetic_length_left, dtype=float)
    g_r = np.asarray(genetic_length_right, dtype=float)
    return np.exp(-rate * g_l), np.exp(-rate * g_r)


def beta_survival_pvalue(genetic_length, k_focal, n_total, t,
                          alpha_prior=1.0, beta_prior=1.0):
    """
    Selection-corrected Beta-integrated survival p-value for one side
    of a segment.

    We condition on the focal chromosome carrying ancestry A at position x.
    Given k out of n chromosomes carry A, the selection-corrected posterior
    for f(x) is Beta(k + alpha_prior + 1, n - k + beta_prior), because
    P(f | k, focal=A) ∝ f · Bin(k|n,f) · Beta(f|α₀,β₀)
                       ∝ f^{k+α₀} (1-f)^{n-k+β₀-1}.

    The marginal survival probability is then:
        exp(-t·g) · ₁F₁(k + α₀ + 1, n + α₀ + β₀ + 1, t·g)

    Parameters
    ----------
    genetic_length : float or ndarray
        Genetic distance (Morgans) from eval position to breakpoint.
    k_focal : int or ndarray
        Number of sampled chromosomes carrying the focal ancestry at position x.
    n_total : int
        Total number of sampled chromosomes at position x.
    t : float
        Generations since admixture.
    alpha_prior, beta_prior : float
        Beta prior parameters (default 1.0, 1.0 = flat/uniform prior).

    Returns
    -------
    float or ndarray -- p-value(s), clipped to [0, 1].
    """
    g = np.asarray(genetic_length, dtype=float)
    z = t * g
    a = np.asarray(k_focal, dtype=float) + alpha_prior + 1  # +1 selection correction
    b = np.asarray(n_total, dtype=float) + alpha_prior + beta_prior + 1
    return np.clip(np.exp(-z) * hyp1f1(a, b, z), 0.0, 1.0)


# ── Gap detection ───────────────────────────────────────────────────────────

def _is_gap(anc):
    """True if ancestry represents missing data (gap)."""
    if anc is None or anc is False or anc == '':
        return True
    if isinstance(anc, str) and anc.upper() == 'NA':
        return True
    try:
        if isinstance(anc, float) and np.isnan(anc):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _fill_implicit_gaps(segs):
    """Insert explicit gap segments for coordinate discontinuities and merge consecutive gaps."""
    if not segs:
        return segs
    # Step 1: fill coordinate discontinuities with explicit gaps
    filled = [segs[0]]
    for i in range(1, len(segs)):
        prev_end = segs[i - 1][1]
        curr_start = segs[i][0]
        if curr_start > prev_end:
            filled.append((prev_end, curr_start, None))
        filled.append(segs[i])
    # Step 2: merge consecutive gap segments
    merged = [filled[0]]
    for i in range(1, len(filled)):
        s, e, a = filled[i]
        ms, me, ma = merged[-1]
        if _is_gap(a) and _is_gap(ma):
            merged[-1] = (ms, e, None)
        else:
            merged.append((s, e, a))
    return merged


def _build_side_context(segs, seg_idx, direction, anc, phys_to_gen, max_chain_depth=1000):
    """
    Walk from segment seg_idx in the given direction, building the gap context.

    Returns None if the boundary is a true breakpoint (adjacent different
    ancestry or chromosome edge with no gap).

    Returns (gap_delta, chain) for gap boundaries, where:
        gap_delta : float — genetic width of the first gap
        chain : list of (segment_width, gap_width_or_None)
            Each entry is a same-ancestry segment beyond a gap.
            The last entry has gap_width=None (terminal).
    """
    step = -1 if direction == 'left' else 1
    neighbor_idx = seg_idx + step

    # Chromosome boundary — true breakpoint
    if neighbor_idx < 0 or neighbor_idx >= len(segs):
        return None

    neighbor = segs[neighbor_idx]
    n_start, n_end, n_anc = neighbor

    # Adjacent known different ancestry — true breakpoint
    if not _is_gap(n_anc):  # n_anc is a known ancestry
        if n_anc != anc:
            return None
        # Adjacent same ancestry (shouldn't happen in well-formed data,
        # but treat boundary as internal — not a breakpoint)
        return None

    # Adjacent gap — compute gap width
    gap_delta = abs(float(phys_to_gen(n_end)) - float(phys_to_gen(n_start)))

    # Look beyond the gap
    beyond_idx = neighbor_idx + step
    if beyond_idx < 0 or beyond_idx >= len(segs):
        # Gap at chromosome boundary — Case G-D
        return (gap_delta, [], None)

    beyond = segs[beyond_idx]
    b_start, b_end, b_anc = beyond

    if _is_gap(b_anc) or b_anc != anc:
        # Different ancestry beyond gap (or another gap) — Case G-D
        return (gap_delta, [], None)

    # Same ancestry beyond gap — Case G-S, build chain
    chain = []
    current_idx = beyond_idx
    depth = 0
    while depth < max_chain_depth:
        depth += 1
        seg_s, seg_e, seg_a = segs[current_idx]
        seg_width = abs(float(phys_to_gen(seg_e)) - float(phys_to_gen(seg_s)))

        next_idx = current_idx + step
        if next_idx < 0 or next_idx >= len(segs):
            # Chromosome boundary — terminal
            chain.append((seg_width, None))
            break

        next_seg = segs[next_idx]
        nx_start, nx_end, nx_anc = next_seg

        if not _is_gap(nx_anc):
            # Known ancestry — true breakpoint, terminal
            chain.append((seg_width, None))
            break

        # Another gap
        next_gap_delta = abs(float(phys_to_gen(nx_end)) - float(phys_to_gen(nx_start)))

        further_idx = next_idx + step
        if further_idx < 0 or further_idx >= len(segs):
            # Gap at chromosome boundary — terminal
            chain.append((seg_width, None))
            break

        further = segs[further_idx]
        f_start, f_end, f_anc = further

        if _is_gap(f_anc) or f_anc != anc:
            # Different ancestry beyond this gap — terminal
            chain.append((seg_width, None))
            break

        # Same ancestry beyond this gap — continue chain
        chain.append((seg_width, next_gap_delta))
        current_idx = further_idx

    if depth >= max_chain_depth:
        raise ValueError(
            f"Gap chain depth exceeded {max_chain_depth} at segment index "
            f"{seg_idx}. This suggests pathological input data with an "
            f"extremely long chain of same-ancestry segments separated by "
            f"gaps. Check your segment data or increase max_chain_depth."
        )

    terminal_bp = segs[current_idx][0] if direction == 'left' else segs[current_idx][1]
    return (gap_delta, chain, terminal_bp)



def _effective_pvalue(d1, context, lam, pi_focal=None, t_total=None):
    """
    Compute gap-aware p-value for one side of a segment.

    Parameters
    ----------
    d1 : float or ndarray
        Genetic distance from eval position(s) to the near segment boundary.
    context : None or (gap_delta, chain, terminal_bp)
        None for true breakpoints; otherwise output of _build_side_context.
    lam : float
        Rate parameter λ for the focal ancestry.
    pi_focal : float, optional
        Stationary probability of the focal ancestry (= lam_other / t_total).
        Required for Case G-S.
    t_total : float, optional
        Total transition rate (lam + lam_other). Required for Case G-S.

    Returns
    -------
    float or ndarray — p-value(s).
    """
    if context is None:
        return np.exp(-lam * d1)

    gap_delta, chain, _terminal_bp = context

    if not chain:
        # Case G-D: breakpoint known to be in gap, uniform average
        mu = lam * gap_delta
        if mu < 1e-12:
            return np.exp(-lam * d1)
        p_end = 1.0 - np.exp(-mu)
        return np.exp(-lam * d1) * p_end / mu

    # Case G-S: Bayesian gap correction
    # Build (gap_width, segment_width) pairs
    pairs = [(gap_delta, chain[0][0])]
    for i in range(len(chain) - 1):
        pairs.append((chain[i][1], chain[i + 1][0]))

    # Process right-to-left: G is the gap-corrected factor beyond d1
    G = None
    for delta, w in reversed(pairs):
        q = pi_focal + (1.0 - pi_focal) * np.exp(-t_total * delta)
        p_spans = np.exp(-lam * delta) / q
        # Marginalized survival when breakpoint is in gap
        M = (1.0 + np.exp(-lam * delta)) / 2.0
        if G is None:
            G = p_spans * np.exp(-lam * (delta + w)) + (1.0 - p_spans) * M
        else:
            G = p_spans * np.exp(-lam * (delta + w)) * G + (1.0 - p_spans) * M

    return np.exp(-lam * d1) * G


def _effective_pvalue_beta(d1, context, k_focal, n_total, t,
                            alpha_prior=1.0, beta_prior=1.0):
    """
    Beta-integrated gap-aware p-value for one side of a segment.

    Uses full Beta integration for the base survival term and point-estimate
    lambda for gap correction factors.
    """
    if context is None:
        return beta_survival_pvalue(d1, k_focal, n_total, t, alpha_prior, beta_prior)

    # Point-estimate lam from selection-corrected Beta posterior mean
    f_hat = (np.asarray(k_focal, dtype=float) + alpha_prior + 1) / (n_total + alpha_prior + beta_prior + 1)
    lam = (1.0 - f_hat) * t

    gap_delta, chain, _terminal_bp = context

    if not chain:
        # Case G-D: breakpoint known to be in gap, uniform average over gap position
        mu = lam * gap_delta
        safe_mu = np.where(np.abs(mu) < 1e-12, 1.0, mu)
        p_end = 1.0 - np.exp(-mu)
        correction = np.where(np.abs(mu) < 1e-12, 1.0, p_end / safe_mu)
        return beta_survival_pvalue(d1, k_focal, n_total, t, alpha_prior, beta_prior) * correction

    # Case G-S: Bayesian gap correction using point-estimate lam
    lam_other = f_hat * t
    t_total = lam + lam_other       # = t (always, since f_hat + (1-f_hat) = 1)
    pi_focal = np.where(t_total > 0, lam_other / t_total, 0.5)

    pairs = [(gap_delta, chain[0][0])]
    for i in range(len(chain) - 1):
        pairs.append((chain[i][1], chain[i + 1][0]))

    G = None
    for delta, w in reversed(pairs):
        q = pi_focal + (1.0 - pi_focal) * np.exp(-t_total * delta)
        p_spans = np.exp(-lam * delta) / q
        M = (1.0 + np.exp(-lam * delta)) / 2.0
        if G is None:
            G = p_spans * np.exp(-lam * (delta + w)) + (1.0 - p_spans) * M
        else:
            G = p_spans * np.exp(-lam * (delta + w)) * G + (1.0 - p_spans) * M

    return beta_survival_pvalue(d1, k_focal, n_total, t, alpha_prior, beta_prior) * G


# ── Segment normalization ────────────────────────────────────────────────────

def _normalize_segments(segments):
    """Convert segments from DataFrame or list-of-lists to canonical form."""
    if isinstance(segments, pd.DataFrame):
        required = {'chrom', 'start', 'end', 'ancestry'}
        if not required.issubset(segments.columns):
            raise ValueError(f"DataFrame must have columns {required}, got {set(segments.columns)}")
        result = []
        for _, grp in segments.sort_values(['chrom', 'start']).groupby('chrom', sort=True):
            result.append(list(zip(grp['start'], grp['end'], grp['ancestry'])))
        return result
    return segments


# ── P-value matrix ───────────────────────────────────────────────────────────

def build_pvalue_matrix(segments, eval_positions, phys_to_gen,
                         ancestry_rate=None, t=None,
                         alpha_prior=1.0, beta_prior=1.0):
    """
    Compute per-position, per-chromosome split p-values for each ancestry label.

    Parameters
    ----------
    segments : list[list[(start_bp, end_bp, ancestry_label)]]
        Outer list indexes chromosomes.
    eval_positions : 1-d array
        Physical positions (bp) at which to evaluate.
    phys_to_gen : callable
        Physical (bp) -> genetic (Morgans).
    ancestry_rate : dict[str, float], optional
        Mapping ancestry label -> rate λ. Used for fixed-rate model.
    t : float, optional
        Generations since admixture. When provided, uses Beta-integrated model
        that conditions on observed local ancestry frequencies.
    alpha_prior, beta_prior : float or dict[str, float]
        Beta prior parameters for the Beta-integrated model (default 1.0 = flat prior).
        If dicts, keyed by ancestry label (e.g. from drift_prior()).

    Returns
    -------
    dict[str, dict] with keys per ancestry label, each containing:
        'p_left':  ndarray (n_pos, n_chrom) — left-half p-values
        'p_right': ndarray (n_pos, n_chrom) — right-half p-values
        'bp_left': ndarray (n_pos, n_chrom) — left breakpoint position (for k_eff)
        'bp_right': ndarray (n_pos, n_chrom) — right breakpoint position (for k_eff)
    NaN where ancestry is absent.
    When t is not None, also includes 'k_counts' and 'n_chrom'.
    """
    segments = _normalize_segments(segments)
    eval_positions = np.asarray(eval_positions, dtype=float)
    n_pos = len(eval_positions)
    n_chrom = len(segments)

    use_beta = t is not None
    if not use_beta and ancestry_rate is None:
        raise ValueError("Must provide either ancestry_rate or t")

    # Discover labels
    if use_beta:
        labels_set = set()
        for ch_raw in segments:
            ch = _fill_implicit_gaps(ch_raw)
            for (_, _, anc) in ch:
                if not _is_gap(anc):
                    labels_set.add(anc)
        labels = sorted(labels_set)
    else:
        labels = sorted(ancestry_rate.keys())

    pval_matrix = {
        a: {
            'p_left':  np.full((n_pos, n_chrom), np.nan),
            'p_right': np.full((n_pos, n_chrom), np.nan),
            'bp_left': np.full((n_pos, n_chrom), np.nan),
            'bp_right': np.full((n_pos, n_chrom), np.nan),
        }
        for a in labels
    }

    # Pre-pass: count per-position ancestry frequencies (for Beta model)
    if use_beta:
        k_counts = {a: np.zeros(n_pos, dtype=int) for a in labels}
        for ci, ch_raw in enumerate(segments):
            ch = _fill_implicit_gaps(ch_raw)
            for (start, end, anc) in ch:
                if _is_gap(anc) or anc not in labels:
                    continue
                mask = (eval_positions >= start) & (eval_positions < end)
                k_counts[anc][mask] += 1

    for ci, ch_raw in enumerate(segments):
        ch = _fill_implicit_gaps(ch_raw)
        for si, (start, end, anc) in enumerate(ch):
            if _is_gap(anc):
                continue
            if not use_beta and anc not in ancestry_rate:
                continue
            if use_beta and anc not in labels:
                continue

            mask = (eval_positions >= start) & (eval_positions < end)
            if not np.any(mask):
                continue
            positions = eval_positions[mask]
            g_start = float(phys_to_gen(start))
            g_end = float(phys_to_gen(end))
            g_positions = np.asarray(phys_to_gen(positions), dtype=float)
            g_l = g_positions - g_start
            g_r = g_end - g_positions

            # Build gap context for each side
            left_ctx = _build_side_context(ch, si, 'left', anc, phys_to_gen)
            right_ctx = _build_side_context(ch, si, 'right', anc, phys_to_gen)

            if use_beta:
                k_focal = k_counts[anc][mask]
                ap = alpha_prior[anc] if isinstance(alpha_prior, dict) else alpha_prior
                bp = beta_prior[anc] if isinstance(beta_prior, dict) else beta_prior
                p_l = _effective_pvalue_beta(g_l, left_ctx, k_focal, n_chrom, t, ap, bp)
                p_r = _effective_pvalue_beta(g_r, right_ctx, k_focal, n_chrom, t, ap, bp)
            else:
                lam = ancestry_rate[anc]
                lam_other = sum(v for k, v in ancestry_rate.items() if k != anc)
                t_total = lam + lam_other
                pi_focal = lam_other / t_total if t_total > 0 else 0.5
                p_l = _effective_pvalue(g_l, left_ctx, lam, pi_focal, t_total)
                p_r = _effective_pvalue(g_r, right_ctx, lam, pi_focal, t_total)

            # Breakpoint identity for k_eff: use terminal breakpoint for gap-spanning
            if left_ctx is not None and left_ctx[1] and left_ctx[2] is not None:
                bp_left_id = left_ctx[2]
            else:
                bp_left_id = start

            if right_ctx is not None and right_ctx[1] and right_ctx[2] is not None:
                bp_right_id = right_ctx[2]
            else:
                bp_right_id = end

            pval_matrix[anc]['p_left'][mask, ci] = p_l
            pval_matrix[anc]['p_right'][mask, ci] = p_r
            pval_matrix[anc]['bp_left'][mask, ci] = bp_left_id
            pval_matrix[anc]['bp_right'][mask, ci] = bp_right_id

    if use_beta:
        pval_matrix['k_counts'] = k_counts
        pval_matrix['n_chrom'] = n_chrom

    return pval_matrix


# ── Fuzzy breakpoint clustering ───────────────────────────────────────────────

def _cluster_1d(values, tol):
    """Cluster sorted 1-d values: consecutive entries within `tol` are merged."""
    if tol == 0:
        uniq, idx = np.unique(values, return_index=True)
        return len(uniq), idx
    order = np.argsort(values)
    sorted_vals = values[order]
    n_clusters = 1
    cluster_start_idx = 0
    rep_indices = [order[0]]
    for i in range(1, len(sorted_vals)):
        if sorted_vals[i] - sorted_vals[cluster_start_idx] > tol:
            n_clusters += 1
            cluster_start_idx = i
            rep_indices.append(order[i])
    return n_clusters, np.array(rep_indices)


def _cluster_pairs(left_vals, right_vals, tol):
    """Count unique (left, right) breakpoint pairs under fuzzy matching."""
    if tol == 0:
        pairs = np.column_stack([left_vals, right_vals])
        return len(np.unique(pairs, axis=0))
    _, _, labels_l = _cluster_1d_labels(left_vals, tol)
    _, _, labels_r = _cluster_1d_labels(right_vals, tol)
    pairs = np.column_stack([labels_l, labels_r])
    return len(np.unique(pairs, axis=0))


def _cluster_1d_labels(values, tol):
    """Like _cluster_1d but also returns per-element cluster labels."""
    order = np.argsort(values)
    sorted_vals = values[order]
    labels = np.empty(len(values), dtype=int)
    cluster_id = 0
    cluster_start_idx = 0
    rep_indices = [order[0]]
    labels[order[0]] = 0
    for i in range(1, len(sorted_vals)):
        if sorted_vals[i] - sorted_vals[cluster_start_idx] > tol:
            cluster_id += 1
            cluster_start_idx = i
            rep_indices.append(order[i])
        labels[order[i]] = cluster_id
    return cluster_id + 1, np.array(rep_indices), labels


# ── Combining p-values across chromosomes ────────────────────────────────────

def min_p(pval_dict, fuzzy_coord=0):
    """
    Min-p across chromosomes with Beta(1,k) correction.

    Uses effective sample size k_eff = number of unique segments (by
    breakpoint pair) at each position.

    Parameters
    ----------
    pval_dict : dict with 'p_left', 'p_right', 'bp_left', 'bp_right' arrays.
    fuzzy_coord : float
        Maximum distance (bp) within which breakpoints are treated as shared.

    Returns
    -------
    ndarray, shape (n_pos,)
        Corrected p-values (NaN where no chromosome has the ancestry).
    """
    p_product = pval_dict['p_left'] * pval_dict['p_right']
    # Product of two independent Uniform[0,1] has CDF F(t) = t·(1 - ln t).
    # Apply F to convert back to Uniform[0,1] before Beta(1,k) correction.
    eps = 1e-300
    p_uniform = p_product * (1.0 - np.log(np.maximum(p_product, eps)))
    n_pos = p_uniform.shape[0]
    result = np.full(n_pos, np.nan)
    for xi in range(n_pos):
        row = p_uniform[xi, :]
        mask = ~np.isnan(row)
        valid = row[mask]
        if len(valid) > 0:
            k = _cluster_pairs(pval_dict['bp_left'][xi, mask],
                               pval_dict['bp_right'][xi, mask],
                               fuzzy_coord)
            result[xi] = -np.expm1(k * np.log1p(-np.min(valid)))
    return result


def fisher_combined(pval_dict, fuzzy_coord=0):
    """
    Fisher combined test using per-side effective sample sizes.

    Left and right half-segment p-values are combined separately using
    unique breakpoints to determine k_eff for each side:
        stat = -2·(Σ_unique_left ln(p_L) + Σ_unique_right ln(p_R))
        df = 2·k_eff_L + 2·k_eff_R

    Parameters
    ----------
    pval_dict : dict with 'p_left', 'p_right', 'bp_left', 'bp_right' arrays.
    fuzzy_coord : float
        Maximum distance (bp) within which breakpoints are treated as shared.

    Returns
    -------
    ndarray, shape (n_pos,)
    """
    eps = 1e-300
    n_pos = pval_dict['p_left'].shape[0]
    result = np.full(n_pos, np.nan)
    for xi in range(n_pos):
        pl = pval_dict['p_left'][xi, :]
        pr = pval_dict['p_right'][xi, :]
        bp_l = pval_dict['bp_left'][xi, :]
        bp_r = pval_dict['bp_right'][xi, :]
        mask = ~np.isnan(pl)
        if not np.any(mask):
            continue
        # Unique left breakpoints → unique left p-values
        k_l, idx_l = _cluster_1d(bp_l[mask], fuzzy_coord)
        unique_pl = pl[mask][idx_l]
        stat_l = -2 * np.sum(np.log(np.maximum(unique_pl, eps)))
        # Unique right breakpoints → unique right p-values
        k_r, idx_r = _cluster_1d(bp_r[mask], fuzzy_coord)
        unique_pr = pr[mask][idx_r]
        stat_r = -2 * np.sum(np.log(np.maximum(unique_pr, eps)))

        result[xi] = chi2.sf(stat_l + stat_r, df=2 * k_l + 2 * k_r)
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

def test_segments(segments, eval_positions, phys_to_gen,
                   ancestry_rate=None, t=None, n_pop=None,
                   alpha=0.05, fuzzy_coord=0,
                   alpha_prior=1.0, beta_prior=1.0):
    """
    Run the full significance testing pipeline.

    Parameters
    ----------
    segments : list[list[(start_bp, end_bp, ancestry_label)]]
    eval_positions : 1-d array of physical positions (bp).
    phys_to_gen : callable, bp -> Morgans.
    ancestry_rate : dict[str, float], optional. Ancestry label -> rate lambda.
    t : float, optional. Generations since admixture (enables Beta-integrated model).
    n_pop : int, optional. Diploid effective population size. When provided
        together with t, uses a drift-informed Beta prior instead of flat,
        and applies coalescent correction for genealogical correlation.
    alpha : float, significance level.
    fuzzy_coord : float, max bp distance for treating breakpoints as shared.
    alpha_prior, beta_prior : float, Beta prior parameters for Beta-integrated
        model. Ignored when n_pop is provided (drift prior is used instead).

    Returns
    -------
    dict with keys:
        pval_matrix, min_p, min_p_bonf, min_p_bh,
        fisher_combined, fisher_combined_bh,
        fisher_joint, fisher_joint_bh
    """
    if ancestry_rate is None and t is None:
        raise ValueError("Must provide either ancestry_rate or t")

    segments = _normalize_segments(segments)
    eval_positions = np.asarray(eval_positions, dtype=float)
    n_eval = len(eval_positions)

    # Discover labels and estimate per-ancestry frequencies
    if t is not None:
        labels_set = set()
        seg_bp_total = {}
        total_bp = 0
        for ch_raw in segments:
            ch = _fill_implicit_gaps(ch_raw)
            for (start, end, anc) in ch:
                if not _is_gap(anc):
                    labels_set.add(anc)
                    seg_bp_total[anc] = seg_bp_total.get(anc, 0) + (end - start)
                    total_bp += (end - start)
        labels = sorted(labels_set)
    else:
        labels = sorted(ancestry_rate.keys())

    # Compute drift-informed priors if n_pop is provided
    if n_pop is not None and t is not None and total_bp > 0:
        freq_est = {a: seg_bp_total.get(a, 0) / total_bp for a in labels}
        prior_pairs = {a: drift_prior(freq_est[a], t, n_pop) for a in labels}
        ap_dict = {a: prior_pairs[a][0] for a in labels}
        bp_dict = {a: prior_pairs[a][1] for a in labels}
    else:
        ap_dict = alpha_prior
        bp_dict = beta_prior

    pm = build_pvalue_matrix(segments, eval_positions, phys_to_gen,
                              ancestry_rate=ancestry_rate, t=t,
                              alpha_prior=ap_dict, beta_prior=bp_dict)

    mp = {a: min_p(pm[a], fuzzy_coord) for a in labels}
    mp_bonf = {a: np.minimum(1.0, mp[a] * n_eval) for a in labels}
    mp_bh = {a: bh_correction(mp[a]) for a in labels}

    fc = {a: fisher_combined(pm[a], fuzzy_coord) for a in labels}
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

def simulate_chromosome(chrom_len, gen_to_phys, phys_to_gen, ancestry_rate, rng, chrom_id=0):
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
    chrom_id : int or str
        Label for the chromosome (stored in the 'chrom' column).

    Returns
    -------
    pd.DataFrame with columns ['chrom', 'start', 'end', 'ancestry'].
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

    return pd.DataFrame(segs, columns=['start', 'end', 'ancestry']).assign(chrom=chrom_id)


def simulate_population(chrom_len, gen_to_phys, phys_to_gen, f_resident,
                        n_pop, t, n_samples, rng, window_size=100_000):
    """Wright-Fisher forward simulation of local ancestry (bit-packed).

    Each chromosome is stored as a bit-packed numpy row (0=resident,
    1=foreign). Crossover positions are sampled proportional to per-window
    genetic distance. Recombination uses precomputed bitwise masks.

    Parameters
    ----------
    chrom_len : int
        Chromosome length in bp.
    gen_to_phys : callable
        Genetic (Morgans) -> physical (bp).
    phys_to_gen : callable
        Physical (bp) -> genetic (Morgans). Used to compute per-window
        crossover weights.
    f_resident : float
        Initial frequency of resident ancestry (e.g., 0.7).
    n_pop : int
        Effective population size (e.g., 10000).
    t : int
        Number of generations to simulate.
    n_samples : int
        Number of chromosomes to sample from the final population.
    rng : numpy Generator
    window_size : int
        Size of each discrete window in bp (default 100,000).

    Returns
    -------
    pd.DataFrame with columns ['chrom', 'start', 'end', 'ancestry'].
        One set of rows per sampled chromosome (chrom=0..n_samples-1).
    """
    n_windows = chrom_len // window_size
    n_packed = (n_windows + 7) // 8

    # Per-window genetic distance for weighted crossover sampling
    window_edges = np.arange(0, chrom_len + 1, window_size)
    gen_edges = np.array([float(phys_to_gen(x)) for x in window_edges])
    gen_per_window = np.diff(gen_edges)
    cum_weights = np.cumsum(gen_per_window)

    # Precompute left-masks: mask[xo] has bits 0..xo-1 set
    all_masks = np.zeros((n_windows, n_packed), dtype=np.uint8)
    for xo_pos in range(n_windows):
        ba = bitarray.bitarray(n_windows)
        ba.setall(0)
        ba[:xo_pos] = 1
        all_masks[xo_pos] = np.frombuffer(ba.tobytes(), dtype=np.uint8)

    # Initialize population as packed bytes: (n_pop, n_packed)
    n_resident = int(round(f_resident * n_pop))
    pop = np.zeros((n_pop, n_packed), dtype=np.uint8)
    pop[n_resident:, :] = all_masks[-1]  # all-1s for foreign

    # Forward simulation with Poisson(gen_total) crossovers per meiosis
    gen_total = cum_weights[-1]
    for _ in range(t):
        p1 = rng.integers(0, n_pop, size=n_pop)
        p2 = rng.integers(0, n_pop, size=n_pop)
        n_xo = rng.poisson(gen_total, size=n_pop)
        total_xo = n_xo.sum()
        combined = np.zeros((n_pop, n_packed), dtype=np.uint8)
        if total_xo > 0:
            # Sample all crossover positions at once, convert to window indices
            u_all = rng.uniform(0, gen_total, size=total_xo)
            xo_all = np.searchsorted(cum_weights, u_all, side='right')
            xo_all = np.clip(xo_all, 1, n_windows - 1)
            # XOR masks per individual to build recombination pattern
            offsets = np.concatenate([[0], np.cumsum(n_xo)])
            for i in range(n_pop):
                for j in range(offsets[i], offsets[i + 1]):
                    combined[i] ^= all_masks[xo_all[j]]
        pop = (pop[p1] & ~combined) | (pop[p2] & combined)

    # Sample chromosomes and run-length encode to segments
    sample_idx = rng.choice(n_pop, size=n_samples, replace=False)
    rows = []
    for chrom_id, idx in enumerate(sample_idx):
        arr = np.unpackbits(pop[idx])[:n_windows]
        changes = np.where(np.diff(arr) != 0)[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [n_windows]])
        for s, e in zip(starts, ends):
            anc = "foreign" if arr[s] else "resident"
            rows.append((chrom_id, s * window_size, e * window_size, anc))

    return pd.DataFrame(rows, columns=['chrom', 'start', 'end', 'ancestry'])


def _recombine(segs_a, segs_b, crossover_positions, chrom_len):
    """Splice two parent segment lists at sorted crossover positions.

    Parameters
    ----------
    segs_a, segs_b : list of (start, end, ancestry) tuples
        Parent segments, sorted by start position, covering [0, chrom_len).
    crossover_positions : array-like of int
        Sorted crossover positions in bp (must be in (0, chrom_len)).
    chrom_len : int
        Chromosome length in bp.

    Returns
    -------
    list of (start, end, ancestry) tuples
        Child segments after recombination.
    """
    # Build interval boundaries: [0, xo1, xo2, ..., chrom_len]
    boundaries = [0] + list(crossover_positions) + [chrom_len]
    parents = [segs_a, segs_b]
    use_a = True  # start with parent a
    result = []
    cursor = [0, 0]  # index into each parent's segment list

    for k in range(len(boundaries) - 1):
        iv_start = boundaries[k]
        iv_end = boundaries[k + 1]
        if iv_start >= iv_end:
            use_a = not use_a
            continue
        p = 0 if use_a else 1
        segs = parents[p]
        i = cursor[p]
        # Advance cursor to first segment overlapping iv_start
        while i < len(segs) and segs[i][1] <= iv_start:
            i += 1
        # Extract segments from this parent within [iv_start, iv_end)
        while i < len(segs) and segs[i][0] < iv_end:
            s, e, a = segs[i]
            cs = max(s, iv_start)
            ce = min(e, iv_end)
            if cs < ce:
                # Merge with previous if same ancestry
                if result and result[-1][2] == a and result[-1][1] == cs:
                    result[-1] = (result[-1][0], ce, a)
                else:
                    result.append((cs, ce, a))
            if e <= iv_end:
                i += 1
            else:
                break
        cursor[p] = i
        use_a = not use_a

    return result


def simulate_population_exact(chrom_len, gen_to_phys, phys_to_gen, f_resident,
                              n_pop, t, n_samples, rng):
    """Wright-Fisher forward simulation of local ancestry (exact, 1-bp resolution).

    Each chromosome is stored as a list of (start, end, ancestry) tuples.
    Crossover positions are sampled in genetic space and converted to physical
    coordinates via gen_to_phys. No windowing or discretization.

    Parameters
    ----------
    chrom_len : int
        Chromosome length in bp.
    gen_to_phys : callable
        Genetic (Morgans) -> physical (bp).
    phys_to_gen : callable
        Physical (bp) -> genetic (Morgans).
    f_resident : float
        Initial frequency of resident ancestry (e.g., 0.7).
    n_pop : int
        Effective population size.
    t : int
        Number of generations to simulate.
    n_samples : int
        Number of chromosomes to sample from the final generation.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    pd.DataFrame
        Columns: chrom, start, end, ancestry.
    """
    n_resident = int(round(f_resident * n_pop))
    gen_total = phys_to_gen(chrom_len)

    # Initialize population
    pop = []
    for _ in range(n_resident):
        pop.append([(0, chrom_len, "resident")])
    for _ in range(n_pop - n_resident):
        pop.append([(0, chrom_len, "foreign")])

    # Forward simulation
    for _ in range(t):
        p1 = rng.integers(0, n_pop, size=n_pop)
        p2 = rng.integers(0, n_pop, size=n_pop)
        n_xo = rng.poisson(gen_total, size=n_pop)
        new_pop = []
        for i in range(n_pop):
            if n_xo[i] == 0:
                # No crossover: child is a copy of parent a
                new_pop.append(list(pop[p1[i]]))
            else:
                # Sample crossover positions in genetic space
                xo_gen = rng.uniform(0, gen_total, size=n_xo[i])
                # Convert to physical positions
                xo_phys = np.round(gen_to_phys(xo_gen)).astype(int)
                # Clip to valid range
                xo_phys = np.clip(xo_phys, 1, chrom_len - 1)
                # Handle duplicates: keep positions with odd occurrence count
                vals, counts = np.unique(xo_phys, return_counts=True)
                xo_final = vals[counts % 2 == 1]
                if len(xo_final) == 0:
                    new_pop.append(list(pop[p1[i]]))
                else:
                    child = _recombine(pop[p1[i]], pop[p2[i]], xo_final,
                                       chrom_len)
                    new_pop.append(child)
        pop = new_pop

    # Sample chromosomes
    sample_idx = rng.choice(n_pop, size=n_samples, replace=False)
    rows = []
    for chrom_id, idx in enumerate(sample_idx):
        for s, e, a in pop[idx]:
            rows.append((chrom_id, s, e, a))

    return pd.DataFrame(rows, columns=['chrom', 'start', 'end', 'ancestry'])


def inject_segment(segs, s0, e0, ancestry):
    """Replace all segments in [s0, e0) with a single segment of given ancestry."""
    is_df = isinstance(segs, pd.DataFrame)
    if is_df:
        chrom_val = segs['chrom'].iloc[0]
        tuples = list(zip(segs['start'], segs['end'], segs['ancestry']))
    else:
        tuples = segs
    new = []
    for (s, e, a) in tuples:
        if e <= s0 or s >= e0:
            new.append((s, e, a))
        else:
            if s < s0:
                new.append((s, s0, a))
            if e > e0:
                new.append((e0, e, a))
    new.append((s0, e0, ancestry))
    new.sort()
    if is_df:
        return pd.DataFrame(new, columns=['start', 'end', 'ancestry']).assign(chrom=chrom_val)
    return new
