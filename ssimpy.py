"""ssimpy: detect statistically significant somatic co-mutations.

Usage (single mutation type):
    python ssimpy.py --gam <gam.tsv> [--maf <file.maf> | --tmb <tmb.tsv>] [options]

Usage (multiple mutation types):
    python ssimpy.py --gam <ms.tsv> <tr.tsv> --tmb <ms_tmb.tsv> <tr_tmb.tsv> [options]

Run from the selectsim/ directory so that local imports resolve correctly.
"""
import argparse
import sys

import numpy as np

from data_io import load_gam, load_maf, load_tmb, tmb_from_gam, save_results
from expected_matrix import (compute_E_matrix, compute_E_matrix_by_class,
                              compute_E_matrix_multitype,
                              compute_E_matrix_multitype_by_class)
from simulation import run_simulations, filter_simulations
from comutation import compute_penalty_vector, compute_weighted_comutation
from statistics import run_significance_analysis


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='SelectSim — significance testing for somatic co-mutations'
    )
    p.add_argument('--gam', required=True, nargs='+',
                   help='One or more GAM TSV files (rows=genes, columns=samples, binary). '
                        'Multiple files represent distinct mutation types (e.g. missense, '
                        'truncating); the combined GAM is their element-wise union and E '
                        'is the element-wise max of per-type E matrices.')
    p.add_argument('--maf', default=None,
                   help='Path to MAF file for TMB estimation (single --gam only)')
    p.add_argument('--tmb', default=None, nargs='+',
                   help='One or more pre-calculated TMB TSV files (sample, tmb [, class]). '
                        'Count must match --gam when provided; TMBs are summed across types '
                        'for the penalty vector. Falls back to GAM column sums if omitted.')
    p.add_argument('--N', type=int, default=1000,
                   help='Number of rcRAS simulations to generate (default: 1000)')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed for reproducibility')
    p.add_argument('--tau', type=float, default=1.0,
                   help='Penalty threshold tau: TMB fold-changes <= tau are not penalized '
                        '(default: 1.0)')
    p.add_argument('--lam', type=float, default=0.3,
                   help='Penalty rate lambda (default: 0.3)')
    p.add_argument('--fdr', type=float, default=0.1,
                   help='Target FDR for significance calls (default: 0.1)')
    p.add_argument('--filter-pct', type=float, default=0.10,
                   help='Fraction of simulations to discard by sample-frequency deviation '
                        '(default: 0.10)')
    p.add_argument('--min-mut', type=int, default=5,
                   help='Minimum number of mutated samples required to keep a gene '
                        '(default: 5); genes with fewer mutations are excluded before analysis')
    p.add_argument('--output', default='selectsim_results.tsv',
                   help='Output TSV file path (default: selectsim_results.tsv)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    gam_paths = args.gam          # list of 1+ paths
    tmb_paths = args.tmb          # list of 1+ paths, or None
    n_types   = len(gam_paths)
    is_multitype = n_types > 1

    # Validate argument combinations
    if is_multitype and args.maf:
        sys.exit("Error: --maf is not supported with multiple --gam files. "
                 "Provide per-type TMB files via --tmb instead.")
    if tmb_paths is not None and len(tmb_paths) != n_types:
        sys.exit(f"Error: {len(tmb_paths)} --tmb file(s) provided but {n_types} --gam "
                 f"file(s) given; counts must match.")

    # ------------------------------------------------------------------
    # Step 0: Load data
    # ------------------------------------------------------------------
    type_label = f" ({n_types} mutation types)" if is_multitype else ""
    print(f"Loading GAM{'s' if is_multitype else ''} ...")

    gams = []
    gene_names = sample_names = None
    for i, path in enumerate(gam_paths):
        print(f"  [{i+1}] {path}")
        g, gn, sn = load_gam(path)
        if i == 0:
            gene_names, sample_names = gn, sn
        else:
            if gn != gene_names:
                sys.exit(f"Error: gene list in '{path}' differs from '{gam_paths[0]}'.")
            if sn != sample_names:
                sys.exit(f"Error: sample list in '{path}' differs from '{gam_paths[0]}'.")
        gams.append(g)

    n_g, n_t = gams[0].shape
    print(f"  {n_g} genes x {n_t} samples{type_label}")

    # Combined GAM: element-wise OR of all mutation-type GAMs
    gam = np.maximum.reduce(gams).astype(np.int8)

    # Filter genes with too few mutations in the combined GAM
    row_sums = gam.sum(axis=1)
    keep = row_sums >= args.min_mut
    n_removed = int((~keep).sum())
    if n_removed:
        print(f"  Excluding {n_removed} gene(s) with < {args.min_mut} mutated samples.")
        gams = [g[keep] for g in gams]
        gam  = gam[keep]
        gene_names = [gn for gn, k in zip(gene_names, keep) if k]
        n_g = gam.shape[0]
        print(f"  {n_g} genes retained.")

    classes = None
    if args.maf:
        print(f"Loading TMB from MAF: {args.maf} ...")
        tmb_single = load_maf(args.maf, sample_names)
        tmbs = [tmb_single]
        tmb  = tmb_single
        print(f"  Total non-silent mutations: {int(tmb.sum())}")
    elif tmb_paths is not None:
        print(f"Loading pre-calculated TMB from {n_types} file(s) ...")
        tmbs = []
        for path in tmb_paths:
            t, cls = load_tmb(path, sample_names)
            tmbs.append(t)
            if cls is not None and classes is None:
                classes = cls
        tmb = np.stack(tmbs).sum(axis=0)
        print(f"  Total TMB: {int(tmb.sum())}")
        if classes is not None:
            unique_cls, cls_counts = np.unique(classes, return_counts=True)
            print(f"  Sample classes: { {c: int(n) for c, n in zip(unique_cls, cls_counts)} }")
    else:
        print("No TMB source provided — estimating TMB from GAM column sums.")
        tmbs = [tmb_from_gam(g) for g in gams]
        tmb  = np.stack(tmbs).sum(axis=0)

    zero_tmb = int((tmb == 0).sum())
    if zero_tmb:
        print(f"  Warning: {zero_tmb} sample(s) have TMB = 0.")

    # ------------------------------------------------------------------
    # Step 1: Expected mutation matrix and rcRAS simulations
    # ------------------------------------------------------------------
    class_slices = None

    if classes is not None:
        label = "per class, per type" if is_multitype else "per class"
        print(f"Computing expected mutation matrix E ({label}) ...")
        if is_multitype:
            E, col_order, class_slices = compute_E_matrix_multitype_by_class(
                gams, tmbs, classes)
        else:
            E, col_order, class_slices = compute_E_matrix_by_class(
                gams[0], tmbs[0], classes)
        # Reorder combined GAM and metadata to match class-sorted column layout of E
        gam          = gam[:, col_order]
        tmb          = tmb[col_order]
        classes      = classes[col_order]
        sample_names = [sample_names[i] for i in col_order]
        # For multi-type, gene_counts_k in class_slices came from a single-type GAM;
        # recompute them from the reordered combined GAM so simulations are correct.
        if is_multitype:
            class_slices = [
                (s, e, gam[:, s:e].sum(axis=1).astype(int))
                for s, e, _ in class_slices
            ]
    else:
        print("Computing expected mutation matrix E ...")
        if is_multitype:
            E = compute_E_matrix_multitype(gams, tmbs)
        else:
            E = compute_E_matrix(gams[0], tmbs[0])

    gene_counts = gam.sum(axis=1).astype(int)   # observed mutations per gene
    gene_freq   = gene_counts / n_t

    print(f"Running {args.N} rcRAS simulations ...")
    sims = run_simulations(E, args.N, gene_counts, seed=args.seed,
                           class_slices=class_slices)

    print(f"Filtering simulations (removing worst {args.filter_pct*100:.0f}%) ...")
    sims = filter_simulations(sims, gam, pct=args.filter_pct)
    print(f"  {len(sims)} simulations retained.")

    # ------------------------------------------------------------------
    # Step 2: Observed weighted co-mutation
    # ------------------------------------------------------------------
    print("Computing penalty vector and observed weighted co-mutation ...")
    p       = compute_penalty_vector(tmb, tau=args.tau, lam=args.lam, classes=classes)
    wCO_obs = compute_weighted_comutation(gam, p)

    # ------------------------------------------------------------------
    # Step 3: Statistical significance
    # ------------------------------------------------------------------
    print("Estimating effect sizes and FDR ...")
    results = run_significance_analysis(
        wCO_obs=wCO_obs,
        sims=sims,
        p=p,
        gene_freq=gene_freq,
        gene_names=gene_names,
        gam=gam,
        target_fdr=args.fdr,
    )

    n_sig = int(results['significant'].sum())
    print(f"Significant co-mutations at FDR < {args.fdr}: {n_sig} gene pairs")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    save_results(results, args.output)
    print(f"Results written to {args.output}")

    return results


if __name__ == '__main__':
    main()
