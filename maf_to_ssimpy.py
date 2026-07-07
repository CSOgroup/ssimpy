"""maf_to_ssimpy.py — Convert a MAF file to ssimpy input files (GAM + TMB).

Usage (combined GAM only):
    python maf_to_ssimpy.py --maf input.maf --prefix cohort --output-dir ./data

Usage (with separate missense / truncating files):
    python maf_to_ssimpy.py --maf input.maf --split-by-type --prefix cohort

Usage (with sample class metadata):
    python maf_to_ssimpy.py --maf input.maf --metadata meta.tsv --split-by-type

Usage (restrict to a gene list and set mutation thresholds):
    python maf_to_ssimpy.py --maf input.maf --gene-list cancer_genes.txt \
        --min-samples 2 --min-mutations 5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


# ── Variant classification groups ────────────────────────────────────────────

_SILENT = {'silent', 'synonymous_mutation', 'synonymous'}

_MISSENSE = {'missense_mutation', 'in_frame_del', 'in_frame_ins'}

_TRUNCATING = {
    'nonsense_mutation', 'frame_shift_del', 'frame_shift_ins',
    'splice_site', 'splice_region', 'nonstop_mutation',
    'translation_start_site',
}


def classify(vc: str) -> str:
    """Return 'silent', 'missense', 'truncating', or 'other' for a variant classification."""
    v = vc.strip().lower()
    if v in _SILENT:
        return 'silent'
    if v in _MISSENSE:
        return 'missense'
    if v in _TRUNCATING:
        return 'truncating'
    return 'other'


# ── Core conversion ──────────────────────────────────────────────────────────

def maf_to_ssimpy(
    maf_path: str,
    output_dir: str = '.',
    prefix: str = 'ssimpy',
    split_by_type: bool = False,
    metadata_path: str = None,
    gene_list_path: str = None,
    min_samples: int = 2,
    min_mutations: int = 1,
) -> None:
    # ── Load MAF ─────────────────────────────────────────────────────────────
    maf = pd.read_csv(maf_path, sep='\t', comment='#', low_memory=False)
    maf.columns = maf.columns.str.strip().str.lower()

    for col in ('hugo_symbol', 'tumor_sample_barcode', 'variant_classification'):
        if col not in maf.columns:
            sys.exit(f"Error: MAF is missing required column '{col}'. "
                     f"Found: {list(maf.columns)}")

    maf = maf[['hugo_symbol', 'tumor_sample_barcode', 'variant_classification']].copy()
    maf.columns = ['gene', 'sample', 'vc']
    maf['vc_class'] = maf['vc'].map(classify)

    # ── Remove silent mutations ───────────────────────────────────────────────
    non_silent = maf[maf['vc_class'] != 'silent'].copy()

    all_samples = sorted(non_silent['sample'].unique())
    n_samples_raw = len(all_samples)

    # ── Load gene list (optional) ─────────────────────────────────────────────
    gene_whitelist = None
    if gene_list_path:
        with open(gene_list_path) as fh:
            gene_whitelist = {
                line.strip() for line in fh
                if line.strip() and not line.startswith('#')
            }
        print(f"Gene list     : {len(gene_whitelist)} genes loaded from {gene_list_path}")
        missing = gene_whitelist - set(non_silent['gene'].unique())
        if missing:
            print(f"  Warning: {len(missing)} gene(s) in list not found in MAF")

    # ── Load metadata (optional) ──────────────────────────────────────────────
    class_map = {}
    if metadata_path:
        meta = pd.read_csv(metadata_path, sep='\t')
        meta.columns = meta.columns.str.lower()
        if 'sample' not in meta.columns or 'class' not in meta.columns:
            sys.exit("Error: metadata file must have 'sample' and 'class' columns.")
        class_map = meta.set_index('sample')['class'].to_dict()

    # ── Build GAMs ────────────────────────────────────────────────────────────
    def build_gam(df, samples, genes):
        """Binary GAM from a subset of mutations."""
        gam = pd.DataFrame(0, index=genes, columns=samples, dtype=np.int8)
        for (gene, sample), _ in df.groupby(['gene', 'sample']):
            if gene in gam.index and sample in gam.columns:
                gam.at[gene, sample] = 1
        gam.index.name = 'Gene'
        return gam

    def build_tmb(df, samples, tmb_col='tmb'):
        counts = df.groupby('sample').size()
        tmb = pd.DataFrame({'sample': samples, tmb_col: [counts.get(s, 0) for s in samples]})
        if class_map:
            tmb['class'] = [class_map.get(s, 'unknown') for s in samples]
        return tmb

    # Combined: all non-silent
    # Apply filters: gene list → min_samples → min_mutations
    combined = non_silent.copy()

    if gene_whitelist is not None:
        combined = combined[combined['gene'].isin(gene_whitelist)]

    gene_sample_counts = combined.groupby('gene')['sample'].nunique()
    gene_mut_counts    = combined.groupby('gene').size()

    valid_genes = sorted(
        gene_sample_counts.index[
            (gene_sample_counts >= min_samples) &
            (gene_mut_counts >= min_mutations)
        ]
    )

    combined = combined[combined['gene'].isin(valid_genes)]

    gam_combined = build_gam(combined, all_samples, valid_genes)
    # TMB counts ALL non-silent mutations per sample, not just those in retained genes
    tmb_combined = build_tmb(non_silent, all_samples, tmb_col='tmb')

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"MAF file      : {maf_path}")
    print(f"Total rows    : {len(maf):,}")
    print(f"Samples       : {n_samples_raw}")
    print(f"Silent removed: {(maf['vc_class'] == 'silent').sum():,}")
    print(f"Missense      : {(non_silent['vc_class'] == 'missense').sum():,}")
    print(f"Truncating    : {(non_silent['vc_class'] == 'truncating').sum():,}")
    print(f"Other non-silent: {(non_silent['vc_class'] == 'other').sum():,}")
    print(f"Genes retained: {len(valid_genes)}  "
          f"(>= {min_samples} sample(s), >= {min_mutations} mutation(s)"
          + (", restricted to gene list" if gene_whitelist else "") + ")")

    # ── Write combined output ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    gam_path = os.path.join(output_dir, f'{prefix}_gam.tsv')
    tmb_path = os.path.join(output_dir, f'{prefix}_tmb.tsv')
    gam_combined.to_csv(gam_path, sep='\t')
    tmb_combined.to_csv(tmb_path, sep='\t', index=False)
    print(f"\nWrote: {gam_path}  ({gam_combined.shape[0]} genes x {gam_combined.shape[1]} samples)")
    print(f"Wrote: {tmb_path}")

    # ── Write per-type output ─────────────────────────────────────────────────
    if split_by_type:
        for type_name, type_class in [('missense', 'missense'), ('truncating', 'truncating')]:
            subset = non_silent[non_silent['vc_class'] == type_class].copy()
            # GAM restricted to valid genes; TMB counts all mutations of this type
            gam_type = build_gam(subset[subset['gene'].isin(valid_genes)], all_samples, valid_genes)
            tmb_type = build_tmb(subset, all_samples, tmb_col='mutation')

            gam_p = os.path.join(output_dir, f'{prefix}_gam_{type_name}.tsv')
            tmb_p = os.path.join(output_dir, f'{prefix}_tmb_{type_name}.tsv')
            gam_type.to_csv(gam_p, sep='\t')
            tmb_type.to_csv(tmb_p, sep='\t', index=False)

            n_ones = int(gam_type.values.sum())
            print(f"Wrote: {gam_p}  ({n_ones} mutations)")
            print(f"Wrote: {tmb_p}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Convert a MAF file to ssimpy input files (GAM + TMB).'
    )
    p.add_argument('--maf', required=True,
                   help='Input MAF TSV file')
    p.add_argument('--output-dir', default='.',
                   help='Directory for output files (default: current directory)')
    p.add_argument('--prefix', default='ssimpy',
                   help='Filename prefix for output files (default: ssimpy)')
    p.add_argument('--split-by-type', action='store_true',
                   help='Also produce separate missense and truncating GAM/TMB files')
    p.add_argument('--metadata', default=None,
                   help='Optional TSV with sample and class columns to annotate TMB files')
    p.add_argument('--gene-list', default=None,
                   help='Text file with one gene name per line; only these genes are '
                        'included in the GAM (lines starting with # are ignored)')
    p.add_argument('--min-samples', type=int, default=2,
                   help='Minimum number of mutated samples to retain a gene (default: 2)')
    p.add_argument('--min-mutations', type=int, default=1,
                   help='Minimum total number of non-synonymous mutations across all '
                        'samples to retain a gene (default: 1)')
    return p.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    maf_to_ssimpy(
        maf_path=args.maf,
        output_dir=args.output_dir,
        prefix=args.prefix,
        split_by_type=args.split_by_type,
        metadata_path=args.metadata,
        gene_list_path=args.gene_list,
        min_samples=args.min_samples,
        min_mutations=args.min_mutations,
    )
