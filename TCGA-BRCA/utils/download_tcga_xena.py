#!/usr/bin/env python3
"""
Script to download TCGA datasets from UCSC Xena using xenaPython.

UCSC Xena provides access to TCGA data including:
- Gene expression (RNA-seq, microarray)
- Copy number variation
- Somatic mutations
- Clinical data
- Phenotype data

This script demonstrates how to:
1. Query available TCGA cohorts
2. List datasets for a specific cohort
3. Download data for TCGA-BRCA

Downloaded artifacts are saved under `TCGA-BRCA/data/TCGA_data`.

Usage:
    python download_tcga_xena.py
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xenaPython as xena

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "TCGA_data"
COHORT_NAME = "TCGA Breast Cancer (BRCA)"  # or "GDC TCGA Breast Cancer (BRCA)"

# Xena hubs
TCGA_HUB = "https://tcga.xenahubs.net"
GDC_HUB = "https://gdc.xenahubs.net"
PANCAN_HUB = "https://pancanatlas.xenahubs.net"


def list_all_cohorts(hub):
    """List all available cohorts from a Xena hub."""
    try:
        cohorts = xena.all_cohorts(hub, [])
        print(f"\n{'='*80}")
        print(f"Available cohorts from {hub}")
        print(f"{'='*80}")
        for i, cohort in enumerate(sorted(cohorts), 1):
            print(f"{i:3d}. {cohort}")
        return cohorts
    except Exception as e:
        print(f"Error listing cohorts: {e}")
        return []


def list_datasets_for_cohort(hub, cohort):
    """List all datasets available for a specific cohort."""
    try:
        print(f"\n{'='*80}")
        print(f"Datasets for: {cohort}")
        print(f"{'='*80}")

        datasets = xena.dataset_list(hub, cohort)
        print(f"Found {len(datasets)} datasets\n")

        dataset_info = []
        for i, dataset in enumerate(datasets, 1):
            try:
                metadata = xena.dataset_metadata(hub, dataset)
                info = {
                    'dataset': dataset,
                    'type': metadata.get('type', 'unknown'),
                    'label': metadata.get('label', dataset),
                    'description': metadata.get('description', ''),
                }
                dataset_info.append(info)

                print(f"{i:3d}. {info['label'][:70]}")
                print(f"     Type: {info['type']}")
                print(f"     Dataset: {dataset[:70]}")
                print()

            except Exception as e:
                print(f"{i:3d}. {dataset}")
                print(f"     Error getting metadata: {e}")
                print()

        return dataset_info

    except Exception as e:
        print(f"Error listing datasets: {e}")
        import traceback
        traceback.print_exc()
        return []


def download_dataset(hub, dataset, cohort, output_dir):
    """
    Download a specific dataset from Xena.

    Args:
        hub: Xena hub URL
        dataset: Dataset identifier
        cohort: Cohort name
        output_dir: Directory to save the data
    """
    try:
        print(f"\nDownloading: {dataset}")

        # Get samples for this cohort
        samples = xena.cohort_samples(hub, cohort, None)
        print(f"Total samples: {len(samples)}")

        # Get fields/features in the dataset
        fields = xena.dataset_field(hub, dataset)
        print(f"Total features: {len(fields)}")

        # Download the data
        # For gene expression data, we fetch values for all genes and samples
        print("Fetching data (this may take a while)...")

        # Use dataset_fetch to get the data
        data = xena.dataset_fetch(hub, dataset, samples, fields[:100])  # Start with first 100 features

        # Convert to DataFrame
        df = pd.DataFrame(data, index=fields[:100], columns=samples)

        # Save the data
        output_path = output_dir / f"{dataset.replace('/', '_')}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(df, f)

        print(f"Saved to: {output_path}")
        print(f"Shape: {df.shape}")

        return df

    except Exception as e:
        print(f"Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def download_gene_expression(hub, cohort, genes, output_dir):
    """
    Download gene expression data for specific genes.

    Args:
        hub: Xena hub URL
        cohort: Cohort name
        genes: List of gene names
        output_dir: Directory to save data
    """
    try:
        # Find gene expression datasets
        datasets = xena.dataset_list(hub, cohort)

        # Look for RNA-seq or expression dataset
        expr_datasets = [d for d in datasets if 'expression' in d.lower() or 'rnaseq' in d.lower()]

        if not expr_datasets:
            print("No gene expression datasets found")
            return None

        dataset = expr_datasets[0]
        print(f"Using dataset: {dataset}")

        # Get samples
        samples = xena.cohort_samples(hub, cohort, None)

        # Fetch data for specific genes
        data = xena.dataset_fetch(hub, dataset, samples, genes)

        # Create DataFrame
        df = pd.DataFrame(data, index=genes, columns=samples)

        # Save
        output_path = output_dir / "gene_expression.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(df, f)

        print(f"Downloaded expression data for {len(genes)} genes across {len(samples)} samples")
        print(f"Saved to: {output_path}")

        return df

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function."""

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("TCGA Data Downloader using UCSC Xena")
    print("=" * 80)

    # Step 1: List available cohorts
    print("\nStep 1: Checking available hubs and cohorts...")

    for hub_name, hub_url in [("TCGA", TCGA_HUB), ("GDC", GDC_HUB), ("PanCan", PANCAN_HUB)]:
        print(f"\nTrying {hub_name} hub: {hub_url}")
        try:
            cohorts = list_all_cohorts(hub_url)

            # Find BRCA
            brca_cohorts = [c for c in cohorts if 'BRCA' in c.upper()]
            if brca_cohorts:
                print(f"\n✓ Found BRCA cohort: {brca_cohorts[0]}")

                # Try to list datasets
                print(f"\nStep 2: Listing datasets for BRCA...")
                datasets = list_datasets_for_cohort(hub_url, brca_cohorts[0])

                if datasets:
                    print(f"\n✓ Successfully accessed {len(datasets)} datasets")
                    print("\nYou can now download specific datasets using the functions above.")
                    print("\nExample: Download specific genes")
                    print("genes = ['TP53', 'BRCA1', 'BRCA2', 'EGFR', 'HER2']")
                    print(f"df = download_gene_expression('{hub_url}', '{brca_cohorts[0]}', genes, OUTPUT_DIR)")
                    break

        except Exception as e:
            print(f"✗ Error with {hub_name} hub: {e}")
            continue

    print("\n" + "=" * 80)
    print("Note: Xena hubs may experience temporary server issues.")
    print("If you encounter errors, try again later or use the Xena Browser web interface:")
    print("https://xenabrowser.net/datapages/")
    print("=" * 80)


if __name__ == "__main__":
    main()
