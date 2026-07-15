# PH_regolith_analysis

Persistent homology (PH) analysis of the void structure of low-binder
resin-coated lunar regolith simulant blocks, and PH texture analysis of
optical micrographs of the raw simulant.

This repository accompanies the paper:

> Shimizu et al., *Low-Binder Resin-Coated Regolith Blocks for Lunar
> Thermal Energy Storage* (submitted).

## Specimens

| Tag    | Classification                      | Resin (PAI) | Quality   |
|--------|-------------------------------------|-------------|-----------|
| mesh01 | classified (fines < 0.1 mm removed) | 3 wt%       | good      |
| mesh02 | unclassified                        | 3 wt%       | defective |
| mesh03 | unclassified                        | 2 wt%       | defective |

Void-phase point clouds were extracted from micro-XCT scans
(Shimadzu XDimensus 300, 5 um voxel; segmentation with VGSTUDIO).

## Repository layout

```
mesh/
  data/            micro-XCT void point clouds (x y z, one point per line)
  scripts/         analysis scripts (see workflow below)
  cache/           precomputed PD pairs (*.npz) so that plotting/statistics
                   scripts run without recomputing persistence diagrams
  output/          generated figures and statistics
  other_scripts/   exploratory scripts and notebook
  other_data/      small subsampled point clouds
  other_png/       point-cloud visualisations
  pdgm/            HomCloud persistence diagram files
images/
  raw/             optical micrographs of the raw simulant
                   (img260512-13 = unclassified, img260512-15 = classified)
  *.py             2D image PH and specific-surface-area analysis
```

## Installation

Python >= 3.10.

```bash
pip install -r requirements.txt
```

## Workflow (mesh / micro-XCT)

All scripts are standalone and use paths relative to the repository, so
they can be run from any working directory:

```bash
python mesh/scripts/pointcloud_nature.py          # local PCA: point cloud = void surface
python mesh/scripts/compute_pd1_full.py           # H1 PDs, spatial chunking (slow)
python mesh/scripts/compute_pd2_full.py           # H2 PDs, spatial chunking (slow)
python mesh/scripts/global_subsample_pd.py        # whole-volume PDs (N = 50,000 subsample)
python mesh/scripts/compute_betti_full.py         # persistent Betti numbers
python mesh/scripts/clump_size_hist.py            # pore count / size distributions
python mesh/scripts/clump_size_stats.py           # Kruskal-Wallis, Cliff's delta, ECDF
python mesh/scripts/validate_compute_replicates.py  # bootstrap replicates (slow)
python mesh/scripts/validate_stability_distance.py  # V1 stability + V2 bottleneck/MDS
python mesh/scripts/validate_classical_stats.py     # V3 classical spatial statistics
python mesh/scripts/validate_emptyspace.py          # V3 empty-space function
python mesh/scripts/pore_network_demo.py            # pore-network (1-skeleton) consistency
```

The heavy PD computations (compute_pd1_full.py, compute_pd2_full.py,
validate_compute_replicates.py) cache their results as *.npz under
mesh/cache/; the precomputed caches are included, so all downstream
plotting and statistics scripts run in seconds. Delete the cache files to
recompute from the raw point clouds.

## Workflow (images / optical microscopy)

```bash
python images/image_pd1.py        # grayscale conversion + H1 PDs (HomCloud bitmap filtration)
python images/microscope_ph.py    # persistence distributions, H0/H1 metrics
python images/microscope_ssa.py   # edge-density / specific-surface-area proxies
```

## Requirements

See requirements.txt. Persistent homology is computed with
[HomCloud](https://homcloud.dev/) (alpha filtration for point clouds,
bitmap levelset filtration for images).

## Citation

If you use this code or data, please cite the paper above.
