# MInA: Micro-Information Analysis

Code and data accompanying the paper

**[Learning the information contained in the microstructure of data distributions]**, [Lucrezia L. Baldo, Chiara Lionello, Edward D. Donkor, Matteo Becchi, and Giovanni M. Pavan*]


## Repository structure

```
MInA/
├── array/
│   ├── descriptor/         # single-particle descriptors (raw and denoised)
│   ├── soap/               # empty in this repository: SOAP arrays to be added (see "Data")
│   └── synthetic/          # synthetic distributions: gauss_<n>.npy, nongauss_<n>.npy
├── code/
│   ├── 00_MInA_functions.py      # MInA functions and plotting utilities
│   ├── 01_syntehtic_MInA.py      # analysis of the synthetic distributions
│   ├── 02_descriptors_MInA.py    # analysis of the descriptors
│   ├── 03_soap_denoised_MInA.py  # analysis of the denoised SOAP components
│   └── 04_soap_raw_MInA.py       # analysis of the raw SOAP components
├── csv/
│   ├── MInA_descriptors.csv
│   ├── MInA_soap_denoised.csv
│   ├── MInA_soap_raw.csv
│   └── MInA_synthetic.csv
├── figures/
│   ├── descriptors/
│   ├── soap_denoised/
│   ├── soap_raw/
│   │   ├── 00_structure_diagram/           # bar plots
│   │   └── 01_information_structure_plot/  # scatter plots and legends
│   └── synthetic/
└── README.md
```

`00_MInA_functions.py` is a module: it contains only functions and is imported
by the numbered scripts. Each numbered script is self-contained, reads its
input from `array/`, writes its results to `csv/` and its figures to
`figures/`.

## Data

**Synthetic distributions** (`array/synthetic/`). Files are named
`<kind>_<n_true>.npy`, where `kind` is `gauss` (mixture of Gaussians only) or
`nongauss` (mixture with a non-Gaussian component), and `n_true` is the number
of Gaussian components used to generate the distribution.

**Molecular dynamics trajectories.** The trajectories from which all
descriptors were computed are not distributed here. They are described in, and
should be obtained from:

> C. Caruso, A. Cardellini, M. Crippa, D. Rapetti and G. M. Pavan,
> TimeSOAP: Tracking high-dimensional fluctuations in complex molecular systems
> via time variations of SOAP spectra, *J. Chem. Phys.* **158**(21) (2023).
> [doi:10.1063/5.0147025](https://doi.org/10.1063/5.0147025)

**Single-particle descriptors** (`array/descriptor/`). LENS, `dist_5`, `nn`,
OTO and `vel`, each in a raw version and in a denoised version (prefix `sp_10_`,
denoted ⟨·⟩ in the figures). Their computation is described in:

> S. Martino, D. Doria, C. Lionello, M. Becchi and G. M. Pavan,
> A data driven approach to classify descriptors based on their efficiency in
> translating noisy trajectories into physically-relevant information,
> *Mach. Learn.: Sci. Technol.* **6**, 035039 (2025).
> [doi:10.1088/2632-2153/adfa66](https://doi.org/10.1088/2632-2153/adfa66)

**SOAP components** (`array/soap/`). Arrays of shape
`(n_particles, n_frames, n_components)`; every component `[:, :, k]` is
analysed as a separate distribution. The computation of SOAP from the
trajectories is described in:

> C. Lionello, M. Becchi, S. Martino and G. M. Pavan,
> Relevant, Hidden, and Frustrated Information in High-Dimensional Analyses of
> Complex Dynamical Systems with Internal Noise,
> *J. Chem. Theory Comput.* **21**, 6683–6697 (2025).
> [doi:10.1021/acs.jctc.5c00374](https://doi.org/10.1021/acs.jctc.5c00374)

Because of their size (about 11 GB each), the two SOAP arrays are not
included in this repository, and the folder `array/soap/` is provided empty.
To run `03_soap_denoised_MInA.py` and `04_soap_raw_MInA.py`, compute SOAP on
the trajectories of Caruso et al. following the procedure of Lionello et al.,
and save the resulting arrays in `array/soap/` as:

- `soap_denoised.npy`: denoised SOAP components, used by `03_soap_denoised_MInA.py`;
- `soap_raw.npy`: raw SOAP components, used by `04_soap_raw_MInA.py`.

Both arrays must have shape `(n_particles, n_frames, n_components)`. The CSV
files and figures obtained from these arrays in the paper are included in
`csv/` and `figures/`.

## Requirements

Tested with Python 3.13.

```bash
pip install numpy scipy pandas matplotlib seaborn infomeasure
```

## Usage

Run the scripts from any directory; paths are resolved relative to the
repository:

```bash
python code/01_syntehtic_MInA.py
python code/02_descriptors_MInA.py
python code/03_soap_denoised_MInA.py
python code/04_soap_raw_MInA.py
```

The SOAP analyses iterate over hundreds of components and can take a long
time; a progress bar is printed on stderr. To keep them running after the
terminal is closed:

```bash
python code/03_soap_denoised_MInA.py
```

The SOAP array is memory-mapped and read in blocks of components; the memory
used per block is set by `MEM_BUDGET_GB` at the top of the script. A larger
budget reduces the number of passes over the file.

### Parameters used in the paper

| Script | `bw_adjust` | `prominence` | `peak_height_threshold` | `resid_thr` | entropy | `n_bins` |
|---|---|---|---|---|---|---|
| 01 synthetic | 0.3 | 0.003 | 0.05 | 0.5 | histogram | 50 |
| 02 descriptors | 2 | 0.0001 | 0.05 | 0.5 | histogram | 50 |
| 03 SOAP denoised | 2 | 0.0001 | 0.05 | 0.5 | histogram | 50 |
| 04 SOAP raw | 2 | 0.0001 | 0.05 | 0.5 | histogram | 50 |

## Output

**CSV files** (`csv/`). One row per distribution (array or SOAP component):

| Column | Method | Meaning |
|---|---|---|
| `n_gaussians` | population | number of fitted Gaussians |
| `H0` | population | Shannon entropy of the whole distribution [bit] |
| `Hclust` | population | population-weighted entropy of the partition [bit] |
| `dI`, `dI_rel` | population | information gain `H0 - Hclust` and `dI / H0` |
| `Ew`, `Ewng` | population | Σ_k W_k and Σ_k (1 − W_k), with W_k the fraction of samples assigned to the first k Gaussians |
| `w_ng`, `H_ng` | population | weight and entropy of the residual population |
| `mu_i`, `w_i`, `H_i` | population | mean, weight and entropy of the i-th Gaussian population |
| `npeaks` | area | number of fitted Gaussians |
| `Ag`, `Ang` | area | areas under and above the cumulative explained-fraction curve |
| `h_g`, `h_ng` | area | explained and unexplained area with all Gaussians included |
| `Egef`, `Eguf` | area | Σgef and Σguf |

The synthetic CSV also contains `kind` and `n_true`.

**Figures** (`figures/`). For each analysis:

- `dI_vs_<metric>.png`: ΔI against Egef, Eguf, Ew and Ewng, coloured per
  array (or per highlighted SOAP component), and `dI_vs_<metric>_npeaks.png`,
  coloured by number of Gaussians;
- `Eguf_vs_Egef.png` and `Ewng_vs_Ew.png`;
- `legend.png` and `legend_npeaks.png`;
- `barplots/`: cumulative explained-fraction bar plots for each method (for
  SOAP, only for the highlighted components).

The bar plots are saved in `figures/*/00_structure_diagram/` and the scatter plots and legends in `figures/*/01_information_structure_plot/`.

## Citation

If you use this code, please cite:

```bibtex
[]
```


## Contact

Giovanni M. Pavan, GMP Lab, Politecnico di Torino — [giovanni.pavan@polito.it]