# Figure-generation scripts for the accompanying manuscript

This archive contains the author's 24 Python scripts for generating main-text and supplementary figures. It was assembled from the supplied `code.zip` on 2026-11-02. The archive contains **code only**: input NetCDF files, the historical boundary shapefile, and generated figure files are not included. The code and accompanying documentation are released under the MIT License; see `LICENSE`.

## What is in the archive

All scripts are in `code/`. Keep them together because several scripts import or load their neighbours.

| Figure or role | Script(s) | Notes |
| --- | --- | --- |
| Main Fig. 1 | `Fig1.py`, `Fig1a.py`, `Fig1b.py` | `Fig1.py` assembles the two panels. The panel scripts can also run separately. |
| Main Fig. 2 | `Fig2.py`, `Fig2a.py`–`Fig2d.py` | `Fig2.py` assembles the four panels. |
| Main Fig. 3 | `Fig3.py` | Calculates model and sea-ice quantities and writes a NetCDF result as well as figures. |
| Main Fig. 4 | `Fig4.py`, `Fig4a.py`–`Fig4e.py`, `Fig4common.py` | `Fig4common.py` provides shared functions; keep it next to all Fig. 4 scripts. |
| Main Fig. 5 | `Fig5.py` | Produces paired-member raincloud plots. |
| Supplementary Fig. S1 | `FigS1.py` | Uses a historical European-boundary shapefile dated 1942-01-31. |
| Supplementary Fig. S2 | `FigS2.py` | Uses monthly and daily Moscow temperature NetCDF files. |
| Supplementary Fig. S3 | `FigS3.py` | Plots PDO, ENSO, raw climate anomalies, and anomalies with PDO and ENSO removed. |
| Supplementary Fig. S4 | `FigS4.py` | Uses observational surface-air-temperature datasets and station data. |
| Supplementary Fig. S5 | `FigS5.py` | Uses reconstructed and HadISST sea-ice data. |
| Supplementary Fig. S6 | `FigS6.py`, `Fig4e.py`, `Fig4common.py` | Produces a two-panel variant of Fig. 4e. |
| Additional member-specific panels | `FigS789_member2_a_d_black_frames.py`, Fig. 4 scripts | Its filename suggests S7–S9, but its actual outputs are three member-specific Fig. 4 composites. **Confirm the manuscript figure numbering before citing this row.** |

The archive should be described as **figure-generation code** unless all other analysis code used by the manuscript is also deposited.

## Software environment

The inspected Anaconda base installation is CPython 3.12.3 with Matplotlib 3.8.4. Existing figure-file metadata identifies Matplotlib 3.8.4, and compiled script caches identify the Python 3.12 series; neither establishes the precise Python patch version used for every original figure. `requirements.txt` records the currently inspected versions of directly used packages and common NetCDF/GIS backends. It is an **observed environment snapshot**, not a tested lockfile for the original runs.

One possible installation workflow is:

```bash
python -m venv .venv
# Activate .venv according to your operating system.
python -m pip install -r requirements.txt
```

The scripts have **not** been rerun end-to-end from this archive because their input datasets were not supplied. Package installation alone will not reproduce the figures.

## Input data and local paths

Before running a script, obtain its input data and edit the path constants near the top of that script. The original scripts use absolute paths on local `E:` and `F:` drives. `LOCAL_PATHS.tsv` inventories every source line containing such a path. `DATA_SOURCES_TEMPLATE.csv` provides blank fields for the authors to fill with each dataset's actual repository, identifier, version or access date, and reuse conditions. **The filenames in the code do not verify data provenance or permission to redistribute a dataset.**

Major inputs referenced by the scripts include:

| Input family inferred from file names and code | Used by |
| --- | --- |
| GISTEMP, CRU TS 4.05, Berkeley, Nansen, GHCN-M station, PDO, Niño 3.4, and HadSLP2 files | Fig. 1 and several supplementary scripts |
| Reconstructed sea-ice concentration and HadISST sea-ice files | Figs. 2–3 and Fig. S5 |
| CAM experiment output for HadISST and reconstructed-sea-ice experiments, including temperature/geopotential-height, winds, and surface temperature | Figs. 2–5 and Fig. S6/member-specific panels |
| Monthly and daily Moscow temperature NetCDF files | Fig. S2 |
| Historical European-boundary shapefile, including its required companion files | Fig. S1 |

Some scripts also use Cartopy's map features. A system may need its Natural Earth assets locally or a network connection for the first run.

## Suggested execution order

After installing dependencies and setting input paths, run the main figure assembly scripts individually from the archive root:

```bash
python code/Fig1.py
python code/Fig2.py
python code/Fig3.py
python code/Fig4.py
python code/Fig5.py
```

Run each supplementary script separately as needed. `FigS6.py` and `FigS789_member2_a_d_black_frames.py` load the Fig. 4 modules. Some scripts show an interactive figure window (`plt.show()`); inspect each script's save and display settings before using a headless server. Output locations vary by script and may need to be changed along with input paths.

## Change from the supplied source ZIP

The only source-code change in this archive is the `OUTPUT_STEM` path in `code/Fig5.py`: the personal desktop output path was replaced with an `outputs/Fig5_6panels` path beside that script. All numerical and plotting logic is otherwise preserved.

## Remaining steps for full reproducibility

The code archive is public. The following improvements would make it easier for readers to reproduce the figures:

1. Fill in `DATA_SOURCES_TEMPLATE.csv` using verified dataset URLs/DOIs, versions, access conditions, and citations; describe any inputs that cannot be redistributed.
2. Confirm that the script-to-figure mapping matches the submitted manuscript, especially the member-specific supplemental figures.
3. Test the scripts using the intended data and record any deviations from the observed package versions.
4. Replace any remaining machine-specific paths with a documented configuration if a directly runnable public package is required.


