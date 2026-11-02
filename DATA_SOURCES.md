# Data sources identified in the manuscript

This inventory transcribes sources and versions stated in the manuscript draft titled **“Arctic sea-ice loss may have contributed to extreme winter cold in Eastern Europe and western Russia in the early 1940s”**. It identifies where readers can start looking for inputs. The manuscript's links are not a guarantee that each exact file used by the scripts is downloadable there. The local filenames, access dates, redistribution rights, and file-to-record correspondence still require author verification in `DATA_SOURCES_TEMPLATE.csv`.

The related article by [Li et al. (2026), *Geophysical Research Letters*, doi:10.1029/2025GL121178](https://doi.org/10.1029/2025GL121178) provides further provenance. Its Section 2.2 (p. 3) says that IAPICE v1.1 uses CRU TS v4.05 surface air temperature, ERSST v5 sea-surface temperature, and HadSLP2 sea-level pressure as predictor fields. Section 2.3 (p. 4) describes the two 20-member CAM4 experiments, which prescribe ERSST v5 sea-surface temperature and respectively HadISST v1.1 or IAPICE v1.1 sea-ice concentration. Its Data Availability Statement (p. 8) lists public source locations for observational, reanalysis, and reference products, the CAM4 model-software page, and a Zenodo record for that article's plotting code. Those references document the provenance of inputs and methods; the statement does not give a distinct download URL or DOI for the IAPICE **v1.1** reconstructed file(s) or these study-specific CAM4 ensemble outputs.

## Observational and reanalysis inputs

| Input described in the manuscript | Version or derivation stated there | Location stated there |
| --- | --- | --- |
| NASA GISTEMP surface air temperature | v4; 2° × 2° grid with 1,200-km smoothing | [NASA GISTEMP](https://data.giss.nasa.gov/gistemp/) |
| Berkeley Earth surface air temperature | 1° × 1° grid; release version not specified | [Berkeley Earth data](https://berkeleyearth.org/data/) |
| Nansen-SAT surface air temperature | 2.5° × 2.5° grid; release version not specified | [Nansen International Environmental and Remote Sensing Centre](https://www.niersc.spb.ru/); direct file link not specified |
| CRU TS surface air temperature | v4.05; 0.5° × 0.5° grid | [CRU TS v4.05](https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.05/) |
| GHCN-Monthly station temperature | v4 | [NOAA NCEI GHCN v4](https://www.ncei.noaa.gov/pub/data/ghcn/v4/) |
| Historical Moscow monthly temperature | 1936–1945 records supplementing GHCN-Monthly | [Pogoda i Klimat station 27612](https://www.pogodaiklimat.ru/history/27612.htm) |
| Historical Moscow daily temperature | December 1941 and January 1942 | [December 1941](https://www.pogodaiklimat.ru/monitor.php?id=27612&month=12&year=1941); [January 1942](https://www.pogodaiklimat.ru/monitor.php?id=27612&month=1&year=1942) |
| HadSLP2 sea-level pressure | HadSLP2 | [Met Office Hadley Centre HadSLP2](https://www.metoffice.gov.uk/hadobs/hadslp2/) |
| HadISST sea-ice concentration | v1.1 | [Met Office Hadley Centre HadISST](https://www.metoffice.gov.uk/hadobs/hadisst/) |
| ERSST sea-surface temperature | v5 | [NOAA PSL ERSST v5](https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html) |
| Pacific Decadal Oscillation index | Monthly index; product version not specified | [NOAA PSL monthly PDO](https://psl.noaa.gov/data/timeseries/month/PDO/) |
| Niño-3.4 anomalies | Derived from HadISST v1.1 sea-surface temperature; not a separate archived input in the manuscript | [Met Office Hadley Centre HadISST](https://www.metoffice.gov.uk/hadobs/hadisst/) |

## Reconstruction, model, and inputs without a stated download location

| Input or software | What the manuscript states | Access detail still needed |
| --- | --- | --- |
| IAPICE sea-ice concentration reconstruction | v1.1; monthly 1° × 1°, 1901–2019; methods cite Semenov et al. (2024) and Li et al. (2026) | The main manuscript gives no public dataset URL, DOI, or accession for the exact reconstruction files used here. |
| CAM-IAPICE and CAM-HadISST experiment outputs | CAM4; 20 matched members per experiment, 1901–2017, 0.95° × 1.25°, 26 levels | The main manuscript describes the experiments but gives no repository/DOI or access procedure for the NetCDF outputs read by these scripts. |
| CAM4 model software | Distributed within CCSM4 | [CESM CCSM4 model page](https://www.cesm.ucar.edu/models/ccsm4.0); this is a model-software page, not an archive of this study's model output. |
| Historical European-boundary shapefile | Used for Fig. S1; the script refers to a 1942-01-31 boundary | The main manuscript and its availability statement do not specify the shapefile's source, download link, or redistribution rights. |
| External forcing | CMIP5 historical protocol for 1901–2005 and RCP8.5 for 2006–2017 | The manuscript supplies the protocol names but no specific forcing-file archive or identifiers. |

Input data are **not included** in this code archive, and the MIT License applies only to the code and accompanying documentation, not to third-party datasets. For any dataset that cannot be redistributed, the final manuscript and this repository should state the controlling source and an access route rather than imply that the data are bundled here.

