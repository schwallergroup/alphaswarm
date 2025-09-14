## Benchmark datasets

This directory contains the datasets that are used in the manuscript accompanying this code repository. The datasets include two Pd-catalysed Buchwald-Hartwig reactions and one Ni-catalysed Suzuki-Miyaura reaction.

### Datasets

#### ⚗️ 1. Buchwald-Hartwig benchmark dataset

This dataset was generated from **384** experimentally collected reactions from HTE experiments conducted in the work of [Sin et al.](https://doi.org/10.1038/s41467-025-61803-0).

- **Training/experimental data**: [buchwald_train_data.csv](experimental_data/buchwald_train_data.csv)
- **Virtual benchmark data**: [buchwald_virtual_benchmark.csv](buchwald_virtual_benchmark.csv)

#### ⚗️ 2. Ni-catalyzed Suzuki coupling benchmark dataset

This dataset was generated from **576** experimentally collected reactions from HTE experiments conducted in the work of [Sin et al.](https://doi.org/10.1038/s41467-025-61803-0).

- **Training/experimental data**: [ni_suzuki_train_data.csv](experimental_data/ni_suzuki_train_data.csv)
- **Virtual benchmark data**: [ni_suzuki_virtual_benchmark.csv](ni_suzuki_virtual_benchmark.csv)

#### ⚗️ 3. Pd-catalysed Buchwald-Hartwig Sulfonamide coupling reaction

This dataset was generated from **585** experimentally collected reactions from HTE experiments described in the manuscript associated with this code repository.

- **Training/experimental data**: [sulfonamide_train_data.csv](experimental_data/sulfonamide_train_data.csv)
- **Virtual benchmark data**: [sulfonamide_virtual_benchmark.csv](sulfonamide_virtual_benchmark.csv)


> [!NOTE]
> All datasets contain normalised features for the ML models.
