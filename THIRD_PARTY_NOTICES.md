# Third-party data notices

The public-preview image includes model artifacts trained from the following data sources. This notice covers data/model provenance; it does not grant rights beyond the source terms.

## Crop Recommendation Dataset

- Source: [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- License reported by Kaggle dataset metadata: Apache License 2.0; verify current source metadata before redistribution.
- The dataset description says records were created by augmenting rainfall, climate, and fertilizer data. The model's random-split score is benchmark performance, not external farm validation.
- The trained MLP is provided only as an educational demonstration and must not be treated as a field-tested crop prescription.

## EuroSAT and Sentinel-2

- Source and dataset references: [EuroSAT project](https://github.com/phelber/EuroSAT), [Zenodo RGB archive](https://zenodo.org/records/7711810), and Helber et al. (2019), DOI [10.1109/JSTARS.2019.2918242](https://doi.org/10.1109/JSTARS.2019.2918242).
- The EuroSAT project lists the dataset under MIT; the included [MIT license text](licenses/EUROSAT-MIT.txt) is from that project.
- Source imagery is Copernicus Sentinel-2 data. Follow the [Copernicus Sentinel Data Legal Notice](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice).
- The included classifier predicts the EuroSAT land-cover labels; it does not infer soil type or local land suitability.

## Soil core image dataset

- Source: [Hugging Face soil core sample image data](https://huggingface.co/datasets/grano1/core_sample_image_data), 3,604 labeled soil-core images.
- The dataset card reports MIT. Cite Granitzer, Beck, Leo, and Tschuchnigg, “Explainable Insight into the Vision-Based Classification of Soil Core Samples from Close-Range Images,” 3FOMLIG 2025.
- The separate model predicts geotechnical main fractions and does not classify agricultural soil types, nutrients, pH, or moisture.

## Excluded datasets

The soil-image dataset is CC BY-NC 4.0 with variable underlying photo provenance and is excluded from the public-preview image. FAOSTAT-derived annual forecasts are also excluded from the public-preview image and endpoints pending a separate license/use review and because their neural models lose to persistence. See `DATA_SOURCES.md` and `PUBLIC_RELEASE.md` for details.
