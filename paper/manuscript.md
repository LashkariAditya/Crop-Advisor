# CropAdvisor: A Reproducible Machine-Learning Prototype for Crop and Land-Cover Decision Support in India

**Author(s):** [Add author names]  
**Affiliation(s):** [Add institution, department, city, country]  
**Corresponding author:** [Add email]  
**Target venue:** [Select conference and apply its current template]

## Abstract

We present CropAdvisor, a local-first decision-support prototype for Indian crop planning that combines a web application, sourced cultivation guidance, transparent farm-economics calculations, live environmental-data adapters, and neural models implemented from scratch with NumPy or fine-tuned from ImageNet weights. We document data provenance and evaluate task-specific models on held-out partitions rather than treating application functionality as evidence of agronomic validity. The separate soil-core fraction experiment uses borehole-grouped splits and achieved 69.46% test accuracy (0.555 macro-F1) on four frequent geotechnical labels; it is not an agricultural soil-type model. A crop-suitability multilayer perceptron (MLP) achieved 98.18% accuracy and 0.982 macro-F1 on a stratified test partition of the 2,200-record Crop Recommendation benchmark. An ImageNet-pretrained ResNet-18 achieved 88.46% accuracy (0.823 macro-F1) on a 182-image soil-photo test split; an exploratory flip-averaging evaluation reached 89.01% accuracy (0.854 macro-F1) on the same previously scored split. A follow-up training experiment adding 3,909 CyAUG generated images did not improve validation performance (83.15% accuracy, 0.768 macro-F1), so the original-only checkpoint remains selected. The shallow network achieved 71.31% accuracy (0.703 macro-F1) on the official 4,050-image EuroSAT test split. For annual India-level yield and producer-price-index prediction, chronological evaluation showed that the MLPs were less accurate than last-observation persistence: test MAE was 1,871 versus 529 kg/ha for yield and 16.78 versus 9.35 index points for price. The system therefore exposes the selected persistence estimates and clearly labels the limitations of benchmark, imbalanced, and aggregate data. The results support the feasibility of a reproducible, transparent software prototype, while indicating that its current models are not validated for farm-level recommendations.

**Keywords:** agricultural decision support, crop suitability, NumPy, model evaluation, India, reproducibility

## 1. Introduction

Smallholder crop decisions depend on local soils, climate, crop calendars, input prices, water access, and market conditions. Many demonstration systems combine predictors and advice without exposing the source, unit, baseline, or limits of their data. This work develops CropAdvisor as an inspectable software prototype: it makes predictions only from local trained artifacts, distinguishes satellite land cover from soil-photo classes, tracks source and license information, exposes input assumptions for economic calculations, and reports simple predictive baselines.

The contribution is an end-to-end, reproducible implementation and a candid evaluation of its present data. We do not claim a field-tested recommendation system. In particular, benchmark classification accuracy cannot establish that recommended crops will be profitable or suitable for an individual farm.

## 2. System design

The application consists of a React/Vite client and a Python FastAPI service. The service accepts validated field images, fetches weather and soil context through external adapters when available, and uses cached local model artifacts for inference. It provides sourced rice and wheat cultivation plans, user-editable economic calculations in INR for acres or hectares, pack-quantity optimization, and an explicitly mock product catalog. Mandi prices are not fabricated: the live adapter requires a configured data.gov.in resource and API key. Unavailable external services are surfaced as unavailable.

The crop model is used only when all seven numeric benchmark inputs are provided (N, P, K, temperature, humidity, pH, and rainfall). The image pathway distinguishes close-up field/soil photographs from satellite scenes. Soil-photo output is a dataset class label; EuroSAT output is a land-cover class. Neither is interpreted as a laboratory soil test. National yield and producer-price time-series endpoints are marked experimental and return units, a neural estimate, the validation-selected predictor, and a warning.

## 3. Data and methods

### 3.1 Data provenance

The crop-suitability experiment uses the Apache 2.0 Crop Recommendation Dataset distributed through Kaggle: 2,200 rows, seven numeric variables, 22 crop labels, and 100 rows per class. Its dataset description reports that records were built by augmenting rainfall, climate, and fertilizer datasets. We use it as a benchmark, not as independent farm observations.

Soil-image classification uses the original photographs in the Soil Classification Dataset repository, pinned at commit `c3108e9ea8277a2762b0aad587e90cc9722e63d4`. The 1,189 original images cover seven imbalanced folder labels (52–284 images/class). The repository specifies CC BY-NC 4.0; underlying photograph provenance may vary. In a separate experiment, we additionally used 3,909 non-exact files from the repository's CyAUG CycleGAN-generated collection, appending them to the training partition only; validation and test retained original photos. Exact duplicates were removed, but generated-source lineage is incomplete and a perceptual-hash audit cannot establish independence from the original split. This experiment is therefore exploratory. The selected production-local checkpoint continues to use original photos only. Use is restricted to non-commercial research/education and requires source-attribution review.

Land-cover classification uses the EuroSAT RGB dataset of 27,000 Sentinel-2 image patches in ten classes and its published 3,000/4,050/4,050 train/validation/test partition. EuroSAT labels are not soil labels.

National time-series experiments are derived from FAOSTAT India crop production/yield and producer-price records. The yield table contains 854 crop-year observations over 14 crop series (1964–2024), expressed in kg/ha. The producer-price table contains 280 observations across eight item series (1991–2025); its target is the producer price index (2014–2016=100), not INR/kg or a mandi quotation. Rows are ordered by target year and lag features use prior observations only. Sources, raw-file checksums, and derived schemas are documented in `DATA_SOURCES.md` and the manifest.

### 3.2 Models and preprocessing

All learned models are implemented with NumPy and trained from random initialization using fixed seed 7. The crop classifier is a fully connected MLP with one ReLU hidden layer, Adam optimization, class weighting, and softmax probabilities. The two regression MLPs have one 32-unit ReLU hidden layer and predict standardized targets using mean-squared error. Numeric means and standard deviations are fit on the training partition only; constant features are protected from division by zero. Models select a checkpoint by validation loss with patience 15 and restore the best validation checkpoint.

The initial image baselines use the project’s shallow convolutional CNN with 32×32 RGB input. The improved soil-photo experiment fine-tunes an ImageNet1K_V1-pretrained torchvision ResNet-18 at 224×224 resolution. Its deterministic stratified 70/15/15 file split (seed 7) contains 829 training, 178 validation, and 182 test images. Training applies random resized crops, flips, rotation, and mild color jitter to training examples only; inverse-frequency class weights address imbalance. The classifier head is trained first, followed by the final residual block. Checkpoint selection uses validation macro-F1 with accuracy as a tie-breaker. The best checkpoint is epoch 14 of 18. A later validation-selected flip-averaging comparison reuses the test split previously evaluated without averaging; that second score is therefore exploratory. EuroSAT remains on the NumPy CNN and uses its official 3,000/4,050/4,050 split. No test data are used for checkpoint selection.

### 3.3 Evaluation protocol

Crop suitability uses a seeded, stratified 70/15/15 train/validation/test split. The soil-image split is stratified 70/15/15. EuroSAT is evaluated on the official test partition. Annual regressions split by whole target year into chronological training, validation, and test partitions. The final test period begins in 2015 for yield and 2021 for the producer-price index. Regression models are compared against a persistence baseline that repeats the most recent observation for the same crop/item. Classification reports accuracy and macro-F1; regression reports MAE, RMSE, and MAPE. Test performance is descriptive of these datasets and split protocols only.

## 4. Results

| Task | Test partition | Neural model | Baseline |
|---|---:|---:|---:|
| Crop recommendation | 330 records, 22 classes | Accuracy 98.18%; macro-F1 0.982 | Not applicable |
| Soil-photo class, shallow CNN baseline | 182 images, 7 imbalanced classes | Accuracy 76.92%; macro-F1 0.728 | Not applicable |
| Geotechnical soil-core main fraction | 609 images, 4 frequent labels; 7 boreholes held out | Accuracy 69.46%; macro-F1 0.555 | Borehole-grouped split |
| Soil-photo class, ResNet-18 transfer learning | 182 images, 7 imbalanced classes | Accuracy 88.46%; macro-F1 0.823 | Validation accuracy 88.20%; macro-F1 0.844 |
| Soil-photo class, flip test-time averaging (exploratory) | Same 182-image test partition, reused | Accuracy 89.01%; macro-F1 0.854 | Validation accuracy 88.20%; macro-F1 0.850 |
| Soil-photo class, CyAUG-generated training supplement (not selected) | 182 original test images; test split reused, exploratory | Accuracy 80.77%; macro-F1 0.760 | Validation accuracy 83.15%; macro-F1 0.768; 3,909 generated files appended to training only |
| EuroSAT land cover | 4,050 images, 10 classes | Accuracy 71.31%; macro-F1 0.703 | Not applicable |
| India annual yield | 140 crop-years, from 2015 | MAE 1,871 kg/ha; RMSE 2,520 kg/ha; MAPE 67.49% | Persistence MAE 529 kg/ha; RMSE 1,258 kg/ha; MAPE 6.52% |
| India producer-price index | 40 item-years, from 2021 | MAE 16.78 points; RMSE 19.64; MAPE 11.29% | Persistence MAE 9.35 points; RMSE 10.98; MAPE 6.59% |

The crop benchmark result is high under a random split, but the dataset’s augmented origin and lack of farm-level identifiers prevent a claim of geographic or temporal generalization. Increasing soil augmentation from one to three synthetic views improved shallow-CNN test accuracy from 72.53% to 76.92%. ImageNet transfer learning raised the same split's test accuracy to 88.46% and macro-F1 to 0.823, but remained below 90%; validation accuracy was 88.20%. A later validation-selected flip-averaging check yielded 89.01% test accuracy and 0.854 macro-F1, still below 90%. Because this reused a test set already scored for the non-averaged model, that comparison is exploratory and requires a fresh external test set. Adding 3,909 CycleGAN-generated CyAUG images to training did not improve generalization: validation accuracy/macro-F1 were 83.15%/0.768 versus 88.20%/0.844 without those images, and exploratory test accuracy was 80.77%. The current application therefore retains the original-only checkpoint. These results do not establish that more synthetic images improve field-photo classification; model quality requires more independently sourced, labeled real photos and a fresh, source-separated evaluation. Alluvial Soil test F1 remains weak (0.286 without averaging; 8 test images), and the overall accuracy does not resolve class imbalance, file-level split leakage risks, or provenance limits. Two augmented views gave a modest validation macro-F1 increase for EuroSAT from 0.686 to 0.693 and improved held-out test accuracy from 70.86% to 71.31%. These results remain specific to the official EuroSAT split and the small CPU-trained CNN; they are not intended to compete with large pretrained remote-sensing models. For both aggregate time-series tasks, persistence is the stronger validation-selected predictor and also has lower test error. The application reports this and does not present the MLP as superior.

## 5. Discussion

The principal result is a runnable and auditable decision-support prototype, not a validated agronomic model. The experiment illustrates why baselines and unit labels matter: the existence of a trained regressor does not justify using it, and an annual producer-price index is not a local selling price. Similarly, near-ceiling benchmark crop accuracy may coexist with poor external validity.

The soil-photo set is small, imbalanced, and subject to non-commercial restrictions and variable source-image provenance. The separate MIT-licensed soil-core dataset contains repeated views and measurements from individual boreholes; official image splits crossed boreholes, so we regrouped by borehole. The resulting grouped test accuracy is only 69.46%, and rare fractions were excluded from this four-class experiment. This geotechnical task is not comparable to agricultural field-soil photos. The image split is random at file level, so related acquisition contexts may cross partitions. Both image and crop results should be followed by grouped or source-level splits, external data, uncertainty calibration, and prospective evaluation. FAOSTAT aggregates are not suitable to estimate plot-level yield or local market prices. Application-level economics uses explicit assumptions and mock retail data must not inform purchasing decisions.

## 6. Limitations and future work

No farmer-held-out or district-held-out test, field trial, calibrated uncertainty assessment, agronomist-led usability study, or prospective economic outcome evaluation has been performed. The crop dataset’s units and measurement context require independent verification before field use. The soil-photo license limits commercial deployment and original source photos require careful attribution. External weather, soil, and market APIs can change or become unavailable. The current plan library covers rice and wheat only and is not a substitute for state-specific extension advice.

Future evaluation should prioritize locally collected, permissioned data; group splits by source, site, farm, and time; current commodity-level mandi data with verified units; comparisons against robust statistical baselines; calibration and subgroup analysis; and evaluation with agricultural experts and farmers. A commercial release would require replacement or separate licensing of the non-commercial soil-image data.

## 7. Conclusion

CropAdvisor demonstrates an end-to-end agricultural planning application with traceable datasets, from-scratch NumPy models, transparent unit-aware calculations, and held-out evaluation. Its crop classifier performs well on one small benchmark and its image models provide limited visual classification; national annual yield and price neural models do not outperform persistence. The software therefore makes model scope and data limitations visible and should be treated as a research prototype pending independent field validation.

## References

[1] S. Helber, B. Bischke, A. Dengel, and D. Borth, “EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification,” *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 12, no. 7, pp. 2217–2226, 2019, doi: [10.1109/JSTARS.2019.2918242](https://doi.org/10.1109/JSTARS.2019.2918242).

[2] A. Ingle, “Crop Recommendation Dataset,” Kaggle, 2020. [https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset). Accessed 8 Nov. 2026. Dataset metadata reports Apache 2.0; verify current terms before redistribution.

[3] Phantom-fs, “Soil Classification Dataset,” GitHub repository, commit `c3108e9ea8277a2762b0aad587e90cc9722e63d4`, 2025. [https://github.com/Phantom-fs/Soil-Classification-Dataset](https://github.com/Phantom-fs/Soil-Classification-Dataset). Accessed 8 Nov. 2026. CC BY-NC 4.0 repository license; underlying source-photo provenance may vary.

[4] F. Sheth, P. Mathur, A. K. Gupta, and S. Chaurasia, “An advanced artificial intelligence framework integrating ensembled convolutional neural networks and Vision Transformers for precise soil classification with adaptive fuzzy logic-based crop recommendations,” *Engineering Applications of Artificial Intelligence*, vol. 158, part A, art. 111425, 2025, doi: [10.1016/j.engappai.2025.111425](https://doi.org/10.1016/j.engappai.2025.111425).

[5] FAO, “FAOSTAT Statistical Database,” Food and Agriculture Organization of the United Nations. [https://www.fao.org/faostat/](https://www.fao.org/faostat/). Accessed 8 Nov. 2026. FAOSTAT records are subject to the [FAO Statistical Database Terms of Use](https://www.fao.org/contact-us/terms/db-terms-of-use/en), including attribution and no-endorsement requirements.

[6] A.-N. Granitzer, J. Beck, J. Leo, and F. Tschuchnigg, “Explainable Insight into the Vision-Based Classification of Soil Core Samples from Close-Range Images,” 3FOMLIG, 2025. Dataset: [Hugging Face grano1/core_sample_image_data](https://huggingface.co/datasets/grano1/core_sample_image_data), MIT license.

[7] CropAdvisor source, data manifest, model cards, training scripts, and API implementation, local project release accompanying this manuscript. [Add a public repository URL and version/commit before submission].

## Author checklist before submission

Replace placeholders; confirm conference scope and template; rerun all commands on the submitted revision; regenerate metrics from current model cards; add repository archive/version and hardware/software versions; verify dataset terms and citations; format references to the target venue; and ensure figures/tables follow its page limits. Do not describe these results as field validated.
