# COVID-19 Rapid Triage Risk Predictor Report

## Project overview
This repository implements a radiomics-based COVID-19 triage and outcome prediction workflow. It is designed to estimate patient risk for three outcomes:

- ICU admission
- Mechanical ventilation
- In-hospital death

The system combines a radiomics feature extraction pipeline with pretrained machine learning models. It supports two prediction modes:

1. radiomics-only models
2. radiomics + clinical/laboratory feature models

## Main purpose
The project aims to support rapid clinical risk stratification by using imaging-derived features and auxiliary patient data to identify high-risk COVID-19 patients early in care.

## Repository structure
- `COVID-19_prediction.py` — main prediction entry point that loads data, standardizes features, runs the trained models, and optionally prints SHAP explanations.
- `proc_radiomic_feature.py` — extracts radiomics features from DICOM and segmentation inputs.
- `models/` — pretrained classification models and preprocessing artifacts.
- `example_lab_input.csv` — sample clinical/lab input file.
- `example_median.csv` — median values used for lab-data imputation.
- `units.txt` — unit definitions for input features.
- `radiomics_patch/` — compatibility patches used to address PyRadiomics issues.
- `README.md` — setup and usage instructions.

## Data pipeline
The workflow has two major stages:

### 1. Feature extraction
The project expects input organized as:

- a DICOM image series per patient/study/series
- a corresponding segmentation mask set stored as PNG files

The feature extraction step converts lesion information into patient-level radiomics summaries and produces a final CSV such as `final_merge_feature.csv`.

### 2. Outcome prediction
The prediction pipeline reads the extracted radiomics CSV and optionally a laboratory/clinical CSV. It performs standardization, aligns feature columns to the trained models, and outputs model predictions for the three predefined outcomes.

## Model logic
The prediction script loads saved classifiers from the `models/` directory and validates them before inference. It also includes compatibility logic for older serialized LightGBM and scikit-learn objects, which is useful when newer library versions are used.

The results are reported in CLI output, and `--explain` can provide SHAP-based feature-attribution summaries that show the strongest contributors for each prediction.

## Input formats
### Radiomics-only mode
Command pattern:

```bash
python COVID-19_prediction.py --radiomics_data <save_root/final_merge_feature.csv>
```

### Combined radiomics + lab mode
Command pattern:

```bash
python COVID-19_prediction.py --radiomics_data <save_root/final_merge_feature.csv> --lab_data <lab_input.csv>
```

## Important considerations
- The project relies on carefully standardized features and model artifacts saved in the repository.
- SHAP explanations are intended to support interpretability, not clinical validation.
- This is a research-oriented pipeline and should be used with domain expertise and appropriate clinical review.

## Summary
This repository provides a complete end-to-end framework for COVID-19 risk prediction using imaging-derived radiomic features and optional laboratory data. It is organized around a feature extraction pipeline, pretrained models, and a command-line prediction workflow for early triage support.
