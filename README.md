# Credit Risk Modeling: Probability of Default (PD)

A comprehensive credit risk analysis and PD modeling project for retail loan portfolios, following Basel III/IV and IFRS 9 frameworks.

## Project Overview

This project develops a Probability of Default (PD) model for retail loans using Lending Club data (2007-2018). The analysis includes:

- Exploratory Data Analysis (EDA) following regulatory frameworks
- Feature engineering and selection (pending)
- PD model development and validation (pending)
- Low Default Portfolio (LDP) identification (pending)

## Data Requirements

This project requires the following dataset from **Lending Club**:
- **Source**: [Lending Club Dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Required files**:
  - `accepted_2007_to_2018Q4.csv` (~1.6 GB)

### Data storage
- Data is stored in [Dropbox](https://www.dropbox.com/scl/fi/gxskg98izg6fmulp22n3r/accepted_2007_to_2018Q4.csv?rlkey=1jl9t026rlg5o009wbf9rbog0&st=eas942l3&dl=1)
- Preprocessed data is stored in [GitHub] (https://github.com/martinezb-jose-c/credit-risk-modeling/releases/tag/v0.2-alpha)

```bash
# Expected structure:
credit-risk-modeling/
├── 01_eda.ipynb
├── 02_feature_engineering.ipynb
├── 03_modeling.ipynb
├── models/
    ├──ohe_encoder.pkl
    ├──scaler.pkl
    ├──features.pkl
    ├──woe_encoder.pkl
├── src/
    ├──feature_engineering.py
├── .gitignore
├── README.md
└── requirements.txt