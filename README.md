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

### Download Instructions
- **Source**: [Lending Club Dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Required files**:
  - `accepted_2007_to_2018Q4.csv` (~1.6 GB)

### Data Setup
1. Download the files from Kaggle
2. Create a `data/` folder in the project root if it doesn't exist
3. Place both CSV files inside `data/`

```bash
# Expected structure:
credit-risk-modeling/
├── data/
│   ├── accepted_2007_to_2018Q4.csv
├── 01_eda.ipynb
├── src/
├── .gitignore
├── README.md
└── requirements.txt

