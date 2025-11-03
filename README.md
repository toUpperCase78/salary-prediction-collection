# Collection of Salary Predictions

## Overview

This repo aims for which studies I've carried out to understand and gain insights about how salaries can be predicted in the best way, against several attributes among datasets from various GitHub repos; also to figure out which machine learning model is the best to deliver reliable salary predictions with regards to measurements.

Generally, these operations are included below, but the methods applied may differ from each other:

* Exploratory Data Analysis (EDA)
* Data Visualizations
* Feature Engineering / Preprocessing
* Predictions with Machine Learning Regression models
* Comparisons of model results with measurements (MSE, REMSE, MAE, R2, Cross Validation)
* Explainable AI (XAI)

## 01. PlayingNumbers / ds_salary_proj

**LINK:** https://github.com/PlayingNumbers/ds_salary_proj

### FILES

* [01_EDA(1)__salary_data_cleaned.ipynb](01_EDA(1)__salary_data_cleaned.ipynb) (Exploratory Data Analysis on salary_data_cleaned.csv file) (599 KB)
* [01_EDA(2)__glassdoor_job.ipynb](01_EDA(2)__glassdoor_job.ipynb) (Exploratory Data Analysis on glassdoor_jobs.csv file) (35 KB)
* [01_EDA(3)__salary_data_cleaned.ipynb](01_EDA(3)__salary_data_cleaned.ipynb) (Additional Exploratory Data Analysis on salary_data_cleaned.csv file) (57 KB)
* [01_Regression__salary_data_cleaned.ipynb](01_Regression__salary_data_cleaned.ipynb) (Predictions with ML Regression models on salary_data_cleaned.csv file) (511 KB)
* [01_eda_data.csv](01_eda_data.csv) (Initial salary dataset) (742 rows) (3075 KB)
* [01_glassdoor_data.csv](01_glassdoor_data.csv) (Salary dataset for Glassdoor jobs) (956 rows) (3784 KB)
* [01_salary_data_cleaned.csv](01_salary_data_cleaned.csv) (Salary dataset as cleaned) (742 rows) (3060 KB)

### FEATURES: salary_data_cleaned.csv

| Feature | Non-Null Count | Data Type | Used for ML | Preprocessing Notes | 
|---------|----------------|-----------|-------------|---------------------|
| Job Title | 742 | string | No | |
| Salary Estimate | 742 | string | No | |
| Job Description | 742 | string | No | |
| Rating | 742 | float64 | Yes | '-1' values were filled with average |
| Company Name | 742 | string | No | |
| Location | 742 | string | No | |
| Headquarters | 742 | string | No | |
| Size | 742 | string | No | |
| Founded | 742 |int64 | No | |
| Type of ownership | 742 | string | No | |
| Industry | 742 | string | No | |
| Sector | 742 | string | No | |
| Revenue | 742 | string | No | |
| Competitors | 742 | string | No | |
| hourly | 742 | int64 | Yes | |
| employer_provided | 742 | int64 | Yes | |
| min_salary | 742 | int64 | No | Converted hourly values |
| max_salary | 742 | int64 | No | Converted hourly values |
| avg_salary | 742 | float64 | Yes | The dependent feature for the ML models |
| company_txt | 742 | string | No | Applied data cleaning |
| job_state | 742 | string | Yes | Applied data cleaning |
| same_state | 742 | int64 | Yes | |
| age | 742 | int64 | Yes | '-1' values were filled with average |
| python_yn | 742 | int64 | Yes | |
| R_yn | 742 | int64 | Yes | |
| spark | 742 | int64 | Yes | |
| aws | 742 | int64 | Yes | |
| excel | 742 | int64 | Yes | |

### ADDITIONAL FEATURES: salary_data_cleaned.csv

| Feature | Non-Null Count | Data Type | Used for ML | Preprocessing Notes | 
|---------|----------------|-----------|-------------|---------------------|
| job_simp | 742 | string | Yes | Obtained from 'Job Title'; created dummy variables |
| seniority | 742 | string | Yes | Obtained from 'Job Title'; created dummy variables |
| desc_len | 742 | int64 | Yes | Obtained from 'Job Description'; created dummy variables |
| num_comp | 742 | int64 | Yes | Obtained from 'Competitors' |

### RESULTS OF ML MODELS: salary_data_cleaned.csv (80% Train, 20% Test)

| Model | MSE | RMSE | MAE | R2 Score | 5-Fold CV Train | 5-Fold CV Test |
|-------|-----|------|-----|----------|-----------------|----------------|
| Linear Reg | 679,94 | 26,08 | 20,77 | 0,5183 | 0,627 | 0,5467 |
| Lasso | 828,48 | 28,78 | 23,41 | 0,4131 | 0,489 | 0,4334 |
| Ridge | 1420,27 | 37,69 | 30,1 | -0,0062 | 0,2515 | 0,2467 |
| Polynomial Lasso | 762,73 | 27,62 | 20,98 | 0,4597 | 0,489 | 0,4334 |
| Polynomial Ridge | 971,67 | 31,17 | 20,27 | 0,3116 | 0,6252 | 0,5511 |
| Decision Tree | 1236,39 | 35,12 | 27,3 | 0,1241 | 0,1974 | 0,2366 |
| Random Forest | 984,39 | 31,37 | 25,35 | 0,3026 | 0,4105 | 0,3394 |
| KNN Reg | 1245,87 | 35,3 | 28,5 | 0,1174 | 0,2521 | 0,0366 |
| SVM Reg | 1467,13 | 38,3 | 30,36 | -0,0394 | 0,005 | -0,0185 |
| MLPR | 1033,53 | 32,15 | 25,96 | 0,2678 | N/A | N/A |
| Gradient Boosting | 564,06 | 23,75 | 18,27 | 0,6004 | 0,8089 | 0,6129 |

All best values were achieved by **Gradient Boosting**.

## 02. Trouble404 / kaggle-Job-Salary-Prediction

**LINK:** https://github.com/Trouble404/kaggle-Job-Salary-Prediction

### FILES

* [02_EDA(1)__Train_rev1.ipynb](02_EDA(1)__Train_rev1.ipynb) (Exploratory Data Analysis on Train_rev1.csv file) (177 KB)
* [02_EDA(2)__Test_rev1.ipynb](02_EDA(2)__Test_rev1.ipynb) (Exploratory Data Analysis on Test_rev1.csv file) (24 KB)
* [02_Location_Tree.csvc](02_Location_Tree.csv) (The single-column dataset with location tree values) (31762 rows) (1665 KB)
* [02_Regression__Train_rev1.ipynb](02_Regression__Train_rev1.ipynb) (Predictions with ML Regression models on Train_rev1.csv file) (473 KB)
* [02_mean_benchmark.csv](02_mean_benchmark.csv) (The two-column dataset with id and normalized salary values) (40663 rows) (1152 KB)
* [02_random_forest_benchmark_test_rev1.csv](02_random_forest_benchmark_test_rev1.csv) (Another two-column dataset with id and normalized salary values) (122463 rows) (3455 KB)
* [02_test.csv](02_test.csv) (Another single-column dataset with id values) (122463 rows) (1196 KB)

**Train_rev1.csv** and **Test_rev1.csv** files are not included due to being too large (420 MB & 205 MB).

### FEATURES: Train_rev1.csv

| Feature | Non-Null Count | Data Type | Used for ML | Preprocessing Notes | 
|---------|----------------|-----------|-------------|---------------------|
| Id | 244471 | int64 | No | |
| Title | 244470 | string | No | |
| FullDescription | 244471 | string | No | |
| LocationRaw | 244471 | string | No | |
| LocationNormalized | 244471 | string | No | |
| ContractType | 65387 | string | No | | 
| ContractTime | 180628 | string | No | |
| Company | 212144 | string | No | |
| Category | 244471 | string | No | |
| SalaryRaw | 244471 | string | No | |
| SalaryNormalized | 244471 | string | Yes | The dependent feature for the ML models|
| SourceName | 244471 | string | No | |

### ADDITIONAL FEATURES: Train_rev1.csv

| Feature | Non-Null Count | Data Type | Used for ML | Preprocessing Notes | 
|---------|----------------|-----------|-------------|---------------------|
| TitleSimp | 244471 | string | Yes | Obtained from 'Title'; created dummy variables |
| CategorySimp | 244471 | string | Yes | Obtained from 'Category'; created dummy variables |
| ContractTypeEdit | 244471 | string | Yes | Obtained from 'ContractType'; created dummy variables |
| ContractTimeEdit | 244471 | string | Yes | Obtained from 'ContractTime'; created dummy variables |
| DescLength | 244471 | string | Yes | Obtained from 'FullDescription' |

### RESULTS OF ML MODELS: Train_rev1.csv (80% Train, 20% Test)

| Model | MSE | RMSE | MAE | R2 Score | 5-Fold CV Train | 5-Fold CV Test |
|-------|-----|------|-----|----------|-----------------|----------------|
| Linear Reg | 242245662,7 | 15564,24 | 11526,77 | 0,1817 | 0,1823 | 0,1761 |
| Lasso | 242239920,0 | 15564,06 | 11526,63 | 0,1817 | 0,1823 | 0,1761 |
| Ridge | 263636399,3 | 16236,88 | 12204,92 | 0,1094 | 0,1073 | 0,1027 |
| Polynomial Lasso | 221676092,9 | 14888,79 | 10880,99 | 0,2512 | 0,1823 | 0,1761 |
| Polynomial Ridge | 221638542,2 | 14887,53 | 10874,37 | 0,2513 | 0,1823 | 0,1761 |
| Random Forest | 266177079,3 | 16314,93 | 12238,75 | 0,1009 | 0,0970 | 0,0924 |
| MLPR | 242795656,5 | 15581,9 | 11627,24 | 0,1798 | N/A | N/A |

All best values were achieved by **Polynomial Ridge**.

## 03. rajpurohitpooja / Salary_Prediction_Portfolio

LINK: https://github.com/rajpurohitpooja/Salary_Prediction_Portfolio

## 04. disha2disha / Employee-Salary-Prediction

LINK: https://github.com/disha2sinha/Employee-Salary-Prediction

## 05. DheerajKumar97 / Employee-Salary-Prediction----python--ML

LINK: https://github.com/DheerajKumar97/Employee-Salary-Prediction----python--ML

## 06. jhanvi1831 / Software-Engineer-Salary-Prediction

LINK: https://github.com/jhanvi831/Software-Engineer-Salary-Prediction

## 07. bonniema / salary-predictor

LINK: https://github.com/bonniema/salary-predictor/tree/master

## 08. Pranjali1049 / Salary_Prediction

LINK: https://github.com/Pranjali1049/Salary_Prediction/tree/main
