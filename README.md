# Collection of Salary Predictions

## Overview

This repo aims for which studies I've carried out to understand and gain insights about how salaries can be predicted in the best way, against several attributes among datasets from various GitHub repos; also to figure out which machine learning model is the best to deliver reliable salary predictions with regards to metrics.

All these studies have been presented in **Jupyter Notebook** files, featuring info about the datasets, preprocessing steps, feature selections, machine learning models used for predictions and results of all metrics in bar graphs.

Generally, these operations are included below, but the methods applied may differ from each other:

* Exploratory Data Analysis (EDA)
* Data Visualizations
* Feature Engineering / Preprocessing
* Predictions with Machine Learning Regression models
* Comparisons of model results with metrics (MSE, RMSE, MAE, R2, Cross Validation)
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

### ADDITIONAL FEATURES: salary_data_cleaned.csv (742 rows)

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

### FEATURES: Train_rev1.csv (244471 rows)

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

**LINK:** https://github.com/rajpurohitpooja/Salary_Prediction_Portfolio

### FILES

* [03_EDA(1)__train_features_salaries.ipynb](03_EDA(1)__train_features_salaries.ipynb) (Exploratory Data Analysis on train_features.csv & train_salaries.csv merged together) (715 KB)
* [03_EDA(2)__test_features.ipynb](03_EDA(2)__test_features.ipynb) (Exploratory Data Analysis on test_features.csv file) (14 KB)
* [03_Regression(T1)__train_features_salaries.ipynb](03_Regression(T1)__train_features_salaries.ipynb) (Predictions with ML Regression models on train_features.csv & train_salaries.csv merged together by using Type-1 feature set) (490 KB)
* [03_Regression(T2)__train_features_salaries.ipynb](03_Regression(T2)__train_features_salaries.ipynb) (Predictions with ML Regression models on train_features.csv & train_salaries.csv merged together by using Type-2 feature set) (476 KB)
* [03_Regression(T3)__train_features_salaries.ipynb](03_Regression(T3)__train_features_salaries.ipynb) (Predictions with ML Regression models on train_features.csv & train_salaries.csv merged together by using Type-3 feature set) (467 KB)
* [04_Regression(T4)__train_features_salaries.ipynb](03_Regression(T4)__train_features_salaries.ipynb) (Predictions with ML Regression models on train_features.csv & train_salaries.csv merged together by using Type-4 feature set) (469 KB)
* [03_XAI_Gradient_Boosting_Classification(T1)__train_features_salaries.ipynb](03_XAI_Gradient_Boosting_Classification(T1)__train_features_salaries.ipynb) (Explaining Gradient Boosting Classification model with SHAP & LIME) (2089 KB)
* [03_XAI_Gradient_Boosting_Reg(T1)__train_features_salaries.ipynb](03_XAI_Gradient_Boosting_Reg(T1)__train_features_salaries.ipynb) (Explaining Gradient Boosting Regression model with SHAP & LIME) (3193 KB)
* [03_XAI_Linear_Regression(T1)__train_features_salaries.ipynb](03_XAI_Linear_Regression(T1)__train_features_salaries.ipynb) (Explaining Linear Regression model with SHAP & LIME) (4131 KB)
* [03_XAI_Logistic_Regression(T1)__train_features_salaries.ipynb](03_XAI_Logistic_Regression(T1)__train_features_salaries.ipynb) (Explaining Logistic Regression model with SHAP & LIME) (2020 KB)
* [03_XAI_Random_Forest_Regression(T1)__train_features_salaries.ipynb](03_XAI_Random_Forest_Regression(T1)__train_features_salaries.ipynb) (Explaining Random Forest Regression model with SHAP & LIME) (2088 KB)
* [03_XAI_XGBoost_Classification(T1)__train_features_salaries.ipynb](03_XAI_XGBoost_Classification(T1)__train_features_salaries.ipynb) (Explaining XGBoost Classification model with SHAP & LIME) (2019 KB)
* [03_XAI_XGBoost_Regression(T1)__train_features_salaries.ipynb](03_XAI_XGBoost_Regression(T1)__train_features_salaries.ipynb) (Explaining XGBoost Regression model with SHAP & LIME) (3245 KB)
 
**train_features.csv**, **train_salaries.csv** and **test_features.csv** files are not included due to being too large (56,8 MB, 20,6 MB & 56,8 MB).

### FEATURES: train_features.csv (1000000 rows) & train_salaries.csv (1000000 rows)

| Feature | Non-Null Count | Data Type | Used for ML | Preprocessing Notes | 
|---------|----------------|-----------|-------------|---------------------|
| jobId | 1000000 | string | No | |
| companyId | 1000000 | string | No | |
| jobType | 1000000 | string | Yes | Created dummy variables |
| degree | 1000000 | string | Yes | Created dummy variables |
| major | 1000000 | string | Yes | Created dummy variables |
| industry | 1000000 | string | Yes | Created dummy variables |
| yearsExperience | 1000000 | int64 | Yes | |
| milesFromMetropolis | 1000000 | int64 | Yes | |
| salary | 1000000 | int64 | Yes | The dependent feature for the ML models |

### FEATURE SET TYPES & CORRELATIONS

As seen from the file names above, there are 4 types of features set used to examine the effects the salary predictions when used with ML models.

* **Type 1:** Use All Features
* **Type 2:** Use Positively Correlated Features Only
* **Type 3:** Use Top 10 Positively Correlated Features Only
* **Type 4:** Use Most Negatively & Most Positively Correlated Features Only

| Column | Correlation | TYPE 1 | TYPE 2 | TYPE 3 | TYPE 4 |
|--------|-------------|--------|--------|--------|--------|
| job_CEO | 0,285245 | Yes | Yes | Yes | Yes |
| job_CFO | 0,188804 | Yes | Yes | Yes | Yes |
| job_CTO | 0,189600 | Yes | Yes | Yes | Yes |
| job_JANITOR | -0,441660 | Yes | No | No | Yes |
| job_JUNIOR | -0,201993 | Yes | No | No | No |
| job_MANAGER | -0,006781 | Yes | No | No | No |
| job_SENIOR | -0,103642 | Yes | No | No | No |
| job_VICE_PRESIDENT | 0,090941 | Yes | Yes | No | No |
| deg_BACHELORS | 0,111923 | Yes | Yes | No | No |
| deg_DOCTORAL | 0,231391 | Yes | Yes | Yes | Yes |
| deg_HIGH_SCHOOL | -0,203543 | Yes | No | No | Yes |
| deg_MASTERS | 0,171992 | Yes | Yes | Yes | No |
| deg_NONE | -0,257349 | Yes | No | No | Yes |
| maj_BIOLOGY | 0,076341 | Yes | Yes | No | No |
| maj_BUSINESS | 0,126124 | Yes | Yes | Yes | No |
| maj_CHEMISTRY | 0,084046 | Yes | Yes | No | No |
| maj_COMPSCI | 0,102987 | Yes | Yes | No | No |
| maj_ENGINEERING | 0,144176 | Yes | Yes | Yes | No |
| maj_LITERATURE | 0,053920 | Yes | Yes | No | No |
| maj_MATH | 0,110401 | Yes | Yes | No | No |
| maj_NONE | -0,371421 | Yes | No | No | Yes |
| maj_PHYSICS | 0,092057 | Yes | Yes | No | No |
| ind_AUTO | -0,069913 | Yes | No | No | No |
| ind_EDUCATION | -0,175148 | Yes | No | No | No |
| ind_FINANCE | 0,154847 | Yes | Yes | Yes | No |
| ind_HEALTH | -0,003439 | Yes | No | No | No |
| ind_OIL | 0,156959 | Yes | Yes | Yes | No |
| ind_SERIVCE | -0,122361 | Yes | No | No | No |
| ind_WEB | 0,058949 | Yes | Yes | No | No |
| yearsExperience | 0,375013 | Yes | Yes | Yes | Yes |
| milesFromMetropolis | -0,297666 | Yes | No | No | Yes |

**Note:** For all these results below, Random Forest's parameters were heavily tuned to reduce its fitting time reasonably, however the model performed worse than expected!

### RESULTS OF ML MODELS (TYPE 1): train_features.csv & train_salaries.csv (80% Train, 20% Test)

| Model | Fit & Pred Time (sec) | MSE | RMSE | MAE | R2 Score | 5-Fold CV Train | 5-Fold CV Test |
|-------|-----------------------|-----|------|-----|----------|-----------------|----------------|
| Linear Reg | 1,760 | 385,208 | 19,627 | 15,855 | 0,7441 | 0,7435 | 0,7435 |
| Lasso | 1,258 | 498,63 | 22,33 | 17,935 | 0,6688 | 0,6687 | 0,6687 |
| Ridge | 0,452 | 385,207 | 19,627 | 15,855 | 0,7441 | 0,7435 | 0,7435 |
| Polynomial Lasso | 134,969 | 478,971 | 21,885 | 17,637 | 0,6818 | 0,6687 | 0,6687 |
| Polynomial Ridge | 9,511 | 354,743 | 18,835 | 15,313 | 0,7643 | 0,7435 | 0,7435 |
| Random Forest | 23,013 | 898,4 | 29,973 | 24,062 | 0,4032 | 0,4061 | 0,4058 |
| Gradient Boosting | 101,026 | 376,869 | 19,413 | 15,697 | 0,7497 | 0,7499 | 0,7495 |
| XGBoost | 5,032 | 358,34 | 18,935 | 15,378 | 0,7618 | 0,7649 | 0,7608 |

### RESULTS OF ML MODELS (TYPE 2): train_features.csv & train_salaries.csv (80% Train, 20% Test)

| Model | Fit & Pred Time (sec) | MSE | RMSE | MAE | R2 Score | 5-Fold CV Train | 5-Fold CV Test |
|-------|-----------------------|-----|------|-----|----------|-----------------|----------------|
| Linear Reg | 0,719 | 607,353 | 24,645 | 19,825 | 0,5965 | 0,5959 | 0,5959 |
| Lasso | 0,498 | 745,128 | 27,297 | 21,924 | 0,5050 | 0,5050 | 0,5050 |
| Ridge | 0,216 | 607,353 | 24,645 | 19,824 | 0,5965 | 0,5959 | 0,5959 |
| Polynomial Lasso | 11,854 | 790,335 | 28,113 | 22,669 | 0,4749 | 0,5050 | 0,5050 |
| Polynomial Ridge | 1,801 | 592,404 | 24,339 | 19,599 | 0,6064 | 0,5959 | 0,5959 |
| Random Forest | 10,241 | 1297,232 | 36,017 | 28,959 | 0,1383 | 0,1392 | 0,1392 |
| Gradient Boosting | 54,024 | 609,923 | 24,696 | 19,914 | 0,5948 | 0,5942 | 0,5942 |
| XGBoost | 4,100 | 594,158 | 24,375 | 19,624 | 0,6053 | 0,6073 | 0,6041 |

### RESULTS OF ML MODELS (TYPE 3): train_features.csv & train_salaries.csv (80% Train, 20% Test)

| Model | Fit & Pred Time (sec) | MSE | RMSE | MAE | R2 Score | 5-Fold CV Train | 5-Fold CV Test |
|-------|-----------------------|-----|------|-----|----------|-----------------|----------------|
| Linear Reg | 0,372 | 741,672 | 27,234 | 21,928 | 0,5073 | 0,5052 | 0,5051 |
| Lasso | 0,241 | 836,352 | 28,920 | 23,234 | 0,4444 | 0,4439 | 0,4439 |
| Ridge | 0,118 | 741,672 | 27,234 | 21,928 | 0,5073 | 0,5052 | 0,5051 |
| Polynomial Lasso | 6,461 | 888,294 | 29,804 | 24,017 | 0,4099 | 0,4439 | 0,4439 |
| Polynomial Ridge | 0,572 | 727,061 | 26,964 | 21,726 | 0,5170 | 0,5052 | 0,5051 |
| Random Forest | 7,525 | 1296,987 | 36,014 | 28,958 | 0,1384 | 0,1392 | 0,1392 |
| Gradient Boosting | 37,452 | 733,024 | 27,074 | 21,819 | 0,5131 | 0,5110 | 0,5108 |
| XGBoost | 3,452 | 727,628 | 26,975 | 21,735 | 0,5166 | 0,5161 | 0,5142 |

### RESULTS OF ML MODELS (TYPE 4): train_features.csv & train_salaries.csv (80% Train, 20% Test)

| Model | Fit & Pred Time (sec) | MSE | RMSE | MAE | R2 Score | 5-Fold CV Train | 5-Fold CV Test |
|-------|-----------------------|-----|------|-----|----------|-----------------|----------------|
| Linear Reg | 0,41 | 587,751 | 24,244 | 19,435 | 0,6096 | 0,6083 | 0,6083 |
| Lasso | 0,359 | 638,587 | 25,270 | 20,297 | 0,5758 | 0,5755 | 0,5755 |
| Ridge | 0,123 | 587,751 | 24,244 | 19,435 | 0,6096 | 0,6083 | 0,6083 |
| Polynomial Lasso | 7,923 | 650,997 | 25,515 | 20,518 | 0,5676 | 0,5755 | 0,5755 |
| Polynomial Ridge | 0,608 | 570,462 | 23,884 | 19,145 | 0,6210 | 0,6083 | 0,6083 |
| Random Forest | 10,940 | 895,186 | 29,920 | 24,024 | 0,4053 | 0,4080 | 0,4077 |
| Gradient Boosting | 48,533 | 572,802 | 23,933 | 19,191 | 0,6195 | 0,6191 | 0,6187 |
| XGBoost | 3,434 | 572,475 | 23,926 | 19,173 | 0,6197 | 0,6223 | 0,6187 |

In all these results, **Ridge** had the fastest training (fit) and prediction time; **Polynomial Ridge** acheived to have the best MSE, RMSE, MAE and R2 Score and **XGBoost** achieved to obtain the highest CV score.

## 04. disha2disha / Employee-Salary-Prediction

**LINK:** https://github.com/disha2sinha/Employee-Salary-Prediction

### FILES

* [04_EDA(1)__empsalupdated.ipynb](04_EDA(1)__empsalupdated.ipynb) (Exploratory Data Analysis on empsalupdated.csv) (164 KB)
* [04_Empsal.xlsx](04_Empsal.xlsx) (Initial Empsal dataset stored as Excel file) (7 KB)
* [04_Regression__empsalupdated.ipynb](04_Regression__empsalupdated.ipynb) (Predictions with ML Regression models on empsalupdated.csv file) (541 KB)
* [04_empsal.csv](04_empsal.csv) (Initial Empsal dataset stored as CSV file) (2 KB)
* [04_empsalupdated.csv](04_empsalupdated.csv) (Updated Empsal dataset stored as CSV file) (3 KB)

### FEATURES: empsalupdated.csv (26 rows)

| Feature | Non-Null Count | Data Type | Used for ML | Preprocessing Notes | 
|---------|----------------|-----------|-------------|---------------------|
| empno | 26 | int64 | No | |
| empname | 26 | string | No | |
| dob | 26 | sttring | No | |
| sex | 26 | string | Yes | Created dummy variables |
| city | 26 | string | Yes | Created dummy variables |
| state | 26 | string | Yes | Created dummy variables |
| expyr | 26 | string | Yes | |
| salary | 26 | int64 | Yes | The dependent feature for the ML models |
| hra | 26 | int64 | No | |
| Age | 26 | int64 | Yes | |
| conv | 26 | float64 | No | |
| total | 26 | float64 | No | |

### RESULTS OF ML MODELS: empsalupdated.csv (80% Train, 20% Test)

| Model | MSE | RMSE | MAE | R2 Score | 5-Fold CV Train | 5-Fold CV Test |
|-------|-----|------|-----|----------|-----------------|----------------|
| Linear Reg | 70359116,26 | 8398,76 | 6409,148 | 0,8343 | 0,9895 | 0,6974 |
| Lasso | 68898977,96 | 8300,54 | 6306,497 | 0,8382 | 0,9895 | 0,6725 |
| Ridge | 25779238,17 | 5077,33 | 4486,409 | 0,9395 | 0,9851 | 0,8723 |
| Polynomial Lasso | 64234377,88 | 8014,64 | 5993,268 | 0,8491 | 0,9895 | 0,6725 |
| Polynomial Ridge | 182450195,10 | 13507,41 | 12964,645 | 0,5715 | 0,9851 | 0,8723 |
| Random Forest | 64942483,11 | 8058,69 | 6218,85 | 0,8475 | 0,9828 | 0,8253 |
| Quantile | 74279271,93 | 8618,54 | 6406,955 | 0,8256 | 0,9774 | 0,5880 |
| KNN Reg | 323822325,80 | 17995,06 | 15431,40 | 0,2395 | 0,7324 | 0,3702 |
| Gradient Boosting | 28300031,56 | 5319,78 | 3756,159 | 0,9335 | 1,0000 | 0,8763 |
| XGBoost | 116914661,80 | 10812,71 | 7118,133 | 0,7254 | 1,0000 | 0,7353 |

Here, **Ridge** model had the best MSE, RMSE and R2 Score values; together with that, **Gradient Boosting** had the best MAE and CV score values.

## 05. DheerajKumar97 / Employee-Salary-Prediction----python--ML

**LINK:** https://github.com/DheerajKumar97/Employee-Salary-Prediction----python--ML

### FILES

* [05_EDA(1)__train.ipynb](05_EDA(1)__train.ipynb) (Exploratory Data Analysis on train.csv file) (645 KB)
* [05_EDA(2)__test.ipynb](05_EDA(2)__test.ipynb) (Exploratory Data Analysis on test.csv file) (25 KB)
* [05_Regression(T1)__Employee_Salary_Prediction.ipynb](05_Regression(T1)__Employee_Salary_Prediction.ipynb) (Predictions with ML Regression models on train.csv file by using Type-1 feature set) (438 KB)
* [05_Regression(T2)__Employee_Salary_Prediction.ipynb](05_Regression(T2)__Employee_Salary_Prediction.ipynb) (Predictions with ML Regression models on train.csv file by using Type-2 feature set) (523 KB)
* [05_Regression(T3)__Employee_Salary_Prediction.ipynb](05_Regression(T3)__Employee_Salary_Prediction.ipynb) (Predictions with ML Regression models on train.csv file by using Type-3 feature set) (516 KB)
* [05_XAI_Gradient_Boosting(T1)__train.ipynb](05_XAI_Gradient_Boosting(T1)__train.ipynb) (Explaining Gradient Boosting Regression model with SHAP & LIME) (5283 KB)
* [05_XAI_Linear_Regression(T1)__train.ipynb](05_XAI_Linear_Regression(T1)__train.ipynb) (Explaining Linear Regression model with SHAP & LIME) (3936 KB)
* [05_XAI_XGBoost(T1)__train.ipynb](05_XAI_XGBoost(T1)__train.ipynb) (Explaining XGBoost Regression model with SHAP & LIME) (5508 KB)

**train.csv** and **test.csv** files are not included due to being too large (24,3 MB & 9,15 MB).

### FEATURES: train.csv (550068 rows)

| Feature | Non-Null Count | Data Type | Used for ML | Preprocessing Notes | 
|---------|----------------|-----------|-------------|---------------------|
| User_ID | 550068 | int64 | No | |
| Product_ID | 550068 | string | No | |
| Gender | 550068 | string | Yes | Created dummy variables |
| Age | 550068 | string | Yes | Changed data into more meaningful one; created dummy variables |
| Occupation | 550068 | int64 | Yes | Created dummy variables |
| City_Category | 550068 | string | Yes | Changed data into more meaningful one; created dummy variables |
| Stay_In_Current_City_Years | 550068 | string | Yes | Created dummy variables |
| Marital_Status | 550068 | int64 | Yes | |
| Product_Category_1 | 550068 | int64 | Yes | Created dummy variables |
| Product_Category_2 | 376430 | float64 | No | |
| Product_Category_3 | 166821 | float64 | No | |
| Purchase | 550068 | int64 | Yes | The dependent feature for the ML models |

### FEATURE SET TYPES & CORRELATIONS

As seen from the file names above, there are 3 types of features set used to examine the effects the salary predictions when used with ML models.

* **Type 1:** Use all features
* **Type 2:** Use positively created features only
* **Type 3:** Use most negatively and most positively correlated features only (>= 0.1 or <= -0.1)

| Column | Correlation | TYPE 1 | TYPE 2 | TYPE 3 |
|--------|-------------|--------|--------|--------|
| gender_F | -0,060346 | Yes | No | No |
| gender_M | 0,060346 | Yes | Yes | No |
| age_1 | -0,011055 | Yes | No | No |
| age_2 | -0,008831 | Yes | No | No |
| age_3 | -0,001830 | Yes | No | No |
| age_4 | 0,067070 | Yes | Yes | No |
| age_5 | -0,003317 | Yes | No | No |
| age_6 | 0,014792 | Yes | Yes | No |
| age_7 | 0,002904 | Yes | Yes | No |
| city_A | -0,042465 | Yes | No | No |
| city_B | -0,019098 | Yes | No | No |
| city_C | 0,061012 | Yes | Yes | No |
| cityyears_1 | -0,006605 | Yes | No | No |
| cityyears_2 | -0,002030 | Yes | No | No |
| cityyears_3| 0,005358 | Yes | Yes | No |
| cityyears_4 | 0,002090 | Yes | Yes | No |
| cityyears_5 | 0,000988 | Yes | Yes | No |
| occ_0 | -0,010576 | Yes | No | No |
| occ_1 | -0,019005 | Yes | No | No |
| occ_2 | -0,013975 | Yes | No | No |
| occ_3 | -0,003095 | Yes | No | No |
| occ_4 | -0,003872 | Yes | No | No |
| occ_5 | 0,002072 | Yes | Yes | No |
| occ_6 | -0,000290 | Yes | No | No |
| occ_7 | 0,011176 | Yes | Yes | No |
| occ_8 | 0,002839 | Yes | Yes | No |
| occ_9 | -0,13409 | Yes | No | No |
| occ_10 | -0,009409 | Yes | No | No |
| occ_11 | -0,001464 | Yes | No | No |
| occ_12 | 0,025995 | Yes | Yes | No |
| occ_13 | 0,001007 | Yes | Yes | No |
| occ_14 | 0,010772 | Yes | Yes | No |
| occ_15 | 0,015416 | Yes | Yes | No |
| occ_16 | 0,005713 | Yes | Yes | No |
| occ_17 | 0,031099 | Yes | Yes | No |
| occ_18 | -0,002073 | Yes | No | No |
| occ_19 | -0,013769 | Yes | No | No |
| occ_20 | -0,021693 | Yes | No | No |
| prod_cat_1 | 0,506021 | Yes | Yes | Yes |
| prod_cat_2 | 0,084282 | Yes | Yes | No |
| prod_cat_3 | 0,032380 | Yes | Yes | No |
| prod_cat_4 | -0,203982 | Yes | No | Yes |
| prod_cat_5 | -0,370193 | Yes | No | Yes |
| prod_cat_6 | 0,257298 | Yes | Yes | Yes |
| prod_cat_7 | 0,116678 | Yes | Yes | Yes |
| prod_cat_8 | -0,179587 | Yes | No | Yes |
| prod_cat_9 | 0,034110 | Yes | Yes | No |
| prod_cat_10 | 0,201011 | Yes | Yes | Yes |
| prod_cat_11 | -0,195911 | Yes | No | Yes |
| prod_cat_12 | -0,133927 | Yes | No | Yes |
| prod_cat_13 | -0,171660 | Yes | No | Yes |
| prod_cat_14 | 0,040677 | Yes | Yes | No |
| prod_cat_15 | 0,118116 | Yes | Yes | Yes |
| prod_cat_16 | 0,147740 | Yes | Yes | Yes |
| prod_cat_17 | 0,005855 | Yes | Yes | No |
| prod_cat_18 | -0,094670 | Yes | No | No |
| prod_cat_19 | -0,099307 | Yes | No | No |
| prod_cat_20 | -0,128030 | Yes | No | Yes |
| Marital_Status | -0,000463 | Yes | No | No |

**Note:** For all these results below, Random Forest's parameters were heavily tuned to reduce its fitting time reasonably, however the model performed worse than expected!

### RESULTS OF ML MODELS (TYPE 1): train.csv (80% Train, 20% Test)

| Model | Fit & Pred Time (sec) | MSE | RMSE | MAE | R2 Score | 5-Fold CV Train | 5-Fold CV Test |
|-------|-----------------------|-----|------|-----|----------|-----------------|----------------|
| Linear Reg | 2,400 | 9045577,92 | 3007,59 | 2280,14 | 0,6407 | 0,6400 | 0,6300 |
| Lasso | 22,752 | 9050253,92 | 3008,36 | 2281,77 | 0,6405 | 0,6398 | 0,6067 |
| Ridge | 1,647 | 9052203,54 | 3008,69 | 2282,71 | 0,6404 | 0,6399 | 0,6390 |
| Polynomial Ridge | 47,812 | 8898842,30 | 2983,09 | 2247,46 | 0,6465 | 0,6400 | 0,6116 |
| Random Forest | 12,323 | 17154135,84 | 4141,75 | 3135,86 | 0,3186 | 0,3175 | 0,3156 |
| MLPR | 411,924 | 8864204,60 | 2977,28 | 2235,06 | 0,6478 | N/A | N/A |
| Gradient Boosting | 56,282 | 9233670,03 | 3038,70 | 2331,49 | 0,6332 | 0,6337 | 0,6103 |
| XGBoost | 2,827 | 8641033,83 | 2939,56 | 2205,99 | 0,6568 | 0,6615 | 0,6361 |

### RESULTS OF ML MODELS (TYPE 2): train.csv (80% Train, 20% Test)

| Model | Fit & Pred Time (sec) | MSE | RMSE | MAE | R2 Score | 5-Fold CV Train | 5-Fold CV Test |
|-------|-----------------------|-----|------|-----|----------|-----------------|----------------|
| Linear Reg | 0,679 | 10623622,35 | 3295,39 | 2593,13 | 0,5780 | 0,5771 | 0,5764 |
| Lasso | 0,298 | 10629010,55 | 3260,22 | 2593,91 | 0,5777 | 0,5769 | 0,5763 |
| Ridge | 0,511 | 10641790,37 | 3262,18 | 2594,14 | 0,5772 | 0,5765 | 0,5758 |
| Polynomial Ridge | 2,184 | 10563328,62 | 3250,12 | 2579,01 | 0,5804 | 0,5771 | 0,5764 |
| Random Forest | 5,213 | 18671504,21 | 4321,05 | 3240,04 | 0,2583 | 0,2569 | 0,2559 |
| MLPR | 257,716 | 10578768,08 | 3252,50 | 2573,54 | 0,5797 | N/A | N/A |
| Gradient Boosting | 26,332 | 10676026,59 | 3267,41 | 2604,81 | 0,5759 | 0,5756 | 0,5747 |
| XGBoost | 2,162 | 10499499,76 | 3240,29 | 2566,29 | 0,5829 | 0,5864 | 0,5818 |

### RESULTS OF ML MODELS (TYPE 3): train.csv (80% Train, 20% Test)

| Model | Fit & Pred Time (sec) | MSE | RMSE | MAE | R2 Score | 5-Fold CV Train | 5-Fold CV Test |
|-------|-----------------------|-----|------|-----|----------|-----------------|----------------|
| Linear Reg | 0,256 | 9820782,14 | 3133,81 | 2369,38 | 0,6099 | 0,6094 | 0,5912 |
| Lasso | 0,342 | 9821523,92 | 3133,93 | 2370,91 | 0,6099 | 0,6093 | 0,5912 |
| Ridge | 0,248 | 12216471,91 | 3495,21 | 2719,24 | 0,5147 | 0,5146 | 0,5138 |
| Polynomial Ridge | 0,429 | 9820780,49 | 3133,81 | 2369,39 | 0,6099 | 0,6094 | 0,5912 |
| Random Forest | 2,934 | 17166166,91 | 4143,21 | 3137,75 | 0,3181 | 0,3171 | 0,3154 |
| MLPR | 35,05 | 9821469,04 | 3133,92 | 2371,53 | 0,6099 | N/A | N/A |
| Gradient Boosting | 11,291 | 9924217,05 | 3150,27 | 2397,72 | 0,6058 | 0,6052 | 0,5890 |
| XGBoost | 1,264 | 9820788,86 | 3133,81 | 2369,38 | 0,6099 | 0,6094 | 0,5912 |

In all types of feature sets, **XGBoost** is the best model across all metrics (except 5-Fold CV Test score in Type 1, where **Ridge** had better score).

## 06. jhanvi1831 / Software-Engineer-Salary-Prediction

**LINK:** https://github.com/jhanvi831/Software-Engineer-Salary-Prediction

### FILES

* [06_EDA(1)__survey_results_public.ipynb](06_EDA(1)__survey_results_public.ipynb) (Exploratory Data Analysis on survey_results_public.csv file) (101 KB)
* [06_Regression(1)__survey_results_public.ipynb](06_Regression(1)__survey_results_public.ipynb) (Predictions with ML Regression models on survey_results_public.csv file (Series 1)) (643 KB)
* [06_Regression(2-T1)__survey_results_public.ipynb](06_Regression(2-T1)__survey_results_public.ipynb) (Predictions with ML Regression models on survey_results_public.csv file (Series 2) by using Type-1 feature set) (642 KB)
* [06_Regression(2-T2)__survey_results_public.ipynb](06_Regression(2-T2)__survey_results_public.ipynb) (Predictions with ML Regression models on survey_results_public.csv file (Series 2) by using Type-2 feature set) (633 KB)
* [06_Regression(2-T3)__survey_results_public.ipynb](06_Regression(2-T3)__survey_results_public.ipynb) (Predictions with ML Regression models on survey_results_public.csv file (Series 2) by using Type-3 feature set) (632 KB)
* [06_Regression(2-T4)__survey_results_public.ipynb](06_Regression(2-T4)__survey_results_public.ipynb) (Predictions with ML Regression models on survey_results_oublic.csv file (Series 2) by using Type-4 feature set) (634 KB)
* [06_XAI_Gradient_Boosting__survey_results_public.ipynb](06_XAI_Gradient_Boosting__survey_results_public.ipynb) (Explaining Gradient Boosting Regression model with SHAP & LIME) (3651 KB)
* [06_XAI_Linear_Regression__survey_results_public.ipynb](06_XAI_Linear_Regression__survey_results_public.ipynb) (Explaining Linear Regression model with SHAP & LIME) (3683 KB)
* [06_XAI_Random_Forest__survey_results_public.ipynb](06_XAI_Random_Forest__survey_results_public.ipynb) (Explaining Random Forest Regression model with SHAP & LIME) (3711 KB)

**survey_results_public.csv** file is not included due to being too large (77,4 MB).

### FEATURES: survey_results_public.csv (83439 rows)

| Feature | Non-Null Count | Data Type | Used for ML | Preprocessing Notes | 
|---------|----------------|-----------|-------------|---------------------|
| ResponseId | 83439 | int64 | No | |
| MainBranch | 83439 | string | No | |
| Employment | 83323 | string | No | Values of "Employed Full-time" are selected only |
| Country | 83439 | string | Yes | Counts >= 400 are kept individually, otherwise marked as "Other"; Applied label encoding (Series 1); Created dummy variables (Series 2) |
| US_State | 14920 | string | No | |
| UK_Country | 4418 | string | No | |
| EdLevel | 83126 | string | Yes | Rows with null values are removed; Simplified values; Applied label encoding (Series 1); Created dummy variables (Series 2) |
| Age1stCode | 83243 | string | No | |
| LearnCode | 82963 | string | No | |
| YearsCode | 81641 | string | No | |
| YearsCodePro | 61216 | string | Yes | Rows with null values are removed; Processed and converted to integer |
| DevType | 66484 | string | No | |
| OrgSize | 60726 | string | No | |
| Currency | 61080 | string | No | |
| CompTotal | 47183 | float64 | No | |
| CompFreq | 52150 | string | No | |
| LanguageHaveWorkedWith | 82357 | string | No |
| LanguageWantToWorkWith | 76821 | string | No |
| DatabaseHaveWorkedWith | 69546 | string | No |
| DatabaseWantToWorkWith | 58299 | string | No |
| PlatformHaveWorkedWith | 52135 | string | No |
| PlatformWantToWorkWith | 41619 | string | No |
| WebframeHaveWorkedWith | 61707 | string | No |
| WebframeWantToWorkWith | 52095 | string | No |
| MiscTechHaveWorkedWith | 47055 | string | No |
| MiscTechWantToWorkWith | 38021 | string | No |
| ToolsTechHaveWorkedWith | 72537 | string | No |
| ToolsTechWantToWorkWith | 65480 | string | No |
| NEWCollabToolsHaveWorkedWith | 81234 | string | No |
| NEWCollabToolsWantToWorkWith | 73022 | string | No |
| OpSys | 83294 | string | No |
| NEWStuck | 83052 | string | No |
| NEWSOSites | 83171 | string | No |
| SOVisitFreq | 82413 | string | No |
| SOAccount | 82525 | string | No |
| SOPartFreq | 67553 | string | No |
| SOComm | 82319 | string | No |
| NEWOtherComms | 82828 | string | No |
| Age | 82407 | string | No |
| Gender | 82286 | string | No |
| Trans | 80678 | string | No |
| Sexuality | 73366 | string | No |
| Ethnicity | 79464 | string | No |
| Accessibility | 77603 | string | No |
| MentalHealth | 76920 | string | No |
| SurveyLength | 81711 | string | No |
| SurveyEase | 81948 | string | No |
| ConvertedCompYearly | 46844 | float64 | Yes | The dependent feature for the ML Models; Renamed to 'Salary'; Rows with null values are removed; Values within 10000-200000 are selected; Removed those from 'Other' countries |

### FEATURE SET TYPES & CORRELATIONS (SERIES 2)

As seen from the file names above, there are 4 types of features set used in **Series 2** to examine the effects the salary predictions when used with ML models.

* **Type 1:** Use all features
* **Type 2:** Use positively correlated features only
* **Type 3:** Use most negatively and most positively correlated features only (>= 0.1 & <= -0.1)
* **Type 4:** Use negatively correlated features only

| Column | Correlation | TYPE 1 | TYPE 2 | TYPE 3 | TYPE 4 |
|--------|-------------|--------|--------|--------|--------|

## 07. bonniema / salary-predictor

**LINK:** https://github.com/bonniema/salary-predictor/tree/master

## 08. Pranjali1049 / Salary_Prediction

**LINK:** https://github.com/Pranjali1049/Salary_Prediction/tree/main
