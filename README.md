# Collection of Salary Predictions

## Overview

This repo aims for which studies I've carried out to understand and gain insights about how salaries can be predicted against several attributes among datasets from various GitHub repos. Generally, these include the operations below, but the methods applied may differ from each other:

* Exploratory Data Analysis (EDA)
* Data Visualizations
* Feature Engineering (Preprocessing)
* Predictions with Machine Learning Regression models
* Comparisons of model results with measurements (MSE, REMSE, MAE, R2, Cross Validation)
* Explainable AI (XAI)

## 01. PlayingNumbers / ds_salary_proj

**LINK:** https://github.com/PlayingNumbers/ds_salary_proj

**FILES:**

* [01_EDA(1)__salary_data_cleaned.ipynb](01_EDA(1)__salary_data_cleaned.ipynb) (Exploratory Data Analysis on salary_data_cleaned.csv file) (599 KB)
* [01_EDA(2)__glassdoor_job.ipynb](01_EDA(2)__glassdoor_job.ipynb) (Exploratory Data Analysis on glassdoor_jobs.csv file) (35 KB)
* [01_EDA(3)__salary_data_cleaned.ipynb](01_EDA(3)__salary_data_cleaned.ipynb) (Additional Exploratory Data Analysis on salary_data_cleaned.csv file) (57 KB)
* [01_Regression__salary_data_cleaned.ipynb](01_Regression__salary_data_cleaned.ipynb) (Predictions with ML Regression models on salary_data_cleaned.csv file) (511 KB)
* [01_eda_data.csv](01_eda_data.csv) (Initial salary dataset) (742 rows) (3075 KB)
* [01_glassdoor_data.csv](01_glassdoor_data.csv) (Salary dataset for Glassdoor jobs) (956 rows) (3784 KB)
* [01_salary_data_cleaned.csv](01_salary_data_cleaned.csv) (Salary dataset as cleaned) (742 rows) (3060 KB)

**FEATURES: salary_data_cleaned.csv**

* Job Title (string)
* Salary Estimate (string)
* Job Description (string)
* Rating (float64) ('-1' values were filled with average)
* Company Name (string)
* Location (string)
* Headquarters (string)
* Size (string)
* Founded (int64)
* Type of ownership (string)
* Industry (string)
* Sector (string)
* Revenue (string)
* Competitors (string)
* hourly (int64)
* employer_provided (int64)
* min_salary (int64) (Converted hourly values)
* max_salary (int64) (Converted hourly values)
* avg_salary (float64) (The dependent feature for the ML models)
* company_txt (string) (Applied data cleaning)
* job_state (string) (Applied data cleaning)
* same_state (int64)
* age (int64) ('-1' values were filled with average)
* python_yn (int64)
* R_yn (int64)
* spark (int64)
* aws (int64)
* excel (int64)

**ADDITIONAL FEATURES: salary_data_cleaned.csv**

* job_simp (string) (Obtained from 'Job Title'; created dummy variables)
* seniority (string) (Obtained from 'Job Title'; created dummy variables)
* desc_len (int64) (Obtained from 'Job Description'; created dummy variables)
* num_comp (int64) (Obtained from 'Competitors')

## 02. Trouble404 / kaggle-Job-Salary-Prediction

LINK: https://github.com/Trouble404/kaggle-Job-Salary-Prediction

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
