# Job Placement Analysis Project

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
This project focuses on **job placement analysis** using **data analytics** and **predictive modeling** to optimize hiring decisions based on factors such as **academic performance (GPA)**, **work experience**, **demographics**, and **salary expectations**. The main objectives include:

1. Identifying the most influential factors affecting job placement status.
2. Exploring correlations between demographic attributes (e.g., gender, educational background) and placement outcomes.
3. Analyzing the impact of past work experience and salary expectations on job placements.
4. Providing actionable insights for stakeholders to improve job placement strategies and increase employment opportunities.

This analysis offers **personalized career guidance** for individuals and **data-driven recruitment strategies** for organizations.

## Dataset
The dataset used in this project contains the following attributes:
- `gender`: Gender of the candidate.
- `age`: Age of the candidate.
- `gpa`: Grade point average (GPA) of the candidate.
- `years_of_experience`: Years of work experience.
- `salary`: Salary offered (for placed candidates).
- `placement_status`: Whether the candidate was placed (1) or not (0).

### Data Preprocessing
- Missing values for `salary` and `years_of_experience` were replaced with 0.
- Categorical features like `gender` and `placement_status` were label encoded.
- Features were normalized using **MinMaxScaler**.
- **SMOTE** was used to handle class imbalance in the target variable (placement status).

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/job-placement-analysis.git
    ```
   
2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure the dataset `job_placement.csv` is in the appropriate directory, or modify the path in the script accordingly.

## Usage

1. Load the dataset and preprocess the data:
    ```python
    df = pd.read_csv("job_placement.csv")
    # Preprocessing steps...
    ```

2. To visualize the correlation between features:
    ```python
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='Reds')
    plt.show()
    ```

3. Train and evaluate placement prediction models:
    ```python
    xc_train, xc_test, yc_train, yc_test = train_test_split(xc, yc, random_state=42, test_size=0.2)
    # Apply SMOTE and train models...
    ```

4. Train and evaluate salary prediction models:
    ```python
    training_regressor(LinearRegression())
    ```

## Modeling Approach

### Placement Prediction
We used the following features to predict whether a candidate would be placed:
- `gender`
- `age`
- `gpa`
- `years_of_experience`

After data preprocessing, **SMOTE** was applied to handle imbalanced classes. The features were normalized, and models such as **Random Forest**, **Extra Trees**, **Gradient Boosting**, **AdaBoost**, **Linear Regression**, and **XGBoost** were evaluated.

### Salary Prediction
For candidates who were placed, **salary** predictions were made using the same models listed above. The `salary` values were predicted based on:
- `gender`
- `age`
- `gpa`
- `years_of_experience`

The models were evaluated using **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)** metrics.

## Results

| Model                 | MAE               | MSE               |
|-----------------------|-------------------|-------------------|
| **Random Forest**      | 0.82              | 1.14e-07          |
| **Extra Trees**        | 6.18e-12          | 7.35e-31          |
| **Gradient Boosting**  | 1.49              | 1.16e-07          |
| **AdaBoost**           | 240.47            | 0.0013            |
| **Linear Regression**  | 1.59e-12          | 2.97e-31          |
| **XGBoost**            | 0.81              | 1.47e-08          |

Key insights:
- **GPA** and **years of experience** were significant predictors of job placement.
- **Gender** and **salary expectations** showed varying impacts on placement outcomes.
- The **Extra Trees Regressor** and **Linear Regression** models performed best for salary prediction, with minimal error.

## Conclusion
This project highlights how data analytics and machine learning can be applied to **job placement analysis**. By identifying the key factors influencing placement success, this project can help individuals receive **personalized career advice** and enable organizations to make **informed recruitment decisions**, fostering a skilled and diverse workforce.
