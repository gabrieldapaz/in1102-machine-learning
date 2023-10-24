# IN1102 Machine Learning

This repository is for the machine learning course, which is part of the Computer Science postgraduate program at CIn/UFPE. It contains projects focused on evaluating and comparing machine learning classifiers using various metrics and statistical tests.

## Description

The projects in this repository use three datasets from the Multiple Features Dataset. The classifiers evaluated are Gaussian Bayesian, k-nearest neighbor-based Bayesian, Parzen window-based Bayesian, and logistic regression classifiers. 

The evaluation process involves the following steps:

a) A "30 × 10-fold" stratified cross-validation is used to evaluate and compare the classifiers using the majority vote rule. When necessary, a validation set (20%) is set aside from the learning dataset to perform hyperparameter tuning. The model is then retrained with the learning dataset + validation set using stratified sampling.

b) A point estimate and a confidence interval are obtained for each classifier evaluation metric (Error rate, Precision, Recall, F-measure).

c) The Friedman test (a non-parametric test) is used to compare the classifiers, and the post hoc test (Nemenyi test) is used for each of the metrics.

## Installation

To install the necessary dependencies for these projects, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the cloned repository.
3. Make the `setup.sh` script executable by running the following command:
    ```
    chmod +x setup.sh
    ```
4. Run the script to create a virtual environment and install the required packages:
    ```
    ./setup.sh
    ```

This will set up your environment with the necessary dependencies.

## Running the Code

To run the code in this repository:

1. Open Jupyter Notebook:
    ```
    jupyter notebook
    ```
2. Navigate to the `project.ipynb` file in the Jupyter Notebook interface.
3. Run the cells in `project.ipynb` to execute the code.

Please note that you may need to adjust file paths or other parameters specific to your system or datasets.
