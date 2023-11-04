"""
A module for training and evaluating classifiers on multiple datasets.

Additionally, this module includes utility functions for finding the majority number in a given row and choosing a random number in case of a tie,
as well as for finding the majority number in each row of a matrix.

Example:
    To use this module, instantiate a `ClassifierTrainer` object with a list of datasets, and then call its methods to train and evaluate classifiers.

    ```
    from my_classifier_module import ClassifierTrainer

    # Define your datasets (df_fac, df_fou, df_zer) here
    datasets = [df_fac, df_fou, df_zer]

    trainer = ClassifierTrainer(datasets)
    trainer.train_classifiers_with_random_states(n_iterations=3)
    ```
"""
from collections import Counter
import random
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.pipeline import Pipeline
import statistics as st
from tqdm import tqdm


class ClassifierTrainer:
    """
    A class for training and evaluating classifiers on multiple datasets.

    Parameters:
    datasets (list): A list of datasets, each containing features and labels.

    Methods:
    - build_pipeline: Build a data preprocessing and classifier pipeline.
    - find_best_model: Find the best hyperparameters for a classifier using GridSearchCV.
    - print_best_parameters: Print the best hyperparameters found for each dataset.
    - train_voting_classifier: Train a voting classifier using the trained classifiers.
    - train_classifiers_with_random_states: Train classifiers with random states for multiple iterations and calculate confidence intervals for evaluation metrics.
    """

    def __init__(self, datasets, scaler, model_class, param_grid):
        self.datasets = datasets
        self.scaler = scaler
        self.model_class = model_class
        self.param_grid = param_grid
        self.classifiers = []
        self.best_params = []

    def create_labels(self):
        labels = []
        for i in range(10):  # For each class '0' through '9'
            labels.extend([i] * 200)  # Add 200 labels of this class
        return labels

    def build_pipeline(self):
        """
        Build a data preprocessing and classifier pipeline.

        Returns:
        Pipeline: A scikit-learn pipeline containing a scaler and a logistic regression classifier.
        """
        return Pipeline(
            [("scaler", self.scaler), ("clf", self.model_class)]
        )

    def find_best_model(self, X_train, y_train, pipeline, param_grid):
        """
        Find the best hyperparameters for a classifier using GridSearchCV.

        Parameters:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        pipeline (Pipeline): A scikit-learn pipeline.

        Returns:
        dict: The best hyperparameters for the classifier.
        Pipeline: The best pipeline with optimized hyperparameters.
        """

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=10,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_pipeline = grid_search.best_estimator_

        return best_params, best_pipeline

    def find_majority_and_choose_random(self, row):
        """
        Find the majority number in a given row and choose a random number in case of a tie.

        Parameters:
            row (list): A list of numbers representing a row.

        Returns:
            int: The majority number or a random number in case of a tie.
        """
        # Count the occurrences of each number in the row
        counts = Counter(row)

        # Find the maximum count
        max_count = max(counts.values())

        # Create a list of majority numbers
        majority_numbers = [
            num for num, count in counts.items() if count == max_count
        ]

        if len(majority_numbers) == 1:
            # If there is only one majority number, return it
            return majority_numbers[0]
        else:
            # If there is a tie, choose a random number from the majority numbers
            return random.choice(majority_numbers)

    def find_majority_numbers(self, matrix):
        """
        Find the majority number in each row of a matrix and return the results as a list.

        Parameters:
            matrix (list of lists): A 2D matrix where each row is a list of numbers.

        Returns:
            list: A list of majority numbers or random numbers in case of ties for each row.
        """
        result = []
        for row in matrix:
            majority_number = self.find_majority_and_choose_random(row)
            result.append(majority_number)
        return result

    def train_classifiers_with_random_states(self, n_iterations=30):
        """
        Train classifiers with different random states for multiple iterations and calculate confidence intervals for evaluation metrics.

        Parameters:
        n_iterations (int): Number of iterations for training classifiers with different random states.
        """
        # Lists to store evaluation metrics for each iteration
        precision_scores = []
        recall_scores = []
        f1_scores = []
        accuracy_scores = []

        # Define the number of rows (assuming 2000 rows)
        total_rows = 2000
        labels = self.create_labels()  # Create labels for the dataset

        for _ in tqdm(range(n_iterations)):
            # Create train and test indices based on the total number of rows
            train_indices, test_indices = train_test_split(
                list(range(total_rows)),
                test_size=0.2,
                random_state=None,
                stratify=labels,
            )

            arr_y_preds = []
            for i, dataset in enumerate(self.datasets):
                # Filter the rows using the train and test indices
                train_data = dataset.iloc[train_indices]
                test_data = dataset.iloc[test_indices]

                X_train, y_train = (
                    train_data.drop("label", axis=1),
                    train_data["label"],
                )
                X_test, y_test = (
                    test_data.drop("label", axis=1),
                    test_data["label"],
                )

                pipeline = self.build_pipeline()
                best_params, best_pipeline = self.find_best_model(
                    X_train, y_train, pipeline, self.param_grid
                )

                # Make predictions
                y_pred = best_pipeline.predict(X_test)
                arr_y_preds.append(y_pred)

            # Transpose the predictions to get majority labels for each instance
            predictions = [list(row) for row in zip(*arr_y_preds)]
            majority_labels = self.find_majority_numbers(predictions)

            # Calculate evaluation metrics using majority labels
            precision = precision_score(
                y_test, majority_labels, average="weighted"
            )
            recall = recall_score(y_test, majority_labels, average="weighted")
            f1 = f1_score(y_test, majority_labels, average="weighted")
            accuracy = accuracy_score(y_test, majority_labels)

            # Store the evaluation metrics for this iteration
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            accuracy_scores.append(accuracy)

        # Calculate confidence intervals for each metric
        precision_confidence = np.percentile(precision_scores, [2.5, 97.5])
        recall_confidence = np.percentile(recall_scores, [2.5, 97.5])
        f1_confidence = np.percentile(f1_scores, [2.5, 97.5])
        accuracy_confidence = np.percentile(accuracy_scores, [2.5, 97.5])

        # Print the confidence intervals for the evaluation metrics
        print(f"Dataset - Precision Mean: {st.mean(precision_scores)}")
        print(
            f"Dataset - Precision Standard Deviation: {st.stdev(precision_scores)}"
        )
        print(
            f"Dataset - Precision Confidence Interval: {precision_confidence} \n"
        )

        print(f"Dataset - Recall Mean: {st.mean(recall_scores)}")
        print(
            f"Dataset - Recall Standard Deviation: {st.stdev(recall_scores)}"
        )
        print(f"Dataset - Recall Confidence Interval: {recall_confidence}\n")

        print(f"Dataset - F1 Mean: {st.mean(f1_scores)}")
        print(f"Dataset - F1 Standard Deviation: {st.stdev(f1_scores)}")
        print(f"Dataset - F1 Confidence Interval: {f1_confidence}\n")

        print(f"Dataset - Accuracy Mean: {st.mean(accuracy_scores)}")
        print(
            f"Dataset - Accuracy Standard Deviation: {st.stdev(accuracy_scores)}"
        )
        print(
            f"Dataset - Accuracy Confidence Interval: {accuracy_confidence}\n"
        )

        print(classification_report(y_test, majority_labels))
