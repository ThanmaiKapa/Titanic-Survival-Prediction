# Titanic Survival Prediction

## Overview

The Titanic Survival Prediction project applies machine learning techniques to predict whether a passenger on the Titanic survived or not. The dataset used for this project is the famous Titanic dataset from Kaggle, which contains details about passengers, such as age, gender, ticket class, and more.

## Dataset

The project uses the Titanic dataset, which contains information about passengers on the Titanic. The dataset was split into training and testing sets for model evaluation.

## Key Features:

PassengerId: Unique identifier for each passenger.

Pclass: Ticket class (1st, 2nd, 3rd).

Name: Passenger name.

Sex: Gender (male/female).

Age: Age in years.

SibSp: Number of siblings/spouses aboard.

Parch: Number of parents/children aboard.

Ticket: Ticket number.

Fare: Passenger fare.

Cabin: Cabin number (if available).

Embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

Survived: Survival status (0 = No, 1 = Yes) [Target variable].

## Project Workflow

Data Preprocessing

Handling missing values.

Encoding categorical variables.

Feature scaling.

Feature engineering (if applicable).

Exploratory Data Analysis (EDA)

Visualizing survival rates based on features.

Identifying key patterns in the dataset.

## Libraries Used

NumPy: Used for numerical computations and handling arrays.

Pandas: Used for data manipulation and preprocessing.

Seaborn: Used for statistical data visualization.

Matplotlib: Used for plotting graphs and charts.

Scikit-learn (sklearn): Machine learning library used for training and evaluating models.

train_test_split: Used to split the dataset into training and testing sets.

LogisticRegression: A classification algorithm used for binary classification problems.

RandomForestClassifier: An ensemble learning method using multiple decision trees for better accuracy.

Support Vector Machine (SVM): A supervised learning algorithm used for classification tasks.

accuracy_score: Used to evaluate the performance of the models.

## Model Selection & Training

Training machine learning models such as Logistic Regression, Random Forest, and Support Vector Machine (SVM).

## Model Evaluation

Comparing models using accuracy score.

## Predictions & Submission

Generating predictions for the test dataset.

## How to Run

### Option 1: Clone the Repository

1. Open a Python interpreter such as Google Colab or Jupyter Notebook.

2. Clone the repository:

git clone https://github.com/ThanmaiKapa/ML_Project_Hub.git

3. Navigate to the project folder and download the dataset.

4. Copy the dataset file path and update it in the code.

5. Run the notebook or script to execute the project.

### Option 2: Download Only the Jupyter Notebook

1. Go to the GitHub repository and open the Jupyter Notebook file (.ipynb).

2. Click on the Raw button to view the raw file.

3. Right-click and select Save As to download the file.

4. Similarly, download the dataset file from the repository.

5. Open the Jupyter Notebook in Google Colab or Jupyter Notebook.

6. Update the dataset file path in the code.

7. Run the notebook to execute the project.

## Model Performance

Logistic Regression Accuracy: 0.79

Random Forest Classifier Accuracy: 0.83

Support Vector Machine Accuracy: 0.72

Since the Random Forest Classifier achieved the highest accuracy, it was chosen for the final prediction model.

Testing the Model

You can input test data and check survival predictions using the code provided in the repository.

## Conclusion

This project demonstrates how machine learning can be used to analyze survival probabilities based on various features. The Random Forest model provided the best results. Future improvements could include feature engineering and additional model tuning to enhance accuracy.
