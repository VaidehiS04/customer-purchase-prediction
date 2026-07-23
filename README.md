# Customer Purchase Prediction - AI/ML Assignment

## Project Overview

This project is my submission for the AI/ML assignment. The goal was to
build a machine learning model that predicts whether a customer will
purchase a product, based on their details like age, income, time spent
on the website, etc.

I used the Predict Customer Purchase Behavior Dataset from Kaggle


The project covers:
- Exploratory Data Analysis (EDA)
- Data preprocessing (encoding + scaling)
- Training 3 ML models (Logistic Regression, Decision Tree, Random Forest)
- Evaluating the models (accuracy, precision, recall, f1, confusion matrix)
- Improving the model (cross-validation, hyperparameter tuning on the best model)
- A simple Streamlit app where you can enter customer details and get a prediction


## Files in this project

- train.py                         - EDA, preprocessing, model training, evaluation, tuning
- app.py                           - Streamlit app for prediction (app.py)
- dataset.csv                      - dataset 
- requirements.txt                 - all the libraries needed
- purchase_prediction_model.pkl    - saved trained model 
- scaler.pkl                       - saved StandardScaler 
- model_columns.pkl                - saved column order 
- README.md


## Setup Instructions

1. Make sure Python is installed.
2. (Optional but recommended) create a virtual environment:
   python -m venv venv
   venv\Scripts\activate 
3. Install all the required libraries:
   pip install -r requirements.txt
4. Download the dataset from the Kaggle:
   dataset.csv

## Required Libraries

Everything needed is listed in requirements.txt:
- pandas
- numpy
- matplotlib
- scikit-learn
- streamlit

## How to Run the Project

Step 1: Run the EDA + model training file first

Open train.py in VS Code, run the file in the terminal.

This will:
- Show all the EDA charts and print the analysis
- Train all three models (Logistic Regression, Decision Tree, Random Forest) and print their scores
- Tune the best performing model (Random Forest) using GridSearchCV
- Save the final tuned model as purchase_prediction_model.pkl
- Also save scaler.pkl and model_columns.pkl

Step 2: Run the Streamlit app

Once the .pkl files are created, run this in the terminal:
streamlit run app.py
This will open the app in the browser where you can enter customer
details (age, income, product category, etc.) and click Predict to
see if the customer is likely to buy or not, along with a confidence
percentage.

## Assumptions Made

- The dataset does not have any missing values, but I still added code
  to fill missing values (using median for numbers) just in case the
  data changes later or a different version of the dataset is used.
- Gender, ProductCategory and LoyaltyProgram were already given as
  numbers (0/1, 0-4) in the dataset, so I just ran them through
  LabelEncoder to make the encoding step clear, instead of assuming
  they needed to be converted from text.
- I checked the class distribution of PurchaseStatus in Task 1 and it
  came out fairly balanced (roughly 55-45), so I assumed class
  balancing techniques (like class_weight="balanced") were not
  necessary for this dataset and skipped that step in Task 5.
- I picked Logistic Regression, Decision Tree and Random Forest as the
  three models since they are commonly used and easy to explain,
  rather than jumping straight into more complex models. I assumed
  Random Forest would perform the best out of the three since it
  combines multiple decision trees, which the results in Task 4
  confirmed, so hyperparameter tuning in Task 5 was applied to Random
  Forest specifically.

## Challenges Faced

- The dataset I found first (an e-commerce behavior dataset) had a
  huge class imbalance problem (almost all rows had the same target
  value), so I had to switch to this Kaggle dataset instead since it
  had a more realistic and balanced target column.
- Understanding why scaling matters for Logistic Regression but not
  really for Decision Tree took me a bit of reading before I actually
  understood it, instead of just doing it blindly.
- Hyperparameter tuning using GridSearchCV took a little while to
  figure out - especially picking which parameters to actually put in
  the grid without making it take too long to run.
- Comparing three models instead of two meant I had to decide which
  one to actually spend time tuning in Task 5 - I went with Random
  Forest since it came out on top in Task 4, instead of tuning all
  three separately.
- Making sure the Streamlit app encodes the user's input in the exact
  same way the training data was encoded (same column order, same
  number codes) took some trial and error, since even a small mismatch
  gives wrong predictions.
