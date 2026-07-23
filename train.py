import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# TASK 1: DATA ANALYSIS
# Step 1: Load the dataset

df = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")
print("Shape of the dataset (rows, columns):", df.shape)

# Step 2: Basic information about the dataset
print("\nFirst 5 rows")
print(df.head())

print("\nDataset Info")
df.info()

print("\nColumn names")
print(df.columns.tolist())

# Step 3: Check for missing values
print("\nMissing values in each column")
print(df.isnull().sum())


df["Age"] = df["Age"].fillna(df["Age"].median())
df["AnnualIncome"] = df["AnnualIncome"].fillna(df["AnnualIncome"].median())
df["NumberOfPurchases"] = df["NumberOfPurchases"].fillna(df["NumberOfPurchases"].median())
df["TimeSpentOnWebsite"] = df["TimeSpentOnWebsite"].fillna(df["TimeSpentOnWebsite"].median())
df["DiscountsAvailed"] = df["DiscountsAvailed"].fillna(df["DiscountsAvailed"].median())

print("\nMissing values after filling")
print(df.isnull().sum())

# Step 4: Check for duplicate rows
print("\nDuplicate rows in the dataset")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\n Duplicate rows after removing")
print(df.duplicated().sum())

# Step 5: Summary statistics
print("\nSummary statistics")
print(df.describe())

print("\nTarget column value counts (PurchaseStatus)")
print(df["PurchaseStatus"].value_counts())
print("\nPercentage split:")
print(df["PurchaseStatus"].value_counts(normalize=True) * 100)

# Step 6: Feature distributions

plt.hist(df["Age"], color="skyblue", edgecolor="black", bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of customers")
plt.show()

plt.hist(df["AnnualIncome"], color="lightgreen", edgecolor="black", bins=10)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Number of customers")
plt.show()

plt.hist(df["NumberOfPurchases"], color="orange", edgecolor="black", bins=10)
plt.title("Number of Purchases Distribution")
plt.xlabel("Number of Purchases")
plt.ylabel("Number of customers")
plt.show()

plt.hist(df["TimeSpentOnWebsite"], color="violet", edgecolor="black", bins=10)
plt.title("Time Spent on Website Distribution")
plt.xlabel("Time Spent on Website (minutes)")
plt.ylabel("Number of customers")
plt.show()

plt.hist(df["DiscountsAvailed"], color="gold", edgecolor="black", bins=6)
plt.title("Discounts Availed Distribution")
plt.xlabel("Discounts Availed")
plt.ylabel("Number of customers")
plt.show()


# target column distribution
purchase_counts = df["PurchaseStatus"].value_counts().sort_index()

plt.bar(["No Purchase (0)", "Purchase (1)"], purchase_counts, color=["red", "green"])
plt.title("Purchase Status Distribution")
plt.xlabel("Purchase Status")
plt.ylabel("Number of customers")
plt.show()

gender_counts = df["Gender"].value_counts().sort_index()
plt.bar(["Male (0)", "Female (1)"], gender_counts, color=["blue", "pink"])
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

category_counts = df["ProductCategory"].value_counts().sort_index()
plt.bar(["Electronics", "Clothing", "Home Goods", "Beauty", "Sports"],
        category_counts, color=["red", "blue", "green", "purple", "orange"])
plt.title("Product Category Distribution")
plt.xlabel("Product Category")
plt.ylabel("Count")
plt.show()

loyalty_counts = df["LoyaltyProgram"].value_counts().sort_index()
plt.bar(["Not Member (0)", "Member (1)"], loyalty_counts, color=["gray", "teal"])
plt.title("Loyalty Program Distribution")
plt.xlabel("Loyalty Program")
plt.ylabel("Count")
plt.show()

# Step 7: Correlation heatmap

corr = df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

fig.colorbar(im)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# TASK 2: DATA PREPROCESSING

# Step 1: Encode categorical features
from sklearn.preprocessing import LabelEncoder

label_transformer = LabelEncoder()

df["Gender"] = label_transformer.fit_transform(df["Gender"])
df["ProductCategory"] = label_transformer.fit_transform(df["ProductCategory"])
df["LoyaltyProgram"] = label_transformer.fit_transform(df["LoyaltyProgram"])

print(df.head())


df_encoded = df.copy()

# Step 2: Split features (X) and target (y)

X = df_encoded.drop(columns=["PurchaseStatus"])
y = df_encoded["PurchaseStatus"]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Step 3: Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

# Step 4: Feature scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling done with StandardScaler.")

# TASK 3: MODEL DEVELOPMENT


# Model 1: Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)
log_acc = accuracy_score(y_test, log_pred)
print("Logistic Regression Accuracy:", round(log_acc, 4))

# Model 2: Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_pred = dt_model.predict(X_test_scaled)
dt_acc = accuracy_score(y_test, dt_pred)
print("Decision Tree Accuracy:", round(dt_acc, 4))

# Model 3: Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", round(rf_acc, 4))

# Compare all three models
compare_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [log_acc, dt_acc, rf_acc]
})
print("\nModel Comparison")
print(compare_df)

# TASK 4: MODEL EVALUATION

def evaluate_model(name, y_true, y_pred):
    """Prints all the evaluation metrics for one model."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{name}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return acc, prec, rec, f1


log_metrics = evaluate_model("Logistic Regression", y_test, log_pred)
dt_metrics = evaluate_model("Decision Tree", y_test, dt_pred)
rf_metrics = evaluate_model("Random Forest", y_test, rf_pred)

# Confusion matrix plots
def plot_cm(ax, y_true, y_pred, title, cmap):
    cm = confusion_matrix(y_true, y_pred)
    ax.imshow(cm, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Purchase", "Purchase"])
    ax.set_yticklabels(["No Purchase", "Purchase"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
plot_cm(axes[0], y_test, log_pred, "Logistic Regression", "Blues")
plot_cm(axes[1], y_test, dt_pred, "Decision Tree", "Greens")
plot_cm(axes[2], y_test, rf_pred, "Random Forest", "Oranges")
plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.show()

# TASK 5: IMPROVE MODEL PERFORMANCE
# Technique 1: Cross-validation

cv_scores = cross_val_score(RandomForestClassifier(random_state=42),
                             X_train_scaled, y_train, cv=5, scoring="f1")
print("Cross-validation F1 scores:", cv_scores)
print("Average CV F1 score:", round(cv_scores.mean(), 4))

# Technique 2: Hyperparameter tuning with GridSearchCV
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
grid_search.fit(X_train_scaled, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best CV F1 score:", round(grid_search.best_score_, 4))

# Final model after tuning
final_model = grid_search.best_estimator_
final_pred = final_model.predict(X_test_scaled)

print("\nFINAL MODEL (Tuned Random Forest)")
evaluate_model("Final Tuned Random Forest", y_test, final_pred)


with open("purchase_prediction_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("\nModel, scaler and column list saved successfully!")
print("Files created: purchase_prediction_model.pkl, scaler.pkl, model_columns.pkl")
