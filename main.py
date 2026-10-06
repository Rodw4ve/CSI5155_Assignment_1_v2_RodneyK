import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("data/customer-churn.csv")
data = data.drop(columns=["customerID"])
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(
    0
)  # turn my bullshit invalid strings to NaN
x = data.iloc[:, :-1]
y = data["Churn"]

numbers = ["SeniorCitizen", "tenure", "MonthlyCharges"]
categories = [
    "gender",
    "Partner",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

models = {
    "LogisticRegression": linear_model.LogisticRegression(),
    "DecisionTree": DecisionTreeClassifier(),
    "SupportVectorMachine": svm.SVC(),
    "KNearestNeighbors": KNeighborsClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
}

# my hyperparameters keys need to be the same as my model key
hyperparameters = {
    "LogisticRegression": {
        "models__C": [0.1, 1.0, 10.0],
        "models__solver": ["lbfgs", "liblinear"],
        "models__max_iter": [100, 500],
    },
    "DecisionTree": {
        "models__max_depth": [None, 5, 10, 20],
        "models__min_samples_split": [2, 5, 10],
    },
    "SupportVectorMachine": {
        "models__C": [0.1, 1, 10],
        "models__kernel": ["linear", "rbf"],
    },
    "KNearestNeighbors": {
        "models__n_neighbors": [3, 5, 7, 9],
        "models__weights": ["uniform", "distance"],
    },
    "RandomForest": {
        "models__n_estimators": [50, 100, 200],  # Number of trees
        "models__max_depth": [None, 10, 20],
    },
    "GradientBoosting": {
        "models__n_estimators": [50, 100],
        "models__learning_rate": [0.01, 0.1],
        "models__max_depth": [3, 5],
    },
}
crossValidation = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=0
)  # shuffle=True and random_state=0 are set rn to reproduce the same shit (deterministic randomness)

preprocessor = ColumnTransformer(
    transformers=[
        ("numbers", StandardScaler(), numbers),
        ("categories", OneHotEncoder, categories),
    ]
)

pipeline = Pipeline(
    [("preprocessor", preprocessor), ("models", models["LogisticRegression"])]
)
print(pipeline)


# # class imbalacing -- undersampling, taking shit out, oversampling, creating fake shit using smote
# undersampling imbalancelearn undersampling
# overampling imbalancelearn smote
