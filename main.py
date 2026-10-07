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
from sklearn.metrics import RocCurveDisplay, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks


data = pd.read_csv("data/customer-churn.csv")
data = data.drop(columns=["customerID"])
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(
    0
)  # turn my bullshit invalid strings to NaN
x = data.iloc[:, :-1]
y = data["Churn"]
y = data["Churn"].map({"Yes": 1, "No": 0})

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
        ("categories", OneHotEncoder(), categories),
    ]
)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

fig, ax = plt.subplots(figsize=(10, 8))

#baseline WITH PREPROCESSING
for model_name, current_model in models.items():
    
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("models", current_model)]
    )
    pipeline.fit(x_train, y_train)
    prediction = pipeline.predict(x_test)
    precision = precision_score(y_test, prediction)
    accurary = accuracy_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    confusionMatrix = confusion_matrix(y_test, prediction)
    RocCurveDisplay.from_estimator(pipeline, x_test, y_test, ax=ax, name=model_name)
    print(model_name, model_name)
    print("BASELINE WITH PREPROCESSING")
    print(f"precision: {precision}")
    print(f"accuracy: {accurary}")
    print(f"recall: {recall}")
    print(f"confusion matrix: \n {confusionMatrix}")
    print("-" * 30)


for model_name, current_model in models.items():
    
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("models", current_model)]
    )
    current_hyperparemeters = hyperparameters[model_name]
    
    print(model_name, current_model, current_hyperparemeters)

    #hyperparameter tuning, finding my best model, testing my best model
    grid_search = GridSearchCV(cv=crossValidation, estimator=pipeline, param_grid=current_hyperparemeters, n_jobs=-1, scoring="f1", error_score='raise')
    grid_search.fit(x_train, y_train)
    prediction = grid_search.predict(x_test)

    #metrics
    precision = precision_score(y_test, prediction)
    accurary = accuracy_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    confusionMatrix = confusion_matrix(y_test, prediction)
    RocCurveDisplay.from_estimator(grid_search, x_test, y_test, ax=ax, name=model_name)
    
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")
    print(f"precision: {precision}")
    print(f"accuracy: {accurary}")
    print(f"recall: {recall}")
    print(f"confusion matrix: \n {confusionMatrix}")
    print("-" * 30)

plt.title("ROC Curves for All 6 Models")
plt.plot([0, 1], [0, 1], linestyle='--', color='black') # Adds a diagonal 
plt.savefig('results/roc_curve.png')
plt.show()
