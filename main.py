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
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0) #turn my bullshit invalid strings to NaN
x = data.iloc[:, :-1]
y = data["Churn"]

# print(x)
# print(y)
# print(data.dtypes)

numbers = [
    data["SeniorCitizen"],
    data["tenure"],
    data["MonthlyCharges"],
]
# print(numbers)

categories = [
    data["gender"],
    data["Partner"],
    data["PhoneService"],
    data["MultipleLines"],
    data["InternetService"],
    data["OnlineSecurity"],
    data["OnlineBackup"],
    data["PhoneService"],
    data["MultipleLines"],
    data["InternetService"],
    data["OnlineSecurity"],
    data["OnlineBackup"],
    data["DeviceProtection"],
    data["TechSupport"],
    data["StreamingTV"],
    data["StreamingMovies"],
    data["Contract"],
    data["PaperlessBilling"],
    data["PaymentMethod"],
]

#base models with no hyperparameters - pretty sure I need my hyperparameters. 
linearRegression = linear_model.LinearRegression()
decisionTree = DecisionTreeClassifier()
supportVectorMachine = svm.SVC()
kNearestNeighbors = KNeighborsClassifier()
randomForest = RandomForestClassifier()
gradientBoosting = GradientBoostingClassifier()


crossValidation = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) # shuffle=True and random_state=0 are set rn to reproduce the same shit (deterministic randomness)

preprocessor = ColumnTransformer(transformers=[("numbers", StandardScaler(), numbers), ("categories", OneHotEncoder, categories)])


# preprocessNumbers_StandardScaler = StandardScaler()
# preprocessNumbers_StandardScaler.fit(numbers)
# preprocessNumbers_StandardScaler.mean_
# preprocessNumbers_StandardScaler.transform(numbers)

# preprocessCategories_OneHotEncoding = OneHotEncoder()
# preprocessCategories_OneHotEncoding.fit(categories)
# preprocessCategories_OneHotEncoding.transform(categories).toarray()


# for train_index, test_index in crossValidation.split(x, y):
#     x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
#     y_train_fold, y_test_fold = y[train_index], y[test_index]


pipeline = Pipeline([("preprocessor", preprocessor), ("linearRegression", linearRegression)])
print(pipeline)

# # class imbalacing -- undersampling, taking shit out, oversampling, creating fake shit using smote 
# undersampling imbalancelearn undersampling
# overampling imbalancelearn smote 