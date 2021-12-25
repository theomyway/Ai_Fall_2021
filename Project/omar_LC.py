import os

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
train_data = pd.read_csv(os.path.join("..", "input", "titanic-machine-learning-from-disaster", "C:/Users/Admin/Desktop/project/titanic_train.csv"))
test_data = pd.read_csv(os.path.join("..", "input", "titanic-machine-learning-from-disaster", "C:/Users/Admin/Desktop/project/titanic_test.csv"))


train_data.head()

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
labels = ["Survived"]

X_train = train_data[features]
y_train = np.ravel(train_data[labels])

X_test = test_data[features]

X_train.head()
def nan_clms(df):
    return [clm for clm in X_train.columns if df[clm].isnull().any()]

print("Columns with missing values in train data: {}.".format(nan_clms(X_train)))
print("Columns with missing values in test data: {}.".format(nan_clms(X_test)))

preprocessor = ColumnTransformer(
    [("num_imputer", SimpleImputer(strategy="median"), ["Age", "Fare"]),
     ("encoder", OneHotEncoder(drop="first"), ["Pclass", "Sex", "Embarked"])], 
    remainder = "passthrough")

model = RidgeClassifier(solver="lsqr")

pipe = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("model", model)])

cv_score = np.mean(cross_val_score(pipe, X_train, y_train, cv=10))
print("The linear classifier has an accuracy of: {:.3f}.".format(cv_score))


test_data = pd.read_csv(os.path.join("..", "input", "titanic-machine-learning-from-disaster", "test.csv"))
X_test = test_data[features]


pipe.fit(X_train, y_train)
predictions  = pipe.predict(X_test)
output = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": predictions})
output.to_csv("submission.csv", index=False)


