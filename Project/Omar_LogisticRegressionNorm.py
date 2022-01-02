import numpy as np
import pandas as pd
import pandas as pds
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

#uploading files of test and train from drive
train_norm = pds.read_csv('/content/drive/MyDrive/omartrain.csv')
test = pds.read_csv('/content/drive/MyDrive/omartest.csv')

#Using variable for labels.
y = train_norm.Cover_Type

#For functions
X = train_norm.drop('Cover_Type', axis=1)

#Splitting The Data y 20% and x 80%
t_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
#For normalized data
def min_max_scaling(df):
    # copying the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())

    return df_norm
#Scaling the data and normalizing it
train_norm = min_max_scaling(train)
train_norm = train_norm.fillna(0.5)
y_train = train_norm.Cover_Type;
train_norm.drop('Cover_Type',inplace=True,axis=1)
t_train = train_norm

scaler = MinMaxScaler()
t_train = scaler.fit_transform(t_train)
t_test = scaler.fit_transform(test)
t_train = t_train.astype('float')
y_train = y_train.astype('float')
t_test = t_test.astype('int')

mnb = linear_model.Lasso(alpha=1)
mnb.fit(train_norm,y_train)
# Predictions
predictions = mnb.predict(test)
print(predictions.shape)


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(t_train, y_train)
Y_pred = logreg.predict(t_test)
acc_log = round(logreg.score(t_train, y_train) * 100, 2)

print("--------------Logistic Regression Model--------------")

print("Linear Classifiers Accuracy =",round(acc_log,2,), "%")
print(Y_pred.shape)
print(Y_pred)
#Exporting columns from dataframe
submission = pd.DataFrame({
        "Id": test["Id"],
        "Cover_Type": Y_pred
    })
submission.to_csv('OmarLogisticRegressionNorm.csv', index=False)


