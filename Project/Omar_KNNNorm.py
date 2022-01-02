import pandas as pds
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

#Mounting Our Drive On Cloab
from google.colab import drive
drive.mount('/content/drive')

#uploading files of test and train from drive
train_norm = pds.read_csv('/content/drive/MyDrive/omartrain.csv')
test = pds.read_csv('/content/drive/MyDrive/omartest.csv')


#Using variable for labels
y = train_norm.Cover_Type

#For functions
X = train_norm.drop('Cover_Type', axis=1)

#Splitting The Data y 20% and x 80%
t_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
#Normalizing the data
def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())

    return df_norm

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


#Applying KNN scikit multiclass identifier
KNN_f = KNeighborsClassifier(n_neighbors = 3)
KNN_f.fit(t_train, y_train)
Y_prediction = KNN_f.predict(t_test)
acc_KNN_f = round(KNN_f.score(t_train, y_train) * 100, 2)
print("  KNN Model  ")
print("KNN Accuracies =",round(acc_KNN_f,2,), "%")
print(Y_prediction.shape)
print(Y_prediction)

#Exporting Columns into dataframe
submission = pds.DataFrame({
        "Id": test["Id"],
        "Cover_Type": Y_prediction
    })
#Making a file
submission.to_csv('Omar_KNN.csv', index=False)



