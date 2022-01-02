
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

#Mounting Our Drive On Cloab
from google.colab import drive
drive.mount('/content/drive')

#uploading test and train files from google drive
train_norm = pd.read_csv('/content/drive/MyDrive/omartrain.csv')
test = pd.read_csv('/content/drive/MyDrive/omartest.csv')
#Using variable for labels
y = train_norm.Cover_Type

#For functions
X = train_norm.drop('Cover_Type', axis=1)

#Splitting The Data y 20% and 80% test train
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

# Applying scikit multiclass Multinomial Naive Bayes Algo
mnb = MultinomialNB()
mnb.fit(t_train, y_train)
Y_pred = mnb.predict(t_test)
acc_mnb = round(mnb.score(t_train, y_train) * 100, 2)
print("Normalized Multinomial Naive Bayes accuracy =",round(acc_mnb,2,), "%")
print(Y_pred.shape)
#Exporting columns from dataframe
submission = pd.DataFrame({
        "Id": test["Id"],
        "Cover_Type": Y_pred
    })
#Making a file converting to csv.
submission.to_csv('Omar_multinomialNaiveBayes.csv', index=False)



