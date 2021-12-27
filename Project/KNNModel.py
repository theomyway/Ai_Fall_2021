import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# KNN Model
KNN_f = KNeighborsClassifier(n_neighbors = 3)
KNN_f.fit(X_train, Y_train)
Y_prediction = KNN_f.predict(X_test)
acc_KNN_f = round(KNN_f.score(X_train, Y_train) * 100, 2)
print("-----***-----")
print("  KNN Model  ")
print("-----***-----")
print("KNN Accuracies =",round(acc_KNN_f,2,), "%")
print(Y_prediction.shape)
print(Y_prediction)

submit = pd.DataFrame({
        "": test_df[""],
        "": Y_prediction
  })
submit.to_csv('OmarKhan_KNN.csv', index=False)

KNN_f = KNeighborsClassifier(n_neighbors = 3)
scores = cross_val_score(KNN_f, X_train, Y_train, cv=10, scoring = "accuracy")
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
print("Scores:\n", pd.Series(scores)
