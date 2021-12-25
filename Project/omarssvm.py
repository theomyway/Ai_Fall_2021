# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# Load the data. 
training_data = pd.read_csv('C:/Users/Admin/Desktop/project/titanic_train.csv')
test_data = pd.read_csv('C:/Users/Admin/Desktop/project/titanic_test.csv')
# Examine the first few rows of data in the training set. 
training_data.head()

def add_title(data):
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    data.Title = data.Title.replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 
                                     'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data.Title = data.Title.replace('Mlle', 'Miss')
    data.Title = data.Title.replace('Ms', 'Miss')
    data.Title = data.Title.replace('Mme', 'Mrs')
    
    # Map from strings to numerical variables.
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    
    data.Title = data.Title.map(title_mapping)
    data.Title = data.Title.fillna(0)

    return data


    missing_emb_training = training_data[pd.isnull(training_data.Embarked) == True]
missing_emb_test = test_data[pd.isnull(test_data.Embarked) == True]
missing_emb_training.head()

missing_emb_test.head()

grid = sns.FacetGrid(training_data[training_data.Pclass == 1], col='Embarked', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Fare', alpha=.5, bins=20)
grid.map(plt.axvline, x=80.0, color='red', ls='dashed')
grid.add_legend();


# Recast port of departure as numerical feature. 
def simplify_embark(data):
    # Two missing values, assign Cherbourg as port of departure.
    data.Embarked = data.Embarked.fillna('C')
    
    le = preprocessing.LabelEncoder().fit(data.Embarked)
    data.Embarked = le.transform(data.Embarked)
    
    return data

    missing_fare_training = training_data[np.isnan(training_data['Fare'])]
missing_fare_test = test_data[np.isnan(test_data['Fare'])]
missing_fare_training.head()


missing_fare_test.head()

# Find median fare, plot over resulting distribution. 
fare_med = np.median(combine.Fare)

sns.kdeplot(combine.Fare, shade=True)
plt.axvline(fare_med, color='r', ls='dashed', lw='1', label='Median')
plt.legend();

test_data['Fare'] = test_data['Fare'].fillna(fare_med)

print('Percentage of Missing Cabin Numbers (Training): %0.1f' % missing_cabin_training)
print('Percentage of Missing Cabin Numbers (Test): %0.1f' % missing_cabin_test)


def simplify_cabins(data):
    data.Cabin = data.Cabin.fillna('N')
    data.Cabin = data.Cabin.apply(lambda x: x[0])
    
    #cabin_mapping = {'N': 0, 'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 
    #                 'F': 1, 'G': 1, 'T': 1}
    #data['Cabin_Known'] = data.Cabin.map(cabin_mapping)
    
    le = preprocessing.LabelEncoder().fit(data.Cabin)
    data.Cabin = le.transform(data.Cabin)
    
    return data

     Recast sex as numerical feature. 
def simplify_sex(data):
    sex_mapping = {'male': 0, 'female': 1}
    data.Sex = data.Sex.map(sex_mapping).astype(int)
    
    return data

# Drop all unwanted features (name, ticket). 
def drop_features(data):
    return data.drop(['Name','Ticket'], axis=1)

# Perform all feature transformations. 
def transform_all(data):
    data = add_title(data)
    data = simplify_embark(data)
    data = simplify_cabins(data)
    data = simplify_sex(data)
    data = drop_features(data)
    
    return data

training_data = transform_all(training_data)
test_data = transform_all(test_data)

all_data = [training_data, test_data]
combined_data = pd.concat(all_data)
# Inspect data.
combined_data.head()



null_ages = pd.isnull(combined_data.Age)
known_ages = pd.notnull(combined_data.Age)
initial_dist = combined_data.Age[known_ages]

# Examine distribution of ages prior to imputation (for comparison). 
sns.distplot(initial_dist)



def impute_ages(data):
    drop_survived = data.drop(['Survived'], axis=1)
    column_titles = list(drop_survived)
    mice_results = fancyimpute.MICE().complete(np.array(drop_survived))
    results = pd.DataFrame(mice_results, columns=column_titles)
    results['Survived'] = list(data['Survived'])
    return results

# Examine distribution of ages after imputation (for comparison). 
sns.distplot(initial_dist, label='Initial Distribution')
sns.distplot(complete_data.Age, label='After Imputation')
plt.title('Distribution of Ages')
plt.legend()


scaler = preprocessing.StandardScaler()
select = 'Age Fare'.split()
complete_data[select] = scaler.fit_transform(complete_data[select])

training_data = complete_data[:891]
test_data = complete_data[891:].drop('Survived', axis=1)

# ----------------------------------
# Support Vector Machines
droplist = 'Survived PassengerId'.split()
data = training_data.drop(droplist, axis=1)
# Define features and target values
X, y = data, training_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


params = {'kernel': 'rbf', 'C': 107.54222939713921, 'gamma': 0.013379109762586716, 'class_weight': None}
clf = SVC(**params)
scores = cross_val_score(clf, X, y, cv=4, n_jobs=-1)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

droplist = 'PassengerId'.split()
clf.fit(X,y)
predictions = clf.predict(test_data.drop(droplist, axis=1))
#print(predictions)
print('Predicted Number of Survivors: %d' % int(np.sum(predictions)))