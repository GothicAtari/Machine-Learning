import pandas as pd
from sklearn import model_selection
from sklearn import svm

#Read dataset
df = pd.read_csv("NormalizedNBAData.data", names = ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TOPG', 'target'])
features = ['PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TOPG']

#Extract features
X = df.loc[:, features].values

#Extract target i.e. salary
Y = df.loc[:, ['target']].values

#Split the data into train/test data sets
#20% testing
#80% training
#Randomly chosen
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.4, random_state=100)

# Build an SVC (Support Vector Classification) model using linear regression
clf = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train.ravel())

print(clf.score(X_test, Y_test))