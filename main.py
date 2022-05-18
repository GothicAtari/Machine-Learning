import pandas as pd
import seaborn as sns
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

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
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.35, random_state=1234)

# Build an SVC (Support Vector Classification) model using linear regression
clf = MultinomialNB()
y_pred = clf.fit(X_train, Y_train.ravel()).predict(X_test)

def accuracy(actual, pred):
    sum = 0
    for x in range(len(actual)):
        if actual[x] == pred[x]:
            sum = sum + 1
        elif pred[x] == (actual[x] - 1):
            sum = sum + 0.5
        elif pred[x] == (actual[x] + 1):
            sum = sum + 0.5
    accuracy = sum / len(actual)
    return accuracy
print("The accuracy of the Multinomial Naive Bayes classifier with a 65% training split is: ")
print(accuracy(Y_test, y_pred))

#Confusion matrix time
cm = metrics.confusion_matrix(Y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True)

