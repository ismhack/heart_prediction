import sklearn
import pandas as pd
import seaborn as sn
from sklearn import tree
from numpy import sqrt
from numpy import argmax
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

heart_df=pd.read_csv("processed.cleveland.csv")
heart_df.head()
heart_df.describe()

#y = heart_df.PRED

features = ['AGE','SEX', 'CP', 'TRESTBPS', 'CHOL', 'FBS', 'REST_ECG', 'THAL_ACH',  'EXANG', 'OLD_PEAK', 'SLOPE',  'CA',  'THAL']
#X = heart_df[features]

#categorical variables
heart_df_dummies = pd.get_dummies(heart_df, columns=['CP','REST_ECG','SLOPE','CA','THAL'])

print('Categorical Column names: ', heart_df_dummies.columns.tolist())
print(heart_df_dummies.head(5))

y = heart_df_dummies['PRED']
X = heart_df_dummies.drop(['PRED'], axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=5, stratify=y)


## Decision Tree ##
classifier = tree.DecisionTreeClassifier(random_state=0)
path = classifier.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")


clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))


train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()
ix = argmax(test_scores)
print('best ccp_alpha value',ccp_alphas[ix],' produce accuracy ',test_scores[ix] )

classifier = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alphas[ix])
classifier.fit(X_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test)
cm_test = confusion_matrix(y_test, y_pred)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_train, y_pred_train)

print('Accuracy for training set for Decision Tree = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
print('Accuracy for test set for Decision Tree = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))

conf_matrix=pd.DataFrame(data=cm_test,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()
plt.clf()

import graphviz
dot_data = tree.export_graphviz(classifier, out_file=None, feature_names=heart_df_dummies.drop(['PRED'], axis=1).columns.tolist(), class_names=['0','1'],
	filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("heart_tree")

fi = pd.DataFrame({'feature': list(X_train.columns), 'importance': classifier.feature_importances_}).sort_values('importance',ascending = False)

print(fi.head())
