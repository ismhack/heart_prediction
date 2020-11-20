
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from matplotlib import pyplot

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

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20, random_state=5)


forest_model = RandomForestClassifier(n_estimators=250)
forest_model.fit(X_train, y_train)
y_pred = forest_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)
print('Accuracy for test set for ANN = {}'.format((cm[0][0] + cm[1][1])/len(y_test)))

print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',

'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',

'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',

'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',

'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()
plt.clf()

# Extract single tree
estimator = forest_model.estimators_[0]
import graphviz
dot_data = export_graphviz(estimator, out_file=None, feature_names=heart_df_dummies.drop(['PRED'], axis=1).columns.tolist(), class_names=['0','1'],
	filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("heart_random_forest")

fi = pd.DataFrame({'feature': list(X_train.columns), 'importance': forest_model.feature_importances_}).sort_values('importance',ascending = False)

print(fi.head())


# plot the roc curve for the model

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='Random Forest')
#pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
# show the plot
pyplot.show()
