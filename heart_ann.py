import tensorflow as tf 
import pandas as pd 
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt 
import keras 
from keras.models import Sequential 
from keras.layers import Dense 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import binarize
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model

heart_df=pd.read_csv("processed.cleveland.csv")

y = heart_df["PRED"].values
X = heart_df.iloc[:, :-1]
print(y)
heart_df_dummies = pd.get_dummies(X, columns=['CP','REST_ECG','SLOPE','CA','THAL'])
print('Categorical Column names: ', heart_df_dummies.columns.tolist())
print(heart_df_dummies.head(5))
sc = StandardScaler()
#heart_df_dummies = sc.fit_transform(heart_df_dummies)
X_train,X_test,y_train, y_test = train_test_split(heart_df_dummies,y,test_size = 0.2 , random_state = 5) 

X_train = sc.fit_transform(X_train)
X_test_cp =  X_test
X_test = sc.transform(X_test)
print(X_test[0])

classifier = Sequential() 
classifier.add(Dense(activation = "relu", input_dim = 25, units = 13, kernel_initializer = "uniform")) 
classifier.add(Dense(activation = "relu", units = 13, kernel_initializer = "uniform")) 
#classifier.add(Dense(input_dim=13*13, units=13))
#classifier.add(tf.keras.layers.Dropout(rate=0.25))
classifier.add(Dense(activation = "sigmoid", units=1)) 
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics = ['accuracy'] )

classifier.fit(X_train , y_train, batch_size = 8 ,epochs = 150)

#y_pred = classifier.predict(X_test)
y_pred = (classifier.predict(X_test) >= 0.5)
y_test_r = y_test.reshape(y_pred.shape)
visual_pred= np.append(X_test_cp, y_pred,  axis=1)
visual_pred= np.append(visual_pred, y_test_r,  axis=1)

for row in visual_pred:
	if row[len(row) - 1] != row[len(row) - 2]:
		print(row)
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

# plot the roc curve for the model

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='ANN')
#pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
# show the plot
pyplot.show()

plot_model(classifier, to_file='hear_ann.png', show_shapes=True, show_layer_names=True)