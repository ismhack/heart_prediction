import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn
from numpy import sqrt
from numpy import argmax
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import binarize
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from sklearn import tree

heart_df=pd.read_csv("processed.cleveland.csv")
heart_df.head()
heart_df.describe()

def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')
        
    fig.tight_layout()  
    plt.show()
#draw_histograms(heart_df,heart_df.columns,6,3)
#sn.countplot(x='PRED',data=heart_df)
#sn.pairplot(data=heart_df)
heart_df.describe()
#add a constant

st.chisqprob = lambda chisq, df: st.chi2.sf(chisq, df)
cols=heart_df.columns[:-1]
model=sm.Logit(heart_df.PRED,heart_df[cols])
result=model.fit()
print(result.summary())

#def back_feature_elem (data_frame,dep_var,col_list):
""" Takes in the dataframe, the dependent variable and a list of column names, runs the regression repeatedly eleminating feature with the highest
    P-value above alpha one at a time and returns the regression summary with all p-values below alpha"""

#    while len(col_list)>0 :
#        model=sm.Logit(dep_var,data_frame[col_list])
#        result=model.fit(disp=0)
#        largest_pvalue=round(result.pvalues,3).nlargest(1)
#        if largest_pvalue[0]<(0.05):
#            return result
#            break
#        else:
#            col_list=col_list.drop(largest_pvalue.index)

#result=back_feature_elem(heart_df_constant,heart_df.PRED,cols)

#view results

#params = np.exp(result.params)
#conf = np.exp(result.conf_int())
#conf['OR'] = params
#pvalue=round(result.pvalues,3)
#conf['pvalue']=pvalue
#conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio','pvalue']
#print ((conf))

#split data 

y = heart_df.PRED

features = ['AGE','SEX', 'CP', 'TRESTBPS', 'CHOL', 'FBS', 'REST_ECG', 'THAL_ACH',  'EXANG', 'OLD_PEAK', 'SLOPE',  'CA',  'THAL']
X = heart_df[features]

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=5)

#train logistic regression
logreg=LogisticRegression(max_iter=1000, solver = 'lbfgs' )
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

#evaluation
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()
plt.clf()
TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)

print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',

'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',

'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',

'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',

'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)


y_pred_prob=logreg.predict_proba(x_test)[:,:]
y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of no heart disease (0)','Prob of Heart Disease (1)'])
y_pred_prob_df.head()

#for i in range(1,6):
#    cm2=0
#    y_pred_prob_yes=logreg.predict_proba(x_test)
#    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]
#    cm2=confusion_matrix(y_test,y_pred2)
#    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
#            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
#          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
gmeans = sqrt(tpr * (1-fpr))
ix = argmax(gmeans)
print(' Index%f, best Gmean value%.3f ' %(ix, gmeans[ix]))
print('Best Threshold=%f, G-Mean=%.3f' %(thresholds[ix], gmeans[ix]))

y_pred2=binarize(y_pred_prob,thresholds[ix])[:,1]
cm2=confusion_matrix(y_test,y_pred2)
TN=cm2[0,0]
TP=cm2[1,1]
FN=cm2[1,0]
FP=cm2[0,1]
print ('With',thresholds[ix],'threshold the Confusion Matrix is ','\n',cm2,'\n',
           'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
         'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')
print('The acuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n')

conf_matrix=pd.DataFrame(data=cm2,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()
plt.clf()

# plot the roc curve for the model
pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
pyplot.plot(fpr, tpr, marker='.', label='Logistic')
pyplot.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
# show the plot
pyplot.show()
print('Area AUC %f ' %sklearn.metrics.roc_auc_score(y_test,y_pred_prob[:,1]))




