import numpy as np
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:13].values
y= dataset.iloc[:,-1].values

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder ,LabelEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('country',OneHotEncoder(),[1])],remainder='passthrough')
x=ct.fit_transform(x)

le = LabelEncoder()
x[:,4]=le.fit_transform(x[:,4])

#dummy variable trap
x=x[:,1:]

#splitting the datasets0
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test= sc.transform(x_test)

'''
a= accuracy on test set
b= Evaluation of model
'''
a={}
b={}

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
def evaluation(name,classifier):
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    acc=accuracy_score(y_test , y_pred)
    a['Test Accuracy '+name]=acc*100
    scores = cross_val_score(classifier,x_train,y_train,cv=10)
    print(scores)
    b["Model Evaluation "+name]=[scores.mean(),scores.std()]



'''
Logistic Regresion
'''
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
evaluation("Logistic Regression",logmodel)

'''
SVM
'''
from sklearn.svm import SVC
svc_r = SVC(kernel='rbf')
evaluation("SVC rbf(kernel)",svc_r)


svc_s = SVC(kernel='sigmoid')
evaluation("SVC sigmoid(kernel)",svc_s)


'''
Decision Tree
'''

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
evaluation("Decision Tree",dtc)

dtc_e = DecisionTreeClassifier(criterion='entropy')
evaluation("Decicion Tree entropy(criterion)",dtc_e)

'''
Random Forest
'''

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
evaluation("Random Forest",rfc)

rfc_e = RandomForestClassifier(n_estimators=100, criterion='entropy')
evaluation("Random forest entropy(criterion)",rfc_e)
