

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


from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout



def build_classifier(optimizer,activation_funct_hidden,activation_funct_output,neuron_1,neuron_2,neuron_3):
    classifier = Sequential()
    classifier.add(Dense(neuron_1,kernel_initializer='uniform', activation=activation_funct_hidden,input_dim=11 ))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(neuron_2, kernel_initializer='uniform' , activation = activation_funct_hidden))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(neuron_3, kernel_initializer='uniform' , activation = activation_funct_hidden))
    classifier.add(Dense(1, kernel_initializer='uniform' , activation=activation_funct_output))
    classifier.compile(optimizer=optimizer , loss = 'binary_crossentropy' , metrics=['accuracy'])
    return classifier




classifier = KerasClassifier(build_fn=build_classifier)
parameters = {
    'batch_size':[16,24,32],
    'epochs':[100,300],
    'optimizer': ['adam', 'rmsprop'],
    'activation_funct_hidden':['relu','softmax','sigmoid'],
    'activation_funct_output':['softmax','sigmoid','relu'],
    'neuron_1':[6,9],
    'neuron_2':[6,9],
    'neuron_3':[3,6],
    }    
    
gs= GridSearchCV(estimator=classifier, param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1,return_train_score=True)
gs=gs.fit(x_train,y_train)
best_param = gs.best_params_
print(best_param)
best_accuracy = gs.best_score_
print(best_accuracy)
re = gs.cv_results_
print(re)    







    