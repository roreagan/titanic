import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score


train = pd.read_csv('../pipeline/train.csv')
test = pd.read_csv('../pipeline/test.csv')

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data

ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction

cv = ShuffleSplit(n_splits=5, random_state=50)
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)
# scores = cross_val_score(lr, self.data[predictors], self.data["Survived"], scoring='f1', cv=cv)

class SKlearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        return self.clf.fit(x, y).feature_importances_


def get_oof(clf, x_train, y_train):
    scores = cross_val_score(clf, x_train, y_train, scoring='f1', cv=cv)
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


print("*********Start First Class Training**************")

rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

rf = SKlearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SKlearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SKlearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SKlearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SKlearnHelper(clf=SVC, seed=SEED, params=svc_params)

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train) # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train) # Support Vector Classifier

print("Training is complete")

rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)

cols = train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_feature,
     'Extra Trees  feature importances': et_feature,
      'AdaBoost feature importances': ada_feature,
    'Gradient Boost feature importances': gb_feature
    })


#####################################################
# View the importance of Features
# Scatter plot
#####################################################

feature_dataframe['mean'] = feature_dataframe.mean(axis=1)  # axis = 1 computes the mean row-wise
feature_dataframe.head(3)

####################################################
# Second Level Train Based on The Former Training
####################################################
base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
base_predictions_train.head()



# Check Whether the outputs of models are correlated
# train models that are more uncorrelated with one another producing better scores

x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)

# Generate Submission File
StackingSubmission = pd.DataFrame({'Survived': predictions})
StackingSubmission.to_csv("StackingSubmission.csv", index=False)


def visualization():
    trace = go.Scatter(
        y = feature_dataframe['Random Forest feature importances'].values,
        x = feature_dataframe['features'].values,
        mode='markers',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size = 25,
    #       size= feature_dataframe['AdaBoost feature importances'].values,
            #color = np.random.randn(500), #set color equal to a variable
            color = feature_dataframe['Random Forest feature importances'].values,
            colorscale='Portland',
            showscale=True
        ),
        text = feature_dataframe['features'].values
    )
    data = [trace]

    layout= go.Layout(
        autosize= True,
        title= 'Random Forest Feature Importance',
        hovermode= 'closest',
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
        yaxis=dict(
            title= 'Feature Importance',
            ticklen= 5,
            gridwidth= 2
        ),
        showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig,filename='scatter2010')

    # Scatter plot
    trace = go.Scatter(
        y = feature_dataframe['Extra Trees  feature importances'].values,
        x = feature_dataframe['features'].values,
        mode='markers',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size = 25,
    #       size= feature_dataframe['AdaBoost feature importances'].values,
            #color = np.random.randn(500), #set color equal to a variable
            color = feature_dataframe['Extra Trees  feature importances'].values,
            colorscale='Portland',
            showscale=True
        ),
        text = feature_dataframe['features'].values
    )
    data = [trace]

    layout= go.Layout(
        autosize= True,
        title= 'Extra Trees Feature Importance',
        hovermode= 'closest',
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
        yaxis=dict(
            title= 'Feature Importance',
            ticklen= 5,
            gridwidth= 2
        ),
        showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig,filename='scatter2010')

    # Scatter plot
    trace = go.Scatter(
        y = feature_dataframe['AdaBoost feature importances'].values,
        x = feature_dataframe['features'].values,
        mode='markers',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size = 25,
    #       size= feature_dataframe['AdaBoost feature importances'].values,
            #color = np.random.randn(500), #set color equal to a variable
            color = feature_dataframe['AdaBoost feature importances'].values,
            colorscale='Portland',
            showscale=True
        ),
        text = feature_dataframe['features'].values
    )
    data = [trace]

    layout= go.Layout(
        autosize= True,
        title= 'AdaBoost Feature Importance',
        hovermode= 'closest',
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
        yaxis=dict(
            title= 'Feature Importance',
            ticklen= 5,
            gridwidth= 2
        ),
        showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig,filename='scatter2010')

    # Scatter plot
    trace = go.Scatter(
        y = feature_dataframe['Gradient Boost feature importances'].values,
        x = feature_dataframe['features'].values,
        mode='markers',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size = 25,
    #       size= feature_dataframe['AdaBoost feature importances'].values,
            #color = np.random.randn(500), #set color equal to a variable
            color = feature_dataframe['Gradient Boost feature importances'].values,
            colorscale='Portland',
            showscale=True
        ),
        text = feature_dataframe['features'].values
    )
    data = [trace]

    layout= go.Layout(
        autosize= True,
        title= 'Gradient Boosting Feature Importance',
        hovermode= 'closest',
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
        yaxis=dict(
            title= 'Feature Importance',
            ticklen= 5,
            gridwidth= 2
        ),
        showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig,filename='scatter2010')


def drawAverageImportance():
    y = feature_dataframe['mean'].values
    x = feature_dataframe['features'].values
    data = [go.Bar(
        x=x,
        y=y,
        width=0.5,
        marker=dict(
            color=feature_dataframe['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale=False
        ),
        opacity=0.6
    )]

    layout = go.Layout(
        autosize=True,
        title='Barplots of Mean Feature Importance',
        hovermode='closest',
        #     xaxis= dict(
        #         title= 'Pop',
        #         ticklen= 5,
        #         zeroline= False,
        #         gridwidth= 2,
        #     ),
        yaxis=dict(
            title='Feature Importance',
            ticklen=5,
            gridwidth=2
        ),
        showlegend=False
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='bar-direct-labels')

def drawModelsHeaMap():
    data = [
        go.Heatmap(
            z=base_predictions_train.astype(float).corr().values,
            x=base_predictions_train.columns.values,
            y=base_predictions_train.columns.values,
            colorscale='Viridis',
            showscale=True,
            reversescale=True
        )
    ]
    py.iplot(data, filename='labelled-heatmap')