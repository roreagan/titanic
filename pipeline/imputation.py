import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np

class MissingVlaueImputation():

    def __init__(self, data):
        self.data = data


    def listMissingItem(self, item):
        print(self.data[self.data[item].isnull()])

    def findAverage(self):
        # method1 watch box graph
        sns.boxplot(hue='Embarked', y='Fare', x='Pclass', data=self.data)
        plt.show()

    def fillMiisingByAverage(self):
        median_fare = self.data[(self.data['Pclass'] == 3) & (self.data['Embarked'] == 'S')]['Fare'].median()
        self.data['Fare'] = self.data['Fare'].fillna(median_fare)

    def fillMissingByRF(self, df):
        age_df = df[['Age', 'Embarked', 'Fare', 'Parch', 'SibSp',
                     'TicketNumber', 'Title', 'Pclass', 'FamilySize',
                     'FsizeD', 'NameLength', "NlengthD", 'Deck']]
        train = age_df.loc[(df.Age.notnull())]  # known Age values
        test = age_df.loc[(df.Age.isnull())]  # null Ages

        # All age values are stored in a target array
        y = train.values[:, 0]
        # All the other values are stored in the feature array
        X = train.values[:, 1::]
        # Create and fit a model
        rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
        rtr.fit(X, y)
        # Use the fitted model to predict the missing values
        predictedAges = rtr.predict(test.values[:, 1::])
        # Assign those predictions to the full data set
        df.loc[(df.Age.isnull()), 'Age'] = predictedAges

        # 通过随机森林可以将特征权重按照重要性采样
        importances = rtr.feature_importances_
        std = np.std([rtr.feature_importances_ for tree in rtr.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        return df