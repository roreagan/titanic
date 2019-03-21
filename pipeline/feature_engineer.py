import seaborn as sns
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    #If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

class FeatureEngineering():

    def __init__(self, data):
        self.data = data

    def cabinClass(self):
        self.data["Deck"] = self.data.Cabin.str[0]
        self.data["Deck"].unique()
        g = sns.factorplot("Survived", col="Deck", col_wrap=4,
                           data=self.data[self.data.Deck.notnull()],
                           kind="count", size=2.5, aspect=.8)

        self.data = self.data.assign(Deck=self.data.Deck.astype(object)).sort("Deck")
        g = sns.FacetGrid(self.data, col="Pclass", sharex=False,
                          gridspec_kws={"width_ratios": [5, 3, 3]})
        g.map(sns.boxplot, "Deck", "Age")

    def family(self):
        self.data["FamilySize"] = self.data["SibSp"] + self.data["Parch"] + 1
        print(self.data["FamilySize"].value_counts())
        self.data.loc[self.data["FamilySize"] == 1, "FsizeD"] = 'singleton'
        self.data.loc[(self.data["FamilySize"] > 1)  &  (self.data["FamilySize"] < 5) , "FsizeD"] = 'small'
        self.data.loc[self.data["FamilySize"] >4, "FsizeD"] = 'large'

        print(self.data["FsizeD"].unique())
        print(self.data["FsizeD"].value_counts())

    def name(self):
        titles = self.data["Name"].apply(get_title)
        print(pd.value_counts(titles))

        # Add in the title column.
        self.data["Title"] = titles

        # Titles with very low cell counts to be combined to "rare" level
        rare_title = ['Dona', 'Lady', 'Countess', 'Capt', 'Col', 'Don',
                      'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']

        # Also reassign mlle, ms, and mme accordingly
        self.data.loc[self.data["Title"] == "Mlle", "Title"] = 'Miss'
        self.data.loc[self.data["Title"] == "Ms", "Title"] = 'Miss'
        self.data.loc[self.data["Title"] == "Mme", "Title"] = 'Mrs'
        self.data.loc[self.data["Title"] == "Dona", "Title"] = 'Rare Title'
        self.data.loc[self.data["Title"] == "Lady", "Title"] = 'Rare Title'
        self.data.loc[self.data["Title"] == "Countess", "Title"] = 'Rare Title'
        self.data.loc[self.data["Title"] == "Capt", "Title"] = 'Rare Title'
        self.data.loc[self.data["Title"] == "Col", "Title"] = 'Rare Title'
        self.data.loc[self.data["Title"] == "Don", "Title"] = 'Rare Title'
        self.data.loc[self.data["Title"] == "Major", "Title"] = 'Rare Title'
        self.data.loc[self.data["Title"] == "Rev", "Title"] = 'Rare Title'
        self.data.loc[self.data["Title"] == "Sir", "Title"] = 'Rare Title'
        self.data.loc[self.data["Title"] == "Jonkheer", "Title"] = 'Rare Title'
        self.data.loc[self.data["Title"] == "Dr", "Title"] = 'Rare Title'

        # self.data.loc[self.data["Title"].isin(['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 
        #                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']), "Title"] = 'Rare Title'

        # self.data[self.data['Title'].isin(['Dona', 'Lady', 'Countess'])]
        # self.data.query("Title in ('Dona', 'Lady', 'Countess')")

        self.data["Title"].value_counts()


    def convertCategorical(self):
        labelEnc = LabelEncoder()
        cat_vars = ['Embarked', 'Sex', "Title", "FsizeD", "NlengthD", 'Deck']
        for col in cat_vars:
            self.data[col] = labelEnc.fit_transform(self.data[col])

    def featureScaling(self):
        std_scale = StandardScaler().fit(self.data[['Age', 'Fare']])
        self.data[['Age', 'Fare']] = std_scale.transform(self.data[['Age', 'Fare']])
