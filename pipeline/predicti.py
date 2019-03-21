import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score

class Predict():

    def __init__(self, data):
        self.data = data

    def LR(self):
        predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
                      "Embarked", "NlengthD", "FsizeD", "Title", "Deck"]
        target = "Survived"
        # Initialize our algorithm class
        alg = LinearRegression()
        # Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
        # We set random_state to ensure we get the same splits every time we run this.
        kf = KFold(self.data.shape[0], n_splits=3, random_state=1)
        predictions = []

        for train, test in kf:
            # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
            train_predictors = (self.data[predictors].iloc[train, :])
            # The target we're using to train the algorithm.
            train_target = self.data[target].iloc[train]
            # Training the algorithm using the predictors and target.
            alg.fit(train_predictors, train_target)
            # We can now make predictions on the test fold
            test_predictions = alg.predict(self.data[predictors].iloc[test, :])
            predictions.append(test_predictions)

        lr = LogisticRegression(random_state=1)
        # Compute the accuracy score for all the cross validation folds.
        cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
        scores = cross_val_score(lr, self.data[predictors], self.data["Survived"], scoring='f1', cv=cv)
        print(scores.mean())

    def Voting(self):
        predictors = []
        lr = LinearRegression()
        rf = RandomForestClassifier(random_state=1, n_estimators=50, max_depth=9,min_samples_split=6, min_samples_leaf=4)
        adb = AdaBoostClassifier()
        eclf1 = VotingClassifier(estimators=[
            ('lr', lr), ('rf', rf), ('adb', adb)], voting='soft')
        eclf1 = eclf1.fit(self.data[predictors], self.data["Survived"])

        test_predictions = eclf1.predict(self.data[predictors])

        test_predictions = test_predictions.astype(int)
        submission = pd.DataFrame({
            "PassengerId": self.data["PassengerId"],
            "Survived": test_predictions
        })

        submission.to_csv("titanic_submission.csv", index=False)