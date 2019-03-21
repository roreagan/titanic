import pandas as pd
import seaborn as sns
from visualization import Visualization
from imputation import MissingVlaueImputation
import matplotlib.pyplot as plt

sns.set(font_scale=1)

titanic = pd.read_csv('train.csv')

# # shape of dataset
# titanic.shape
# # reverse of dataser
# titanic.T
# # count, mean, variance of dataset
# titanic.describe()
# # type of every columns
# titanic.info()
# # information of null
# titanic.isnull()
# titanic.isnull().any()  # whether null exists
# titanic.isnull().sum()  # sum the numbers of null
#


# visualization
visualization = Visualization(titanic)
visualization.drawDistributionOfColumn("Age")
plt.show()


# Missing value imputation
# imputation = MissingVlaueImputation(titanic)
# imputation.findAverage()

