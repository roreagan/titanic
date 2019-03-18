import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from visualization import Visualization

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

visualization = Visualization(titanic)
visualization.missingvalues()
