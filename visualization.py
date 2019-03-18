import matplotlib.pyplot as plot
import seaborn as sns
import numpy as np

sns.set(font_scale=1)

class Visualization():

    def __init__(self, data):
        self.data = data

    def missingvalues(self):
        null_columns = [self.data.isnull().any()]
        labels = []
        values = []
        for col in null_columns:
            labels.append(col)
            values.append(self.data[col].isnull().sum())
        ind = np.arange(len(labels))
        width = 0.6
        fig, ax = plot.subplots(figsize=0.6)
        rects = ax.barh(ind, np.array(values), color='blue')
        ax.set_yticks(ind + (width / 2))
        ax.set_xlabel("Count of Missing Value")
        ax.set_ylabel("Count Names")
        ax.set_title("Variables with missing value")


