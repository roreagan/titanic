import matplotlib.pyplot as plot
import seaborn as sns
import numpy as np

class Visualization():

    def __init__(self, data):
        self.data = data

    def missingvalues(self):
        null_columns = self.data.columns[self.data.isnull().any()]
        labels = []
        values = []
        for col in null_columns:
            labels.append(col)
            values.append(self.data[col].isnull().sum())
        ind = np.arange(len(labels))
        width = 0.6
        fig, ax = plot.subplots()
        rects = ax.barh(ind, np.array(values), color='blue')
        ax.set_yticks(ind + (width / 2))
        ax.set_yticklabels(labels, rotation='horizontal')
        ax.set_xlabel("Count of Missing Value")
        ax.set_ylabel("Count Names")
        ax.set_title("Variables with missing value")

    # 各个维度的柱状图
    def drawColumns(self):
        self.data.hist(bins=10, figsize=(9, 7), stacked=True)

    # 绘制多个变量之间的关系——柱状图
    def drawCovariance1(self):
        g = sns.FacetGrid(self.data, col="Sex", row="Survived", margin_titles=True)
        g.map(plot.hist, "Age", color="green")

    # 多个变量关系的点状图
    def drawCovariance2(self):
        g = sns.FacetGrid(self.data, hue="Survived", col="Pclass", margin_titles=True,
                          palette={1: "seagreen", 0: "gray"})
        g = g.map(plot.scatter, "Fare", "Age", edgecolor="w").add_legend()

    # 不同的点状图，其中hue为点的元素，FaceGrid为根据col的类别画多少张图
    def drawConvariance3(self):
        g = sns.FacetGrid(self.data, hue="Survived", col="Sex", margin_titles=True,
                          palette="Set1", hue_kws=dict(marker=["^", "v"]))
        g.map(plot.scatter, "Fare", "Age", edgecolor="w").add_legend()
        plot.subplots_adjust(top=0.8)
        g.fig.suptitle('Survival by Gender , Age and Fare')

    # 画该特征各个类别的统计柱状图
    def drawPer(self):
        self.data.Embarked.value_counts().plot(kind='bar', alpha=0.55)
        plot.title("Passengers per boarding location")

    def draw2Feature(self):
        sns.factorplot(x='Embarked', y="Survived", data=self.data, color="r")

    def drawCovarianceConcrete(self):
        sns.set(font_scale=1)
        g = sns.factorplot(x="Sex", y="Survived", col="Pclass",
                           data=self.data, saturation=.5,
                           kind="bar", ci=None, aspect=.6)
        (g.set_axis_labels("", "Survival Rate")
         .set_xticklabels(["Men", "Women"])
         .set_titles("{col_name} {col_var}")
         .set(ylim=(0, 1))
         .despine(left=True))
        plot.subplots_adjust(top=0.8)
        g.fig.suptitle('How many Men and Women Survived by Passenger Class');

    def draw2FeatureBox(self):
        ax = sns.boxplot(x="Survived", y="Age",
                         data=self.data)
        ax = sns.stripplot(x="Survived", y="Age",
                           data=self.data, jitter=True,
                           edgecolor="gray")
        # sns.plot.title("Survival by Age", fontsize=12)

    # 绘制不同class的数据分布，y轴表示密度
    def drawDistribution(self):
        self.data.Age[self.data.Pclass == 1].plot(kind='kde')
        self.data.Age[self.data.Pclass == 2].plot(kind='kde')
        self.data.Age[self.data.Pclass == 3].plot(kind='kde')
        # plots an axis lable
        plot.xlabel("Age")
        plot.title("Age Distribution within classes")
        # sets our legend for our graph.
        plot.legend(('1st Class', '2nd Class', '3rd Class'), loc='best')

    # draw coefficient matrix and heat map
    def drawCovarianceMatrix(self):
        corr = self.data.corr()  # ["Survived"]
        plot.figure(figsize=(10, 10))

        sns.heatmap(corr, vmax=.8, linewidths=0.01,
                    square=True, annot=True, cmap='YlGnBu', linecolor="white")
        print(self.data.corr()['Survived'])
        plot.title('Correlation between features')


    def drawViolin(self):
        g = sns.factorplot(x="Age", y="Embarked",
                           hue="Sex", row="Pclass",
                           data=self.data[self.data.Embarked.notnull()],
                           orient="h", size=2, aspect=3.5,
                           palette={'male': "purple", 'female': "blue"},
                           kind="violin", split=True, cut=0, bw=.2)

    def drawDistributionOfColumn(self, col):
        with sns.plotting_context("notebook", font_scale=1.5):
            sns.set_style("whitegrid")
            sns.distplot(self.data[col].dropna(),
                         bins=80,
                         kde=True,
                         color="red")
            plot.ylabel("Count")