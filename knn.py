import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  # library only used for splitting the dataset


def import_data():
    df = pd.read_csv('Datasets/Iris.csv')
    df.drop('Id', axis=1, inplace=True)
    return df


def print_data_sample(df):
    print(df.sample(5))     # print 5 samples of dataset


def plot_data(df, feature1, feature2, output):
    # plotting all samples
    sns.FacetGrid(df, height=6, aspect=0.8, hue=output).map(plt.scatter, feature1, feature2).add_legend()


def split_data(df, n_test):
    X = df[:][['SepalLengthCm', 'SepalWidthCm', 'PetalWidthCm', 'PetalLengthCm']]
    y = df[:]['Species']
    return train_test_split(X, y, test_size=n_test, random_state=0)


# k-nearest neighbour implementation
def knn(test_n):
    k = int(input('\nEnter k: '))
    y_predict = []
    for i in range(0, test_n):
        dist = np.linalg.norm(X_train - X_test.iloc[i], axis=1)  # calculating distance of all training samples from test sample
        nn = dist.argsort()  # nn has index values of samples sorted acc to min dist
     
        knn = nn[:k]  # k nearest neighbour index
        y_predict.append(max(set(y_train.iloc[knn].values)))  # store predicted values in a list

        # plotting test samples in red and neighbours in black
        plt.scatter(X_test['SepalLengthCm'].values[i], X_test['PetalLengthCm'].values[i], marker="o", s=20, color="red")
        plt.scatter(X_train['SepalLengthCm'].values[knn], X_train['PetalLengthCm'].values[knn], marker="o", s=10,color="black")

    return y_predict


def check_accuracy(y_test, y_pred):
    print(f"\nPredicted values : {y_pred}")
    print(f"Original values : {y_test.values}")
    print(f"Prediction : {y_pred == y_test.values}")
    total = 0
    true = 0
    for i in (y_pred == y_test.values):
        total = total + 1
        if i:
            true = true + 1
    acc = "%.2f" % (true / total * 100)
    print(f"Accuracy : {acc}%")


data = import_data()
print_data_sample(data)
plot_data(data, 'SepalLengthCm', 'PetalLengthCm', 'Species')
plt.show()
X_train, X_test, y_train, y_test = split_data(data, 30)  # test size = 30
plot_data(data, 'SepalLengthCm', 'PetalLengthCm', 'Species')
prediction = knn(30)  # knn implementation considering 5 nearest neighbours of each of 30 test samples
check_accuracy(y_test, prediction)
plt.show()
