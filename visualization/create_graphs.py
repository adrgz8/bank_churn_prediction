import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from src.cleaning import clean_data


def plot_graphs(path):
    df = pd.read_csv(path)
    _, axes = plt.subplots(2, 3, figsize=(15, 5))
    sns.countplot(data=df, x='Gender', hue='Exited', ax=axes[0, 0])
    sns.countplot(data=df, x='Geography', hue='Exited', ax=axes[0, 1])
    sns.countplot(data=df, x='HasCrCard', hue='Exited', ax=axes[0, 2])
    sns.countplot(data=df, x='Tenure', hue='Exited', ax=axes[1, 0])
    sns.countplot(data=df, x='IsActiveMember', hue='Exited', ax=axes[1, 1])
    sns.countplot(data=df, x='NumOfProducts', hue='Exited', ax=axes[1, 2])
    plt.tight_layout()
    plt.show()

    _, axes = plt.subplots(1, 4, figsize=(16, 4))
    sns.kdeplot(
        data=df, x='Age', hue='Exited', bw_adjust=1, ax=axes[0])
    sns.kdeplot(
        data=df, x='EstimatedSalary', hue='Exited', bw_adjust=1, ax=axes[1])
    sns.kdeplot(
        data=df, x='Balance', hue='Exited', bw_adjust=1, ax=axes[2])
    sns.kdeplot(
        data=df, x='CreditScore', hue='Exited', bw_adjust=1, ax=axes[3])
    plt.tight_layout()
    plt.show()

    sns.violinplot(
        data=df, x='NumOfProducts', y='Age', hue='Exited', dodge=True)
    plt.show()

    sns.violinplot(
        data=df, x='NumOfProducts', y='Balance', hue='Exited', dodge=True)
    plt.show()

    sns.countplot(data=df, x='Exited')
    plt.show()


if __name__ == '__main__':
    path = 'data/raw/train.csv'
    plot_graphs(path)
