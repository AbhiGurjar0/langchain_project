import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic.csv")


def get_dataframe():
    return df

def plot_age_histogram():
    plt.figure()
    sns.histplot(df['Age'].dropna(), bins=20)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.savefig("age_hist.png")
    return "age_hist.png"