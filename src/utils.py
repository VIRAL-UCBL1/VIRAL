import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_sumrwdperepi(sum_rewards: list):
    "trace courbe de somme des rec par episodes"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(sum_rewards)), sum_rewards)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()


def plot_sumrwdperepi_movingavg(sum_rewards: list, avgs: list):
    "trace courbe de somme des rec (sum_rewards) et moyenne glissante (avgs) par episodes"
    print("sum_rwd:", type(sum_rewards))
    print("avgs:", type(avgs))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(sum_rewards)), sum_rewards, label="sum_rwd")
    plt.plot(np.arange(len(avgs)), avgs, c="r", label="average")
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.legend(loc="upper left")
    plt.show()


def plot_sumrwdperepi_overseed2(rewards_over_seeds: list):
    """
    trace courbe de somme des rec par episodes moyenne + std sur plusieurs seeds

    """
    # rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
    df1 = pd.DataFrame(rewards_over_seeds).melt()
    df1.rename(columns={"variable": "episodes", "value": "rwd"}, inplace=True)
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="rwd", data=df1, estimator=np.mean, errorbar="sd").set(
        title=""
    )
    plt.show()
