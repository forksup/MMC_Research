
import threading
import numpy as np
from progressbar import progressbar

from Models.model_sources.markov_source import MarkovChain
from Models.MMC import MMC
from Models.HMC import HMC
from Models.DBN import DBN
from datetime import datetime
from collections import defaultdict
from Models.model_sources.mtd_source import MTD
from Datasets import Markov_Data, MMC_Data, Fruit_Data

from Datasets import MMC_Data, Fruit_Data
from scipy import stats


class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)

        super().__init__(group=group, target=function, name=name, daemon=daemon)


import sys
sys.path.append("/mnt/watchandhelp/PycharmProjects/mtd-learn")

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [20, 20]

methods = [MMC, MTD, HMC, DBN]

mr = []
mt = []

metrics = ["Testing Accuracy", "Training Times", " Testing Times"]
types = [m.__name__ for m in methods]

# creating the dataset
def create_bar_graph(data, title):
    courses = data
    values = types

    plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(courses, values, color='maroon',
            width=0.4)

    plt.title(title)
    plt.show()


def train_then_test(model, args_train, args_test):
    train = MarkovChain.calculate_time(model.train, args_train)
    test = MarkovChain.calculate_time(model.test, args_test)
    return train, test, model.name


def find_average(arr):
    return sum(arr) / len(arr)


def plot_data(x, title, metric: str, ax, colors: str):
    for key in data_results:
        for method in data_results[key]:
            y = []
            st_dev = []

            for kk in data_results:
                y.append(data_results[kk][method][metric][0][0])
                st_dev.append(data_results[kk][method][metric][0][1])

            ax.plot(x, y, label=f"{method}", color=colors[types.index(method)])

            ax.fill_between(x, [d - st_dev[i] for i, d in enumerate(y)], [d + st_dev[i] for i, d in enumerate(y)],
                            alpha=.2, edgecolor='#3F7F4C', facecolor=colors[types.index(method)],
                            linewidth=0)
            ax.set_xlabel("State Space Size")

            if "accuracy" in metric:
                ax.set_ylabel("Prediction Accuracy")
            else:
                ax.set_ylabel("Time")

    ax.set_title(title)
    ax.legend()


# data_size_args( initial value, max value, step)
def run_experiment(methods, data_size_args, state_size_args, amount_to_average, data_generator, order, runthreads):

    print("Beginning experiment")
    data_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    total_iterations = len(range(*data_size_args)) * len(range(*state_size_args)) * amount_to_average * len(methods)
    bar = progressbar.ProgressBar(maxval=total_iterations)
    bar.start()
    sgo_type = None
    for state_space_size in range(*state_size_args):
        for data_size in range(*data_size_args):

            d_to_average = [[[] for _ in range(len(metrics))] for _ in range(len(methods))]

            for e in range(amount_to_average):

                (X_train, X_test, y_train, y_test) = data_generator.gen_data(state_space_size, order, data_size)

                args_training = {"X_train": X_train, "y_train": y_train}
                args_testing = {"X_test": X_test, "y_test": y_test}

                threads = []

                for i, m in enumerate(methods):
                    if "MMC" in m.__name__:
                        if not sgo_type:
                            sgo_type = m(state_space_size, order).sgom

                    thread = ThreadWithResult(target=train_then_test,
                                              args=(m(state_space_size, order), args_training, args_testing))
                    threads.append(thread)
                    thread.start()
                    if not runthreads:
                        thread.join()
                    # acc_train, time_train = MarkovChain.calculate_time(model.train, args_training)
                    # acc_test, time_test = MarkovChain.calculate_time(model.test, args_testing)
                for i in range(len(threads)):
                    bar.update(bar.currval + 1)
                    threads[i].join()

                for i in range(len(threads)):
                    # 0 is train
                    # 1 is test
                    # metrics = ["Testing Accuracy", "Training Times"," Testing Times"]
                    mm = [threads[i].result[1][0],
                          threads[i].result[0][1],
                          threads[i].result[1][1]]
                    # mm = [acc_test, time_train, time_test]
                    for b in range(len(metrics)):
                        d_to_average[i][b].append(mm[b])
            for j, r in enumerate(d_to_average):
                for jj in range(len(metrics)):
                    data_results[state_space_size][types[j]][metrics[jj]].append(
                        (find_average(d_to_average[j][jj]), np.std(d_to_average[j][jj])))
    # print("Minutes Taken:")
    # print((datetime.now() - start_time).total_seconds() // 60)
    print("Experiment completed")
    return data_results, sgo_type


#data_generator = Markov_Data.HMM_Data
if __name__ == "__main__":

    data_generator = MMC_Data.MMC_data
    order = 5
    data_size_args = (120000, 120001, 10000)
    state_size_args = (15, 20, 1)
    avg_amt = 10
    threading = False
    data_results, sgo_type = run_experiment(methods, data_size_args, state_size_args, avg_amt, data_generator, order, threading)



"""

for st in range(*state_size_args):
    print("State Space", {st})
    for m1, m2 in list(itertools.permutations(methods, 2)):
        n1 = m1.__name__
        n2 = m2.__name__
        print(f"T-Test for {n1} & {n2}")
        tstat, pval = stats.ttest_ind([s[0] for s in data_results[st][n1]['Testing Accuracy']], [s[0] for s in data_results[st][n2]['Testing Accuracy']])
        print("t-value: ", tstat, " p-value: ", pval)

fig, axs = plt.subplots(len(metrics), len(data_results), figsize=(20, 20))
fig.suptitle(f'Data Type: {data_generator.__name__}', y=1.08)
colors = ["#21d185", "#d1218b", "#0000FF", "#FFA500"]
for plot_index, (state_key, d) in enumerate(data_results.items()):
    for im, met in enumerate(metrics):
        plot_data(list(range(*data_size_args)), d, met, met, axs[im], colors)

fig.tight_layout()
fig.show()

# %lprun -f run_experiment run_experiment()
from matplotlib import pyplot as pl

pl.clf()

x = np.linspace(0, 30, 100)
y = np.sin(x) * 0.5
pl.plot(x, y, '-k')

x = np.linspace(0, 30, 30)
y = np.sin(x / 6 * np.pi)
error = np.random.normal(0.1, 0.02, size=y.shape) + .1
y += np.random.normal(0, 0.1, size=y.shape)

pl.plot(x, y, 'k', color='#CC4F1B')
pl.fill_between(x, y - error, y + error,
                alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

y = np.cos(x / 6 * np.pi)
error = np.random.rand(len(y)) * 0.5
y += np.random.normal(0, 0.1, size=y.shape)
pl.plot(x, y, 'k', color='#1B2ACC')
pl.fill_between(x, y - error, y + error,
                alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
                linewidth=4, linestyle='dashdot', antialiased=True)

y = np.cos(x / 6 * np.pi) + np.sin(x / 3 * np.pi)
error = np.random.rand(len(y)) * 0.5
y += np.random.normal(0, 0.1, size=y.shape)
pl.plot(x, y, 'k', color='#3F7F4C')
pl.fill_between(x, y - error, y + error,
                alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99',
                linewidth=0)

pl.show()
"""
#%%
