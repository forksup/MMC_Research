import pickle
import threading
import numpy as np
from progressbar import progressbar
import argparse
import pandas as pd
from scipy import stats
from tabulate import tabulate
from Models.model_sources.markov_source import MarkovChain
from Models.MMC import MMC
from Models.HMC import HMC
from Models.DBN import FMC
from collections import defaultdict
from Models.model_sources.mtd_source import MTD
from Datasets import (
    Markov_Data,
    MMC_Data,
    Fruit_Data,
    Markov_Data_Large,
    Financial_Data,
)
import matplotlib.pyplot as plt


class ThreadWithResult(threading.Thread):
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None
    ):
        def function():
            self.result = target(*args, **kwargs)

        super().__init__(group=group, target=function, name=name, daemon=daemon)


mr = []
mt = []

metrics = ["Testing Accuracy", "Training Times", " Testing Times"]


def create_bar_graph(data, title, types):
    courses = data
    values = types

    plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(courses, values, color="maroon", width=0.4)

    plt.title(title)
    plt.show()


def train_then_test(model, args_train, args_test):
    train = MarkovChain.calculate_time(model.train, args_train)
    test = MarkovChain.calculate_time(model.test, args_test)
    return train, test, model.name


def perform_ttest(methods, x, latex=False):
    method_perms = []
    for m in methods:
        if m != MMC:
            method_perms.append((MMC, m))

    t_test = {}

    for m1, m2 in method_perms:
        n1 = m1.__name__
        n2 = m2.__name__
        if n1 == n2:
            continue

        tstat, pval = stats.ttest_rel(x[n1], x[n2])

        t_test[f"{m1.__name__}-{m2.__name__}"] = [pval]

    result_df = pd.DataFrame(t_test)

    return result_df


def dd():
    return defaultdict(dd2)


def dd2():
    return defaultdict(list)


def find_average(arr):
    return sum(arr) / len(arr)


def plot_data(
    x,
    data_results,
    title,
    metric: str,
    ax,
    colors: str,
    types,
    xlabel_size=20,
    ylabel_size=20,
    title_size=20,
):
    for key in data_results:
        for method in data_results[key]:
            y = []
            st_dev = []

            for kk in data_results:
                y.append(data_results[kk][method][metric][0][0])
                st_dev.append(data_results[kk][method][metric][0][1])

            ax.plot(x, y, label=f"{method}", color=colors[types.index(method)])

            ax.fill_between(
                x,
                [d - st_dev[i] for i, d in enumerate(y)],
                [d + st_dev[i] for i, d in enumerate(y)],
                alpha=0.2,
                edgecolor="#3F7F4C",
                facecolor=colors[types.index(method)],
                linewidth=0,
            )

            if "Accuracy" in metric:
                ax.set_ylabel("Prediction Accuracy", fontsize=xlabel_size)
            else:
                ax.set_ylabel("Time (s)", fontsize=ylabel_size)
        break

    ax.set_title(title, fontsize=title_size)
    ax.legend()


# This function will accept a metric and
def run_experiment(
    methods,
    amount_to_average,
    data_generator,
    runthreads,
    m_to_test,
    data_size_args=None,
    state_size_args=None,
    order_size_args=None,
    save_path="/storage/data/experiment_results.pkl",
):
    types = [m.__name__ for m in methods]
    data_results = defaultdict(dd)
    total_iterations = amount_to_average * len(methods)
    sgo_type = None

    if m_to_test == "order":
        total_iterations *= len(range(*order_size_args))
        range_args = order_size_args
    elif m_to_test == "state_space":
        total_iterations *= len(range(*state_size_args))
        range_args = state_size_args
    elif m_to_test == "data_size":
        total_iterations *= len(range(*data_size_args))
        range_args = data_size_args

    bar = progressbar.ProgressBar(maxval=total_iterations)
    bar.start()

    for r_arg in range(*range_args):
        d_to_average = [[[] for _ in range(len(metrics))] for _ in range(len(methods))]

        for e in range(amount_to_average):
            if m_to_test == "data_size":
                data_size = r_arg
                state_space_size = state_size_args
                order = order_size_args

            elif m_to_test == "state_space":
                data_size = data_size_args
                state_space_size = r_arg
                order = order_size_args
            else:
                order = r_arg
                data_size = data_size_args
                state_space_size = state_size_args

            (X_train, X_test, y_train, y_test) = data_generator.gen_data(
                state_space_size, order, data_size
            )

            args_training = {"X_train": X_train, "y_train": y_train}
            args_testing = {"X_test": X_test, "y_test": y_test}

            threads = []

            for i, m in enumerate(methods):
                if "MMC" in m.__name__:
                    if not sgo_type:
                        sgo_type = m(state_space_size, order).sgom

                thread = ThreadWithResult(
                    target=train_then_test,
                    args=(m(state_space_size, order), args_training, args_testing),
                )
                threads.append(thread)
                thread.start()

                if not runthreads:
                    thread.join(timeout=999999)

            for i in range(len(threads)):
                bar.update(bar.currval + 1)
                threads[i].join(timeout=999999)

            for i in range(len(threads)):
                # 0 is train
                # 1 is test
                mm = [
                    threads[i].result[1][0],
                    threads[i].result[0][1],
                    threads[i].result[1][1],
                ]

                for b in range(len(metrics)):
                    if isinstance(mm[b], list):
                        d_to_average[i][b].append(mm[b][0])
                    else:
                        d_to_average[i][b].append(mm[b])
    
        for j, r in enumerate(d_to_average):
            for jj in range(len(metrics)):
                data_results[r_arg][types[j]][metrics[jj]].append(
                    (d_to_average[j][jj], np.std(d_to_average[j][jj]))
                )

        with open(save_path, "wb") as f:
            pickle.dump(data_results, f)

    print("Experiment completed")
    return data_results, sgo_type


def load_and_plot(metric_to_test, metrics, storage_path):
    with open(storage_path, "rb") as f:
        res = pickle.load(f)

        fig, axs = plt.subplots(len(metrics), 1, figsize=(20, 20))
        colors = ["#21d185", "#d1218b", "#0000FF", "#FFA500"]
        for im, met in enumerate(metrics):
            plot_data(list(res.keys()), res, met, met, axs[im], colors, types)

        fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    methods = [HMC, FMC, MMC, MTD]
    types = [m.__name__ for m in methods]

    # type_dict = {m.__name__: method for m in methods}

    parser.add_argument("--methods", type=list)

    parser.add_argument("--all", action=argparse.BooleanOptionalAction)

    parser.add_argument("--meth", choices=types, help="M")

    # Select each model to be tested

    # Supply the data generator reference. The Gen_data function will be used
    data_generator = Financial_Data.financial_data
    data_size_args = 120000
    state_size_args = 1000
    order_size_args = (2, 5, 1)
    avg_amt = 1
    threading = True

    metric_to_test = "order"

    # metrics we can test
    # order, state_space, or data_size

    data_results, sgo_type = run_experiment(
        methods,
        avg_amt,
        data_generator,
        threading,
        metric_to_test,
        data_size_args,
        state_size_args,
        order_size_args,
        save_path="/home/mitch/DataspellProjects/thesis_research/storage.pkl",
    )
