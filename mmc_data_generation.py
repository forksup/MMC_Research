from Models.model_sources.markov_source import MarkovChain
from Models.MMC import MMC
from Models.HMC import HMC
from Models.DBN import DBN
from Models.model_sources.mtd_source import MTD

from Datasets import Blocksworld_Data

#from Datasets.Markov_Data import HMM_Data

from Datasets.MMC_Data import MMC_data
from Datasets.Fruit_Data import fruit_domain
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")


amount_to_average = 3

training_master = []
testing_master = []

state_count = 7
order = 3
sgo_type = "greedy"
methods = [MMC, HMC, DBN, MTD]
types = [m.__name__ for m in methods]
dataset = Blocksworld_Data.blocks

print(f"Dataset: {dataset.__name__}")
for _ in range(amount_to_average):
    if dataset == Blocksworld_Data.blocks:
        (X_train, X_test, y_train, y_test), state_count = dataset.gen_data(state_count, order, 50000, True)  ## Fitting model
    else:
        X_train, X_test, y_train, y_test = dataset.gen_data(state_count, order, 50000, True)  ## Fitting model

    args_training = {"X_train": X_train, "y_train": y_train}
    args_testing = {"X_test": X_test, "y_test": y_test}
    results_training = []
    results_testing = []

    for m in methods:
        model = m(state_count, order=order)
        training = MarkovChain.calculate_time(model.train, args_training)[0]
        testing = MarkovChain.calculate_time(model.test, args_testing)[0]

        print(model.__class__.__name__)
        print(f"Training: {training}")
        print(f"Testing: {testing}")

        #print(m.__name__)
        #print(results_testing[-1],end="\n\n")
    print(results_training)
    print(results_testing)
    training_master.append(results_training)
    testing_master.append(results_testing)


def find_average(arr):
    return sum(arr) / len(arr)



# creating the dataset
def create_bar_graph(data, title):
    courses = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(courses, values, color='maroon',
            width=0.4)

    plt.title(title)
    plt.show()



create_bar_graph({types[i]: acc_training[i] for i in range(len(acc_training))}, "Training Accuracy")
create_bar_graph({types[i]: acc_testing[i] for i in range(len(acc_testing))}, "Testing Accuracy")
create_bar_graph({types[i]: training_times[i] for i in range(len(training_times))}, "Training Times")

#%%
