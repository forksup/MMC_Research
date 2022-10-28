from collections import defaultdict

from Models.HMC import HMC
from Models.model_sources.markov_source import MarkovChain
from Models.MMC import MMC
from Models.DBN import FMC

from Datasets import Blocksworld_Data, Markov_Data_Casual, MMC_Data, watch_and_help

import matplotlib.pyplot as plt
import numpy as np
import warnings

from Models.model_sources.mtd_source import MTD

warnings.filterwarnings("ignore")

amount_to_average = 1

training_master = []
testing_master = []

state_count = 10
order = 3
sgo_type = "greedy"
methods = [FMC, MMC, ]  # HMC,] #MTD]  # FMC]
types = [m.__name__ for m in methods]
dataset = Blocksworld_Data.blocks()

dataset_size = 2500
print(f"Dataset: {dataset.__class__.__name__}")
# upload
action_prediction = True

for _ in range(amount_to_average):
    if dataset == Blocksworld_Data.blocks:
        X_train, X_test, y_train, y_test = dataset.gen_data(state_count, order, dataset_size, False, True,
                                                            True)  ## Fitting model
    else:
        X_train, X_test, y_train, y_test = dataset.gen_data(state_count, order, dataset_size)  ## Fitting model

    print(f"Dataset Size: {len(X_train) + len(y_train)}")

    reverse_states = {v: k for k, v in dataset.state_keys.items()}
    actions = set()
    action_index = 0
    for t in reverse_states:
        actions.add(reverse_states[t][action_index])

    if action_prediction:
        print(f"Actions: {actions}")
        print(f"Action Size: {len(actions)}")

    state_count = len(set(np.unique(X_train)) | set(y_train) | set(np.unique(X_test)) | set(y_test))
    args_training = {"X_train": X_train, "y_train": y_train}
    args_testing = {"X_test": X_test, "y_test": y_test}
    results_training = []
    results_testing = []

    for m in methods:
        model = m(state_count, order=order)
        #print(f"Start training for {model.__class__.__name__}")
        training = MarkovChain.calculate_time(model.train, args_training)

        # Specifically for blocksworld
        if action_prediction:
            state_dict = model.states
            pred_res = []
            act_and_index = [(reverse_states[key][0], key) for key, value in enumerate(state_dict)]

            # drop_indexes = [key for key,value in state_dict.items() if "drop" in value]
            # rise_indexes = [key for key,value in state_dict.items() if "rise" in value]

            for lag, y in zip(X_test, y_test):
                cpt_row = model.return_probs(lag)
                action_dict = defaultdict(float)
                for t in act_and_index:
                    try:
                        action_dict[t[0]] += cpt_row[t[1]]
                    except:
                        ...

                most_likely_action = max(action_dict, key=action_dict.get)
                if most_likely_action == reverse_states[y][action_index]:
                    pred_res.append(1)
        testing = MarkovChain.calculate_time(model.test, args_testing)
        print()
        print(model.__class__.__name__)
        if action_prediction:
            print(f"Action Prediction: {round(sum(pred_res) / len(y_test)*100)}%")
        print(f"Training: {round(training[0]*100,2)+'%' if training[0] else 'Na'} {round(training[1],2)}s")
        print(f"Testing: {round(testing[0]*100,2)}% {round(testing[1],2)}s")
        print("")

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

# %%
