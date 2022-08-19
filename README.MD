## MMC Implementation

This repository contains the implementation of the Max Markov Model as described in the MMC paper. 
There are two class types implemented, data types and models in /Datasets and /Models respectively. 
Datasets contain various methods of testing data, where state sequences of order O are generated.

### Running Experiments

The run_experiments method of run_experiments.py has been used to generate all data.

```python
def run_experiment(methods, amount_to_average, data_generator, runthreads, m_to_test, data_size_args=None,
state_size_args=None, order_size_args=None, save_path="/storage/data/experiment_results.pkl"):
```

An array of methods includes the various **models** to be ran.
The data generator is the data source as shown below to be used to generate a dataset.
runthreads is a boolean value used to enable threads, otherwise it will be ran sequentially. m_to_test can be eitehr, 'order', 'state_space', 'data_size'. m_to_test represents which metric to vary while keeping two other args static. For example, we may choose to give order_size_args a range of values (1,10,1), while keeping the state_size_args and data_size_args static at 15 and 20,000. One must be a range and the other two may be static values.




Models included in repository:

| Models                                        |                 Description                 |
|-----------------------------------------------|:-------------------------------------------:|
| MMC                                           |              Max Markov Chain               |
| [MTD](https://github.com/PiotrekGa/mtd-learn) | Generalized Mixture Transition Distribution |
| FMC                                           |          First Order Markov Chain           | 
| HMC                                           |           High Order Markov Chain           |

Data Sources included in repository:

| Methods     |                                   Description                                   |
|-------------|:-------------------------------------------------------------------------------:|
| MMC         |                   Data generated following the MMC assumption                   |
| HMC         |                     Data generated from a random CPT table                      |
| Causal HMC  | HMC however the appearance of a lag increases the probability of the same state | 
| Large HMC   |          High probability HMC, one state is at least double all others          |
| Fruit       |    Theoretical testing domain where states are connected with varying order     |
| Blocksworld |      IPC domain of either 4 or 6 blocks depending on configuration options      |

