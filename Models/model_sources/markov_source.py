"""
High order representation for Markov Chains
Copyright (C) 2017 - Pietro Mascolo
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: Pietro Mascolo
Email: iz4vve@gmail.com
"""
# pyxlint: disable=E1101
import collections
import itertools
import numpy as np
import pandas as pd
from scipy import sparse
import datetime
from sklearn import preprocessing


class MarkovChain(object):
    """
    High order Markov chain representation of sequences of states.

    The class is designed to work with a numeric list of state IDs
    in the range [0; number of states - 1].
    If your state have different names, please generate a sorted
    map to a range(number_of_states).
    """

    def __init__(self, n_states, order=1, verbose=False):
        """
        :param n_states: number of possible states
        :param order: order of the Markov model
        """
        self.number_of_states = n_states
        self.order = order
        self.verbose = verbose
        self.gap = 0
        # creates a map of state and label
        # (necessary to recover state label in High Order MC)

        self.possible_states = {
            self.convert_state(j): i for i, j in
            enumerate(itertools.product(range(n_states), repeat=order))
        }

        # allocate transition matrix
        self.transition_matrix = sparse.dok_matrix((
            (len(self.possible_states), self.order)
        ), dtype=np.float64)

    @staticmethod
    def calculate_time(function, args):
        time = datetime.datetime.now()
        res = function(**args)
        duration = (datetime.datetime.now() - time).total_seconds()
        return (res, duration)

    def normalize_transitions(self):
        """
        Normalizes the transition matrix by row
        """
        self.transition_matrix = preprocessing.normalize(
            self.transition_matrix, norm="l1"
        )

    @staticmethod
    def convert_state(s):
        return str(''.join(map(str, s)))


    def _update_transition_matrix(self, states_sequence, normalize=True):
        """
        Updates transition matrix with a single sequence of states
        :param states_sequence: sequence of state IDs
        :type states_sequence: iterable(int)
        :param normalize: whether the transition matrix is normalized after the
           update (set to False and manually triggered when
           training multiple sequences)
        """

        # i is the state 1,2,3,4,5,6
        # convert the state into a row using the possible state dict
        # then convert the actual state to the column number

        funct = self.convert_state


        #if isinstance(states_sequence[0][0], int) or isinstance(states_sequence[0][0], np.int64):
            #funct = lambda t: self.convert_state([t])

        for x, y in zip(states_sequence[0], states_sequence[1]):
            self.transition_matrix[
                self.possible_states[funct(x)],
                y
            ] += 1
        if normalize:
            self.normalize_transitions()

    def fit(self, train_data, normalize=True):
        """
        Fits the model with many sequences of states
        :param state_sequences: iterable of state sequences
        """
        """
        try:
            for index, sequence in enumerate(state_sequences):
                if self.verbose and not index % 10000:
                    print(f"{index} sequences processed")
                self._update_transition_matrix(sequence, gap, normalize=False)
        except TypeError:  # not a list of sequences
            self._update_transition_matrix(state_sequences, gap)
        finally:
            self._normalize_transitions()
        """
        self._update_transition_matrix(train_data, normalize)

    def transition_df(self):
        """
        This returns the transition matrix in form of a pandas dataframe.
        The results are not stored in the model to avoid redundancy.

        Example:
                 A,A     A,B     A,C     ...
            A,A  1       0       0       ...
            A,B  0.33    0.33    0.33    ...
            A,C  0.66    0       0.33    ...
            B,A  0       0       0       ...
            B,B  0       0.5     0.5     ...
            B,C  0.33    0       0.66    ...
            C,A  1       0       0       ...
            C,B  0       1       0       ...
            C,C  0       0       1       ...


        :return: Transition states data frame
        """
        sdf = pd.DataFrame(self.transition_matrix.toarray())

        sdf.index = sorted(self.possible_states)
        sdf.columns = sorted(range(self.number_of_states))

        return sdf.fillna(0)

    def predict_state(self, current_state, num_steps=1):
        """
        :param current_state: array representing current state
        :param num_steps: number of steps for which a prediction is made
        :return: evolved state arrays
        """
        _next_state = sparse.csr_matrix(current_state).dot(
            np.power(self.transition_matrix, num_steps)
        )

        return _next_state[0]

    def possible_states_lookup(self):
        """
        Reverses keys and values of self.possible_states
        (for lookup in transition_matrix)
        """
        return {v: k for k, v in self.possible_states.items()}

    def evolve_states(self, initial_state, num_steps=1, threshold=0.1):
        """
        Evolves the states for num_steps iterations and returns
        a mapping of initial, final and intermediate states.

        :param initial_state: Initial state for the evolution
        :param num_steps: number of iterations
        :param threshold: minimum probability for a state to be considered

        :rtype: defaultdict(list)
        """
        state_id = 0
        state_vector = collections.defaultdict(list)
        # TODO - change all labels to module level constants
        for step in range(num_steps + 1):
            # initial step
            if not state_vector:
                start = initial_state.nonzero()
                for i in start[0]:
                    state_repr = np.zeros(self.transition_matrix.shape[0])
                    state_repr[i] = 1
                    # metadata needed for the representation
                    state_vector[step] += [
                        {
                            "state_id": state_id,
                            "state": i,
                            "weight": initial_state[i],
                            "prev_state": None,
                            "state_repr": state_repr,
                            "actual": initial_state[i]
                        }
                    ]
                continue

            # get last state
            last_states = state_vector.get(step - 1)

            for _state in last_states:
                prediction = self.predict_state(_state.get("state_repr"))

                _, predicted_states = prediction.nonzero()

                for predicted_state in sorted(predicted_states):
                    state_id += 1
                    state_repr = np.zeros(self.transition_matrix.shape[0])
                    state_repr[predicted_state] = 1

                    if prediction[
                        0, predicted_state
                    ] * _state.get("actual") > threshold:
                        state_vector[step] += [
                            {
                                "state_id": state_id,
                                "state": predicted_state,
                                "weight": prediction[0, predicted_state],
                                "prev_state": _state.get("state_id"),
                                "state_repr": state_repr,
                                "actual": prediction[
                                              0, predicted_state
                                          ] * _state.get("actual")
                            }
                        ]

        return state_vector

    @staticmethod
    def build_pos(states):
        """
        build_pos generates a dictionary of positions for nodes

        Used within generate_graph.
        """
        pos = dict()
        for key, state in states.items():
            for n, _state in enumerate(state):
                pos[_state["state_id"]] = (key, -n)
        return pos

# %%
