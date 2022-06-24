
import numpy as np
import pickle

## CNN functions
import torch
from torch.nn import Conv2d, ReLU, Linear, Sequential, MaxPool2d, AvgPool2d, Dropout, Module, CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

## SBI
import sbi.utils as utils

## Define model
def gillespie_simulator(propose_rates):
    t = 0.
    stop_time = 5000.
    s = torch.tensor([20., 40.])
    path = np.insert(s, 0, t, axis=0).reshape(1, 3)

    rate_functions = [lambda s: propose_rates[0] * s[0],
                      lambda s: propose_rates[1] * s[1] * s[0],
                      lambda s: propose_rates[2] * s[1]]
    n_func = len(rate_functions)

    transition_matrix = torch.tensor([[1, 0], [-1, 1], [0, -1]])

    for i in range(5001):

        sampling_weights = [f(s) for f in rate_functions]
        total_weight = sum(sampling_weights)

        probs = np.array([weight / total_weight for weight in sampling_weights])
        sample = np.random.choice(n_func, p=probs)
        t += np.random.exponential(1.0 / total_weight)

        s = s + transition_matrix[sample]
        s = torch.normal(s, .25)
        s[0] = max(1, s[0])
        s[1] = max(1, s[1])
        if i % 5 == 0:
            path = torch.cat((path, np.insert(s, 0, t, axis=0).reshape(1, 3)), axis=0)

    return path[2:].T

def main():

    # Prior used to train nn (need it to span area for inference)
    prior = utils.BoxUniform(
        torch.tensor([0.005, 0.0001, 0.01]),
        torch.tensor([0.02, 0.001, 0.05])
    )

    # # Sample 10000 traces
    obs_list = list()
    labels = list()
    for i in range(20000):
        prior_sample = prior.sample()
        labels.append(prior_sample)
        obs_list.append(gillespie_simulator(prior_sample))

    x = torch.stack(obs_list, axis=0)
    y = torch.stack(labels, axis=0)

    # Save observations
    print("trying to save cnn obs")
    with open(r'/scratch/kohler.d/code_output/biosim/lv_obs_20000.pickle', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("trying to save cnn labels")
    with open(r'/scratch/kohler.d/code_output/biosim/lv_labels_20000.pickle', 'wb') as handle:
        pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved")

if __name__ == '__main__':
    main()