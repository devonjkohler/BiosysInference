## Imports
import numpy as np
import torch

import pickle
import sbi.utils as utils

# Gillespie implementation
def organelle_sim(rates):
    def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:

        pad_size = target_length - array.shape[axis]

        if pad_size <= 0:
            return array

        npad = [(0, 0)] * array.ndim
        npad[axis] = (0, pad_size)

        return torch.tensor(np.pad(array, pad_width=npad, mode='constant', constant_values=0))

    t = 0.
    stop_time = 42000.
    s = np.random.uniform(100, 600)
    path = torch.tensor([s, t]).reshape(1, 2)

    rate_functions = [lambda s: rates[0],
                      lambda s: rates[1] * s,
                      lambda s: rates[2] * s]
    n_func = len(rate_functions)

    transition_matrix = torch.tensor([[1], [1], [-1]])
    eject = True

    while (t < stop_time) & eject:

        sampling_weights = [f(s) for f in rate_functions]
        total_weight = sum(sampling_weights)

        probs = np.array([weight / total_weight for weight in sampling_weights], dtype='float64')

        ## Fix dumb rounding issues
        remaining = 1 - sum(probs)
        probs[2] = probs[2] + remaining

        sample = np.random.choice(n_func, p=probs)
        t += np.random.exponential(1.0 / total_weight)

        s = s + transition_matrix[sample][0]
        s = torch.tensor(np.random.normal(s, 2))
        s = max(1, s)
        # if i % 5 == 0:
        path = torch.cat((path, torch.tensor([s, t]).reshape(1, 2)), axis=0)

        if len(path) == 20000:
            eject = False

    ## Flatten obs
    path = pad_along_axis(path, 20000, axis=0)
    # path = torch.flatten(path)
    return path

def main():
    prior = utils.BoxUniform(
        low=torch.tensor([.00001, .00001, .00001]),
        high=torch.tensor([.0003, .0003, .0003]))

    # # Sample 10000 traces
    obs_list = list()
    labels = list()
    for i in range(100000):
        prior_sample = prior.sample()
        labels.append(prior_sample)
        obs_list.append(organelle_sim(prior_sample))

    x = torch.stack(obs_list, axis=0)
    y = torch.stack(labels, axis=0)

    # Save observations
    print("trying to save obs")
    with open(r'/scratch/kohler.d/code_output/biosim/organelle_obs.pickle', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("trying to save cnn labels")
    with open(r'/scratch/kohler.d/code_output/biosim/organelle_labels.pickle', 'wb') as handle:
        pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved")

if __name__ == '__main__':
    main()