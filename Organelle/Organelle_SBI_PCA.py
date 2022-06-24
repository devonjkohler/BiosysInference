## Imports
import numpy as np
import torch

import pickle

import sbi.utils as utils
from sbi.inference.base import infer
from sklearn.decomposition import PCA

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return torch.tensor(np.pad(array, pad_width=npad, mode='constant', constant_values=0))

def organelle_sim(rates):

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
    path = torch.flatten(path)
    return path


# Gillespie implementation
def organelle_sim_pca(rates):

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
    path = torch.flatten(path)
    path = (path - v0_min) / (v0_max - v0_min)
    path = pca.transform(path.reshape(1, 40000))
    return path

def main():

    ## Define Prior
    prior = utils.BoxUniform(
        low=torch.tensor([.00001, .00001, .00001]),
        high=torch.tensor([.0003, .0003, .0003]))

    ## Generate obs to train PCA
    ## Gen some dataz
    example_sims = [organelle_sim(prior.sample()) for _ in range(2000)]
    example_sims = torch.stack(example_sims)
    with open(r'/scratch/kohler.d/code_output/rate_inference/organelle_sims.pickle', 'wb') as handle:
        pickle.dump(example_sims, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ## Train PCA
    global v0_min
    global v0_max
    v0_min = example_sims.min()
    v0_max = example_sims.max()
    example_sims = (example_sims - v0_min) / (v0_max - v0_min)

    global pca
    pca = PCA(n_components=40)
    pca.fit(example_sims)

    ## Run SBI
    num_sim = 2000
    method = 'SNLE'  # SNPE or SNLE or SNRE
    posterior = infer(
        organelle_sim_pca,
        prior,
        # See glossary for explanation of methods.
        #    SNRE newer than SNLE newer than SNPE.
        method=method,
        num_workers=-1,
        num_simulations=num_sim
    )

    ## test on sim data
    sim_rates = [2.51 * 10 ** -4, 10 * 10 ** -5, 10.5 * 10 ** -5]
    sim_data = [organelle_sim_pca(sim_rates) for _ in range(16)]
    sim_data = np.array([x.squeeze() for x in sim_data])
    samples = posterior.sample((1000,), x=sim_data)

    with open(r'/scratch/kohler.d/code_output/rate_inference/org_pca_SNLE_samples.pickle', 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
