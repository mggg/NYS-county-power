import pickle

import numpy as np
import torch
from torch import optim
from torch.ao.nn.quantized.functional import threshold

from numpy_power_function import computer_power_numpy as compute_power, computer_power_helper_numpy_relaxed, computer_power_numpy_simple


def matrix_of_minone_one_combinations(dim):
    # return a matrix with 2^dim rows and dim columns, where the rows are all possible {-1, 1} vectors of length dim.
    return np.c_[tuple(i.ravel() for i in np.mgrid[tuple(slice(-1, 2, 2) for _ in range(dim))])].astype(np.float64)


def q(a):
    # for each entry in a, calculate (1 + a**2) / 2 if -1 <= a <= 1 else abs(a).

    within_bounds = (a >= -1) & (a <= 1)
    result_within_bounds = (1 + a**2) / 2
    result_else = torch.abs(a)
    result = torch.where(within_bounds, result_within_bounds, result_else)

    return result


def g(w, n, c):
    xs = torch.from_numpy(matrix_of_minone_one_combinations(n))
    exp = torch.mean(q(torch.mv(xs, w)))
    return exp - torch.dot(w, c)


def ftilde_w0(w, n, c):
    xs = torch.from_numpy(matrix_of_minone_one_combinations(n))
    exp = (torch.mv(torch.transpose(xs, 0, 1), torch.clip(torch.mv(xs, w), min=-1., max=1.))) / 2**(n-1)
    return exp - c


if __name__ == '__main__':
    populations = np.array([6945.,4158,2322,10242,1464,2087,4156,7508,4452,5341,2695,725,765,1583,2292,1157,3187])
    threshold = 1/2
    assert threshold == 1/2, "Optimization currently optimized is only for threshold 0 in Diakonikolas terminology, i.e., 1/2 for us."

    n = len(populations)
    target = torch.tensor(populations / sum(populations))

    with open("NYS_counties_livingston_full.pkl", "rb") as f:
        A_subsets = pickle.load(f)

    subset_masks = np.zeros((n, len(A_subsets)), dtype=np.float64)
    for j, subset in enumerate(A_subsets):
        subset_masks[subset, j] = True
    subset_masks_bool = subset_masks.astype(np.bool_)

    w = torch.tensor(target, requires_grad=True)
    optimizer = optim.SGD([w], lr=0.1)

    i = 0
    while True:
        optimizer.zero_grad()
        res = g(w, n, target)

        res.backward()

        renormalize_weights = (w / w.sum()).detach().numpy()
        power_error = computer_power_numpy_simple(renormalize_weights[:, np.newaxis], subset_masks_bool, threshold) - target.numpy()
        relaxed_error = computer_power_helper_numpy_relaxed(target.numpy(), w.detach().numpy(), subset_masks, 0)
        np.set_printoptions(precision=3)
        if i % 50 == 0:
            print("iteration:", i, "\tg:", float(res), "\tgradient L2:", np.linalg.norm(w.grad.numpy(), ord=2),
                  "\tftilde_w0-c L2", np.linalg.norm(ftilde_w0(w.detach(), n, target).numpy(), ord=2),
                  "\tLBF L2 error:", np.linalg.norm(relaxed_error, ord=2),
                  "\tpower L1 error:", np.linalg.norm(power_error, ord=1), "\tweights:", w.detach().numpy())

        #print(np.linalg.norm(w.grad.numpy(), ord=2))
        optimizer.step()
        i += 1