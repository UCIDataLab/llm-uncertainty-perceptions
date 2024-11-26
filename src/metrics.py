import numpy as np

# -------------------------
# add cpa.py dependency
# -------------------------
import sys, pathlib, os

def absolute_error(dist1: dict, dist2: dict):
    assert dist1.keys() == dist2.keys()
    error = []
    for key in dist1:
        error.append(np.abs(dist1[key] - dist2[key]))
    return np.cumsum(error)[-1], error


def ranking__inversion_stats_no_ties(ref_order: list, eval_dist: dict):
    pass  



def get_statistics_per_distribution(histogram: dict):
    maxs, mins, means, medians, uniques = {}, {}, {}, {}, {}
    for uncertainty, distribution in histogram.items():
        numbers, prob = zip(*distribution.items())
    
        for ix, cdf in enumerate(np.cumsum(prob)):
            if cdf >= 0.5 and uncertainty not in medians:
                medians[uncertainty] = numbers[ix]

        
        mean = 0
        for num, p in distribution.items():
            mean += float(num) * p
        means[uncertainty] = mean

        max_id = np.argmax(prob)
        maxs[uncertainty] = numbers[max_id]
        
        min_id = np.argmax(prob)
        mins[uncertainty] = numbers[min_id]
        
        uniques[uncertainty] = len(np.unique(prob))
    
    return {
        "mean": means,
        "median": medians,
        "max": maxs,
        "min": mins,
        "num_unique": uniques,
    }