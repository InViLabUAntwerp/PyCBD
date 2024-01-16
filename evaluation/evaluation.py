import numpy as np


def evaluate_corners(gt, pred):
    error = []
    correct_corners = 0
    for corner in gt:
        # Find the nearest corner
        min_idx = np.argmin(np.abs(pred[:, 0] - corner[0]) + np.abs(pred[:, 1] - corner[1]))
        # Euclidian distance
        calc_euc = np.sqrt((corner[0] - pred[min_idx][0]) ** 2 + (corner[1] - pred[min_idx][1]) ** 2)
        if calc_euc < 3:
            error.append(calc_euc)
            correct_corners += 1
    euc_distance_avg = np.mean(np.array(error))
    return euc_distance_avg, correct_corners
