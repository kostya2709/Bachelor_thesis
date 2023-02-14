import os

import matplotlib.pyplot as plt
import numpy as np

from src.utils.definitions import IMG_DIR


def draw(inputs, labels, results):
    left = 0
    right = 15
    points_num = 20
    step = (right - left) / points_num
    x = np.arange(left, right, step)

    input = list(inputs[0, :, 0].detach())
    label = list(labels[0, :, 0].detach())
    result = list(results[0, :, 0].detach())
    label.insert(0, input[-1])
    result.insert(0, input[-1])
    known = len(input)

    smooth_x = np.arange(left, right, step / 2)
    sin_x = np.sin(smooth_x)
    plt.plot(smooth_x, sin_x, label="sin(x)")

    plt.plot(x[:known], input, label="known data")
    plt.plot(x[known - 1 :], label, "r-", label="ground truth")
    plt.plot(x[known - 1 :], result, "g--", label="predicted", linewidth=2)

    plt.title("Predictions")
    plt.xlabel("Time")
    plt.ylabel("Coordinate")
    plt.grid(True, which="both")
    plt.legend()

    img_path = os.path.join(IMG_DIR, "simple", "prediction.png")
    plt.savefig(img_path)
