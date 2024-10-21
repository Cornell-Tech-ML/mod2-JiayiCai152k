import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generates a list of 2D points with random coordinates.

    Args:
    ----
        N (int): The number of data points to generate.

    Returns:
    -------
        List[Tuple[float, float]]: A list of tuples, each containing two floats.
        Each tuple represents the x and y coordinates of a point.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generates a simple dataset where the classification is based on whether
    the first coordinate (x_1) of the point is less than 0.5.

    Args:
    ----
        N (int): The number of data points to generate.

    Returns:
    -------
        Graph: A Graph dataclass containing the number of points (N),
               the list of points (X), and the corresponding labels (y).

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generates a diagonal split dataset. Points below the line x_1 + x_2 = 0.5
    are labeled as 1, otherwise as 0.

    Args:
    ----
        N (int): The number of data points to generate.

    Returns:
    -------
        Graph: A Graph dataclass with the points and their binary labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates a dataset where points are classified based on x_1 values.
    Points with x_1 less than 0.2 or greater than 0.8 are labeled as 1.

    Args:
    ----
        N (int): The number of data points to generate.

    Returns:
    -------
        Graph: A Graph dataclass containing the points and their binary labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates an XOR dataset where points are classified based on an XOR logic.
    Points in opposite quadrants relative to x_1 = 0.5 and x_2 = 0.5 are labeled as 1.

    Args:
    ----
        N (int): The number of data points to generate.

    Returns:
    -------
        Graph: A Graph dataclass with the points and corresponding labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generates a circular dataset. Points inside a circle centered at (0.5, 0.5)
    with a radius that corresponds to x1^2 + x2^2 > 0.1 are labeled as 1.

    Args:
    ----
        N (int): The number of data points to generate.

    Returns:
    -------
        Graph: A Graph dataclass containing the points and their binary labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates a spiral dataset consisting of two interleaving spirals, one for each class.

    Args:
    ----
        N (int): The number of data points to generate, divided equally between the two spirals.

    Returns:
    -------
        Graph: A Graph dataclass with the coordinates of points and class labels.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
