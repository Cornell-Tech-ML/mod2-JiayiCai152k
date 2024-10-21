"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(a: float, b: float) -> float:
    """Multiplies two numbers"""
    return a * b


def id(a: float) -> float:
    """Returns the input unchanged"""
    return a


def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b


def neg(a: float) -> float:
    """Negates a number."""
    return -a


def lt(a: float, b: float) -> float:
    """Checks if one number is less than another."""
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Checks if two numbers are equal."""
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Returns the larger of two numbers."""
    return a if a > b else b


def is_close(a: float, b: float, tol: float = 1e-2) -> float:
    """Checks if two numbers are close in value."""
    return abs(a - b) < tol


def sigmoid(a: float) -> float:
    """Calculates the sigmoid function."""
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    else:
        exp_x = math.exp(a)
        return exp_x / (1.0 + exp_x)


def relu(a: float) -> float:
    """Applies the ReLU activation function."""
    return a if a > 0 else 0.0


EPS = 1e-6


def log(a: float) -> float:
    """Calculates the natural logarithm."""
    # if a <= 0:
    #    raise ValueError("log undefined for non-positive values")
    return math.log(a + EPS)


def exp(a: float) -> float:
    """Calculates the exponential function."""
    return math.exp(a)


def inv(a: float) -> float:
    """Calculates the reciprocal"""
    # if a == 0:
    #    raise ZeroDivisionError("Cannot calculate reciprocal of zero")
    return 1 / a


def log_back(a: float, b: float) -> float:
    """Computes the derivative of log times a second argument."""
    # if a <= 0:
    #    raise ValueError("log undefined for non-positive values")
    return b / (a + EPS)


def inv_back(a: float, b: float) -> float:
    """Computes the derivative of reciprocal times a second argument."""
    # if a == 0:
    #    raise ZeroDivisionError("Cannot calculate derivative of reciprocal for zero.")
    # return -b / (a**2)
    return -(1.0 / a**2) * b


def relu_back(a: float, b: float) -> float:
    """Computes the derivative of ReLU times a second argument."""
    return b if a > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a given function to each element of an iterable and returns a new iterable with the results.

    Args:
    ----
    fn(Callable[[float], float]): The function to apply to each element. This function should take a float and return a float.

    Returns:
    -------
    Callable[[Iterable[float]], Iterable[float]]: A function that takes an iterable of floats and returns a new iterable of floats, where each element is the result of applying `fn` to the corresponding element of the input iterable.

    """

    def apply(it: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in it:
            ret.append(fn(x))
        return ret

    return apply


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Combines elements from two iterables using a given function and returns a new iterable with the results.

    Args:
    ----
        fn(Callable[[float, float], float]): The function to apply to pairs of elements. This function should take two floats and return a single float.

    Returns:
    -------
        Callable[[Iterable[float], Iterable[float]], Iterable[float]]: A function that takes two iterables of floats and returns a new iterable of floats, where each element is the result of applying `fn` to the corresponding elements of the input iterables.

    """

    def apply(xls: Iterable[float], yls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(xls, yls):
            ret.append(fn(x, y))
        return ret

    return apply


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduces an iterable to a single value using a given function.

    Args:
    ----
        fn(Callable[[float, float], float]): The function to apply to pairs of elements. It should take two floats and return a single float, combining them in some way.
        start (float): The initial value to start the reduction process. This value is used
            as the initial accumulator value in the reduction.

    Returns:
    -------
        Callable[[Iterable[float]], float]: A function that takes a list of floats and returns a single float, which is the result of repeatedly applying `fn` to reduce the list to a single value.

    """

    def apply(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return apply


def addLists(list1: Iterable[float], list2: Iterable[float]) -> Iterable[float]:
    """Adds two lists element-wise.

    Args:
    ----
        list1: The first list of numbers.
        list2: The second list of numbers.

    Returns:
    -------
        A list containing the element-wise sum of list1 and list2.

    """
    addmylists = zipWith(add)
    return addmylists(list1, list2)


def negList(numbers: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map

    Args:
    ----
        numbers: A list of numbers.

    Returns:
    -------
        A list containing the negated values of the original list elements.

    """
    negate = map(neg)
    return negate(numbers)


def sum(numbers: Iterable[float]) -> float:
    """Sum all elements in a list using reduce

    Args:
    ----
        numbers: A list of numbers.

    Returns:
    -------
        The reduced number

    """
    sumbyreduce = reduce(add, 0.0)
    return sumbyreduce(numbers)


def prod(numbers: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce

    Args:
    ----
        numbers: A list of numbers.

    Returns:
    -------
        The product

    """
    prodbyreduce = reduce(mul, 1.0)
    return prodbyreduce(numbers)
