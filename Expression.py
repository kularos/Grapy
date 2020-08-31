import numpy as np
import itertools


"""
This File exists to help manage the storage and rendering of mathematical expressions
This feature isn't currently used as much as it ought to; many expressions are currently hardcoded
"""


def sanitize_functions(equation, domain):
    """ Takes the internal list of functions, and evaluates them along the domain.

    Future implementation will also aim to separate the internal function
    list with respect to 'closed' variables, and 'open' ones.

    :return:
    """

    # discretize functions
    disc_funcs = []
    for func in equation:

        # scalar case
        if isinstance(func, (int, float, complex)):
            discrete = np.tile(func, domain.cardinality)

        # un-rendered function
        elif callable(func):
            # apply along the entire axis of positon vectors (axis = 1 by convnetion)
            discrete = np.apply_along_axis(func, 1, domain)

        # iterable (rendered) function assumed otherwise
        else:
            discrete = func

        disc_funcs.append(discrete)

    return np.array(disc_funcs).T


class Vector:
    domain_rank = None
    codomain_rank = None


class Operator:
    input_rank = None
    output_rank = None
    _is_valid = False
    _matrix = None

    def __init__(self, domain=None, codomain=None, eval_args=None, eval_kwargs=None):
        self.domain = domain
        self.codomain = codomain

        self._args = eval_args
        self._kwargs = eval_kwargs

    def evaluate(self, subdomain, *args, **kwargs):
        """ Evaluates *self* over a given *subdomain* ∈ *self.domain*
        Must be overwritten in subclasses
        """
        return NotImplemented

    def _render(self, domain):
        """
        Evaluates *self* at each point of *domain*, allowing vector elements of *domain*
        to be evaluated through matrix multiplication.
        """
        matrix = np.zeros((domain.cardinality, domain.cardinality))

        for i, indices, subdomain in itertools.count, domain.subdomain_indices, domain.subdomais:
            matrix[i, indices] = self.evaluate(subdomain, *self._args, **self._kwargs)

        self._matrix = matrix
        self._is_valid = True

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def __mul__(self, other):
        # operator multiplication, handled on a type-wise basis
        if isinstance(other, Operator):
            return OperatorFactory.operator_multiplication(self, other)

        if isinstance(other, (int, float, complex)):
            return OperatorFactory.scalar_multiplication(self, other)

        if isinstance(other, Vector):
            return VectorFactory.operate(other, self)

    def __rmul__(self, other):
        # arguments are swapped to protect non-commutative multiplication
        if isinstance(other, Operator):
            return OperatorFactory.operator_multiplication(other, self)

        if isinstance(other, (int, float, complex)):
            return OperatorFactory.scalar_multiplication(other, self)

        if isinstance(other, Vector):
            return VectorFactory.operate(other, self)

    def __add__(self, other):
        if isinstance(other, Operator):
            return OperatorFactory.operator_addition(self, other)

        if isinstance(other, (int, float, complex)):
            return OperatorFactory.scalar_multiplication(self, other)

        if isinstance(other, Vector):
            return VectorFactory.operate(other, self)


class DifferentialOperator(Operator):
    pass


class VectorFactory:
    """Factory class for abstract representation of vector transformations"""

    @staticmethod
    def operate(V, A):
        """ Returns the vector result of operation A. V' = AV """


class OperatorFactory:
    """Factory class for creating composite operators"""

    @staticmethod
    def operator_multiplication(A, B):
        """ Returns the composite operator C, where C = A * B"""

    @staticmethod
    def scalar_multiplication(A, n):
        """ Returns the composite operator C, where C = nA, and n is a scalar"""

    @staticmethod
    def operator_addition(A, B):
        """ Returns the composite operator C, where C = A + B"""
