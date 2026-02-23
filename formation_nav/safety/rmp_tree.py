"""
RMPflow tree structure: RMPRoot / RMPNode / RMPLeaf.
Ported from algorithms/multi-robot-rmpflow/rmp.py (Anqi Li, 2019).
"""

import numpy as np


class RMPNode:
    """A generic RMP node in the tree."""

    def __init__(self, name, parent, psi, J, J_dot, verbose=False):
        self.name = name
        self.parent = parent
        self.children = []

        if self.parent:
            self.parent.add_child(self)

        self.psi = psi
        self.J = J
        self.J_dot = J_dot

        # State
        self.x = None
        self.x_dot = None

        # RMP
        self.f = None
        self.a = None
        self.M = None

        self.verbose = verbose

    def add_child(self, child):
        self.children.append(child)

    def pushforward(self):
        if self.verbose:
            print(f'{self.name}: pushforward')

        if self.psi is not None and self.J is not None:
            self.x = self.psi(self.parent.x)
            self.x_dot = np.dot(self.J(self.parent.x), self.parent.x_dot)
            assert self.x.ndim == 2 and self.x_dot.ndim == 2

        for child in self.children:
            child.pushforward()

    def pullback(self):
        for child in self.children:
            child.pullback()

        if self.verbose:
            print(f'{self.name}: pullback')

        f = np.zeros_like(self.x, dtype='float64')
        M = np.zeros((max(self.x.shape), max(self.x.shape)), dtype='float64')

        for child in self.children:
            J_child = child.J(self.x)
            J_dot_child = child.J_dot(self.x, self.x_dot)
            assert J_child.ndim == 2 and J_dot_child.ndim == 2

            if child.f is not None and child.M is not None:
                f += np.dot(J_child.T,
                            child.f - np.dot(np.dot(child.M, J_dot_child), self.x_dot))
                M += np.dot(np.dot(J_child.T, child.M), J_child)

        self.f = f
        self.M = M


class RMPRoot(RMPNode):
    """Root node of the RMP tree."""

    def __init__(self, name):
        super().__init__(name, None, None, None, None)

    def set_root_state(self, x, x_dot):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x_dot.ndim == 1:
            x_dot = x_dot.reshape(-1, 1)
        self.x = x
        self.x_dot = x_dot

    def pushforward(self):
        if self.verbose:
            print(f'{self.name}: pushforward')
        for child in self.children:
            child.pushforward()

    def resolve(self):
        if self.verbose:
            print(f'{self.name}: resolve')
        self.a = np.dot(np.linalg.pinv(self.M), self.f)
        return self.a

    def solve(self, x, x_dot):
        """Given root state, solve for accelerations."""
        self.set_root_state(x, x_dot)
        self.pushforward()
        self.pullback()
        return self.resolve()


class RMPLeaf(RMPNode):
    """Leaf node that evaluates an RMP function."""

    def __init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func):
        super().__init__(name, parent, psi, J, J_dot)
        self.RMP_func = RMP_func
        self.parent_param = parent_param

    def eval_leaf(self):
        self.f, self.M = self.RMP_func(self.x, self.x_dot)

    def pullback(self):
        if self.verbose:
            print(f'{self.name}: pullback')
        self.eval_leaf()

    def add_child(self, child):
        pass  # Leaves cannot have children

    def update_params(self):
        pass

    def update(self):
        self.update_params()
        self.pushforward()
