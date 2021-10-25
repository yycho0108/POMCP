#!/usr/bin/env python3

from typing import List
import logging
import graph_tool
from graph_tool import Graph, Vertex
from graph_tool.util import find_vertex
import numpy as np

from dataclasses import dataclass
from simple_parsing import Serializable


def ucb(N: int, n: int, value: int, c: float = 1):
    return value + c * np.sqrt(np.log(N) / n)


class POMCP:
    @dataclass
    class Config(Serializable):
        gamma: float = 0.5
        c: float = 1.0  # UCB parameter
        threshold: float = 0.005  # Discount threshold
        max_iter: int = int(2**13)  # Number of runs from node.
        num_particle: int = 1024  # Number of particles.

        def __post_init__(self):
            if self.gamma >= 1:
                raise ValueError('Gamma should be less than 1.')

    def __init__(self, cfg: Config, generate):
        self._cfg = cfg
        self._generate = generate
        self._tree = Graph(directed=True)  # tree
        self._prop = {}  # propety map
        self._init_tree()

    def _init_tree(self):
        self._prop['value'] = self._tree.new_vertex_property('double')
        self._prop['N'] = self._tree.new_vertex_property('int')
        self._prop['is_action'] = self._tree.new_vertex_property('bool')

        # NOTE(ycho): I gues the only reason that belief is a vector<int>
        # is because it's currently just a set of integer-valued states.
        # It could/should probably be a much more complex thing, e.g. an actual
        # distribution.
        self._prop['belief'] = self._tree.new_vertex_property('vector<int>')
        self._prop['belief_index'] = self._tree.new_vertex_property('int')
        self._prop['action'] = self._tree.new_edge_property('int')

        # Create the root node.
        root = self._add_node(False)
        self._root = root

    def _add_node(self, is_action: bool, value: float = 0.0,
                  N: int = 0, belief=None, belief_index: int = 0) -> Vertex:
        v = self._tree.add_vertex()

        self._prop['value'][v] = value
        self._prop['N'][v] = N
        self._prop['is_action'][v] = is_action

        if not is_action:
            self._prop['belief'][v] = []
            self._prop['belief_index'][v] = 0
        logging.debug(F'added node {v} action={is_action}')
        return v

    def initialize(self, states, actions, observations):
        self._states = states
        self._actions = actions
        self._observations = observations

    def search_best_action(
            self, node: int, explore: bool = True):
        """Given the current node, try to give the best action."""

        # NOTE(ycho): Let's just accept either option.
        # TODO(ycho): consider `singledispatch`.
        if isinstance(node, int):
            v = self._tree.vertex(node)
        else:
            v = node

        # NOTE(ycho): In POMCP, observation<->action nodes alternate.
        # This means we can't get an action node from an action node.
        if self._prop['is_action'][v]:
            raise ValueError(
                'Cannot get action after action node!')

        if explore:
            # Use UCB / other acquisition function,
            # instead of raw values.
            max_value = None
            result = None
            result_a = None

            for e in v.out_edges():
                action = self._prop['action'][e]
                child = e.target()
                if self._prop['N'][child] == 0:
                    logging.debug('exploring child with no visits!')
                    return (action, child)

                # Compute utility via UCB
                value = ucb(self._prop['N'][v], self._prop['N'][child],
                            self._prop['value'][child],
                            self._cfg.c)
                logging.debug(F'action value = {action}:{value}')

                # argmax
                if max_value is None or max_value < value:
                    max_value = value
                    result = child
                    result_a = action
            if max_value is None:
                raise ValueError(F'No outgoing action from {v}!!')
            return result_a, result
        else:
            # TODO(ycho): unify this block with the above code.
            max_value = None
            result = None
            result_a = None  # ?
            for e in v.out_edges():
                action = self._prop['action'][e]
                child = e.target()
                value = self._prop['value'][child]
                if max_value is None or max_value < value:
                    max_value = value
                    result = child
                    result_a = action

            return result_a, result

    def rollout(self, state, node: int, depth: int) -> float:
        if (self._cfg.gamma ** depth < self._cfg.threshold):
            return 0.0

        # (1) sample action(node) from rollout policy
        # FIXME(ycho): hardcoded uniform-random policy
        a = np.random.choice(self._actions)

        # (2) Generate next state from current action; repeat rollout.
        s1, _, r = self._generate(state, a)
        return r + self._cfg.gamma * self.rollout(s1, None, depth + 1)

    def simulate(self, state, node: int, depth: int) -> float:
        # Stop simulation when updates are no longer meaningful.
        if (self._cfg.gamma ** depth < self._cfg.threshold):
            return 0.0

        # NOTE(ycho): Let's just accept either option.
        if isinstance(node, int):
            source = self._tree.vertex(node)
        else:
            source = node

        # Leaf Node?
        if self._prop['N'][source] == 0:
            # Add child node & connecting edge.
            for action in self._actions:
                v = self._add_node(True)
                e = self._tree.add_edge(source, v)
                self._prop['action'][e] = action
                logging.debug(F'Added action from {source} -> {v}')

            # Update source(parent of new node)
            new_value = self.rollout(state, node, depth)
            self._prop['N'][source] += 1
            self._prop['value'][source] = new_value
            return new_value

        a, action_node = self.search_best_action(node)

        s1, o, r = self._generate(state, a)
        obs_node = self.get_observation_node(action_node, o)
        cum_reward = r + self._cfg.gamma * \
            self.simulate(s1, obs_node, depth + 1)

        # Update belief state.
        b = self._prop['belief'][node]
        if len(b) >= self._cfg.num_particle:
            # Replace existing element (oldest first).
            i = self._prop['belief_index'][node]
            b[i] = state

            # Increment index.
            i2 = (i + 1) % self._cfg.num_particle
            self._prop['belief_index'][node] = i2
        else:
            # Append!
            b.append(state)

        # Bookkeeping for the number of visits.
        self._prop['N'][node] += 1
        self._prop['N'][action_node] += 1

        # Update rule for value...
        dv = (cum_reward - self._prop['value']
              [action_node]) / self._prop['N'][action_node]
        self._prop['value'][action_node] += dv

        return cum_reward

    def get_observation_node(self, node: int, observation) -> Vertex:

        # NOTE(ycho): Find corresponding observation node ...
        # FIXME(ycho): Is there an easier way to do this?
        vertex = self._tree.vertex(node)
        for e in vertex.out_edges():
            a = self._prop['action'][e]
            if (a == observation):
                return e.target()

        # NOTE(ycho): In case the nodes doesn't exist, create one.
        # v = self._tree.add_vertex()
        v = self._add_node(False)
        e = self._tree.add_edge(vertex, v)
        # TODO(ycho): rename `action` to something more like
        # action_or_observation? :/
        self._prop['action'][e] = observation
        logging.debug(F'added obs. node {v}')

        return v

    def search(self) -> int:
        node = self._root

        # NOTE(ycho): Why is this copied?
        # TODO(ycho): Check if iteratively updating
        # the belief distribution during the loop results
        # in an invalid sampling procedure.
        belief = np.copy(self._prop['belief'][node])

        for _ in range(self._cfg.max_iter):
            if (belief is None) or len(belief) <= 0:
                s = np.random.choice(self._states)
            else:
                s = np.random.choice(belief)  # Belief@node
            self.simulate(s, node, 0)
        action, _ = self.search_best_action(node, explore=False)
        return action

    def _sample_posterior(self, belief, action, observation):
        # Sample previous state based on prior belief.
        if (belief is None) or len(belief) <= 0:
            s = np.random.choice(self._states)
        else:
            s = np.random.choice(belief)

        # Sample posterior state, based on transition distribution.
        # FIXME(ycho): what if _generate(...) never
        # retrieves the exact observation??
        while True:
            s_next, o_next, _ = self._generate(int(s), action)
            if o_next == observation:
                return s_next

    def _prune(self, node: Vertex, del_nodes: List[Vertex], depth: int = 0):
        # Accumulate nodes to delete within the tree.
        del_nodes.append(node)
        for e in node.out_edges():
            self._prune(e.target(), del_nodes, depth + 1)

        # NOTE(ycho): Cannot remove edges while iterating,
        # since the edge descriptors are invalidated.
        es = [e for e in node.out_edges()]
        for e in es:
            self._tree.remove_edge(e)

        # Finalize pruning, if at the root level.
        if depth == 0:
            self._tree.remove_vertex(del_nodes)

    def update(self, action, observation):
        """stateful update, current node & belief."""

        # (1) Find action node.
        action_node = None
        for e in self._root.out_edges():
            a = self._prop['action'][e]
            if (a == action):
                action_node = e.target()
                break
        else:
            raise ValueError('Target action node not found')

        # (2) Find observation node.
        obs_node = self.get_observation_node(action_node, observation)

        # (3) Prune.
        self._tree.remove_edge(
            self._tree.edge(action_node, obs_node))  # root - action -/- obs
        self._prune(self._root, [])

        # (4) Update the root node. (==obs_node)
        # Since the previous vertex descriptor is invalidated,
        # we need to find the corresponding vertex again.
        # NOTE(ycho): If you think about it, it's pretty "obvious"
        # that the root node will correspond to index==0.
        # However, since this is not a "guaranteed" property,
        # we also robustify the root-finding process with a boolean flag.
        self._root = None

        # NOTE(ycho): Fast check based on the assumption that
        # the root index will be generally 0.
        root = self._tree.vertex(0)
        if root.in_degree == 0:
            self._root = root

        # NOTE(ycho): Slow check based on the property
        # that the root node does not have any parents.
        if self._root is None:
            roots = find_vertex(self._tree, 'in', 0)
            if len(roots) != 1:
                raise ValueError('More than one root found!')
            self._root = roots[0]

        # (5) Update belief.
        prior = self._prop['belief'][self._root]
        posterior = []
        for _ in range(self._cfg.num_particle):
            posterior.append(
                self._sample_posterior(
                    prior, action, observation))
        self._prop['belief'][self._root] = posterior
        logging.debug(np.mean(posterior), np.var(posterior))


def get_problem():
    # transition probs
    a = [9.000000000000000222e-01,
         4.000000000000000222e-01,
         3.499999999999999778e-01,
         8.000000000000000444e-01,
         1.000000000000000056e-01,
         5.999999999999999778e-01,
         6.500000000000000222e-01,
         2.000000000000000111e-01, ]
    b = [
        8.000000000000000444e-01,
        4.000000000000000222e-01,
        2.999999999999999889e-01,
        5.000000000000000000e-01,
        2.000000000000000111e-01,
        5.999999999999999778e-01,
        6.999999999999999556e-01,
        5.000000000000000000e-01,
    ]
    r = [
        1.000000000000000000e+00,
        5.000000000000000000e+01,
        1.000000000000000000e+00,
        2.000000000000000000e+00,
    ]
    a = np.reshape(a, (2, 2, 2))
    b = np.reshape(b, (2, 2, 2))
    r = np.reshape(r, (2, 2))

    # Make it less confusing
    a = np.transpose(a, (1, 2, 0))
    b = np.transpose(b, (1, 2, 0))

    # A --> transition
    # [[[0.9  0.1 ]
    #   [0.4  0.6 ]]
    #
    #  [[0.35 0.65]
    #   [0.8  0.2 ]]]

    # a(0,0) --> (0.9, 0.1)
    # a(0,1) --> (0.4, 0.6)
    # a(1,0) --> (0.35, 0.65)
    # a(1,1) --> (0.8, 0.2)
    # ^^^ I guess this generally means lower probability of transitions
    # being successful.

    # B --> observation
    # it's a function of
    # (next_state, prev_action) ... why?
    #[[[0.8 0.2]
    #  [0.4 0.6]]
    #
    # [[0.3 0.7]
    #  [0.5 0.5]]]

    # b(0,0) --> (0.8, 0.2)
    # b(0,1) --> (0.4, 0.6)
    # b(1,0) --> (0.3, 0.7)
    # b(1,1) --> (0.5, 0.5)
    # I'm not sure why it's not just a function of
    # the

    return a, b, r


def main():
    import numpy as np
    # np.random.seed(0)

    # from auxilliary import BuildTree, UCB
    from numpy.random import binomial, choice, multinomial

    # NOTE(ycho): No idea what this is
    a, b, r = get_problem()
    print('a')
    print(a)
    print('b')
    print(b)
    print('r')
    print(r)

    def _generate(s, act):
        ss = multinomial(1, a[s, act, :])
        ss = int(np.nonzero(ss)[0])
        o = multinomial(1, b[ss, act, :])
        o = int(np.nonzero(o)[0])
        rw = r[s, act]
        return ss, o, rw

    agent = POMCP(POMCP.Config(), _generate)
    agent.initialize([0, 1], [0, 1], [0, 1])

    for i in range(10):
        action = agent.search()
        print('action = ', action)
        observation = np.random.choice([0, 1])
        agent.update(action, observation)


if __name__ == '__main__':
    main()
