#!/usr/bin/env python3

# import networkx as nx
from graph_tool import Graph
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
        threshold: float = 0.005  # discount threshold
        max_iter: int = int(1e4)  # "number of runs from node"
        num_particle: int = 1200  # number of particles??

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
        self._prop['is_root'] = self._tree.new_vertex_property(
            'int', val=False)

        # NOTE(ycho): I gues the only reason that belief is a vector<int>
        # is because it's currently just a set of integer-valued states.
        # It could/should probably be a much more complex thing, e.g. an actual
        # distribution.
        self._prop['belief'] = self._tree.new_vertex_property('vector<int>')
        self._prop['belief_index'] = self._tree.new_vertex_property('int')
        self._prop['action'] = self._tree.new_edge_property('int')

        # Create the root node.
        root = self._add_node(False, is_root=True)
        self._root = root

    def _add_node(self, is_action: bool, value: float = 0.0,
                  N: int = 0, belief=None, belief_index: int = 0,
                  is_root: bool = False):
        v = self._tree.add_vertex()

        self._prop['value'][v] = value
        self._prop['N'][v] = N
        self._prop['is_action'][v] = is_action
        self._prop['is_root'][v] = is_root

        if not is_action:
            self._prop['belief'][v] = []
            self._prop['belief_index'][v] = 0
        # print(F'added node {v} action={is_action}')
        return v

    def initialize(self, states, actions, observations):
        self._states = states
        self._actions = actions
        self._observations = observations

    def search_best_action(
            self, node: int, explore: bool = True, debug: bool = False):
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
                    if debug:
                        print('exploring child with no visits!')
                    return (action, child)

                # Compute utility via UCB
                value = ucb(self._prop['N'][v], self._prop['N'][child],
                            self._prop['value'][child],
                            self._cfg.c)

                if debug:
                    print(F'action value = {action}:{value}')

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

    def rollout(self, state, node: int, depth: int):
        if (self._cfg.gamma ** depth < self._cfg.threshold):
            return 0.0

        # (1) sample action(node) from rollout policy
        # FIXME(ycho): hardcoded uniform-random policy
        a = np.random.choice(self._actions)

        # (2) s1,o,r = generate(s,a)
        s1, _, r = self._generate(state, a)
        return r + self._cfg.gamma * self.rollout(s1, None, depth + 1)

    # @profile
    def simulate(self, state, node: int, depth: int) -> float:
        if (self._cfg.gamma ** depth < self._cfg.threshold):
            return 0.0

        # NOTE(ycho): Let's just accept either option.
        # print('node', node)  # // 0
        if isinstance(node, int):
            source = self._tree.vertex(node)
        else:
            source = node

        # Leaf Node?
        # if source.out_degree == 0 or self._prop['N'][source] == 0:
        if self._prop['N'][source] == 0:
            # Add child node & connecting edge.
            for action in self._actions:
                v = self._add_node(True)
                # print(F'added action from {source} -> {v}')
                # v = self._tree.add_vertex()
                e = self._tree.add_edge(source, v)
                self._prop['action'][e] = action
                # self._prop['is_action'][v] = True

            # Update source(==parent of new node)
            new_value = self.rollout(state, node, depth)
            self._prop['N'][source] += 1
            self._prop['value'][source] = new_value
            return new_value

        # print('is action?', self._prop['is_action'][node])
        # print('node', node)
        a, next_node = self.search_best_action(node)
        # print(a, next_node)

        s1, o, r = self._generate(state, a)
        next_obs_node = self.get_observation_node(next_node, o)
        # print(F'got obs.node {next_node}')
        cum_reward = r + self._cfg.gamma * \
            self.simulate(s1, next_obs_node, depth + 1)

        # "Backtrack" <<?
        # Wait, so `state` is a float/double??
        b = self._prop['belief'][node]
        if len(b) >= self._cfg.num_particle:
            # Replace existing element (oldest first)
            # print('reached max')
            i = self._prop['belief_index'][node]
            b[i] = state

            # Increment index ...
            i2 = (i + 1) % self._cfg.num_particle
            # print(F'i = {i} -> {i2}')
            self._prop['belief_index'][node] = i2
        else:
            # Append!! :)
            b.append(state)

        self._prop['N'][node] += 1
        self._prop['N'][next_node] += 1

        # nodes num children increases ?
        # TODO(ycho): is Nc number of visits?
        # child-node's num children [ALSO] increases ?
        dv = (cum_reward - self._prop['value']
              [next_node]) / self._prop['N'][next_node]
        self._prop['value'][next_node] += dv
        return cum_reward

    def get_observation_node(self, node: int, sample):

        # NOTE(ycho): Find corresponding observation node ...
        # FIXME(ycho): Is there an easier way to do this?
        vertex = self._tree.vertex(node)
        for e in vertex.out_edges():
            a = self._prop['action'][e]
            if (a == sample):
                return e.target()

        # NOTE(ycho): In case the nodes doesn't exist, create one.
        # v = self._tree.add_vertex()
        v = self._add_node(False)
        # print(F'added obs. node {v}')
        e = self._tree.add_edge(vertex, v)
        # self._prop['is_action'][v] = False
        self._prop['action'][e] = sample
        # NOTE(ycho): No need to repeat the search ...
        # return self.get_observation_node(node, sample)
        return v

    def search(self):
        node = self._root

        # NOTE(ycho): why is this copied?
        belief = np.copy(self._prop['belief'][node])

        for _ in range(self._cfg.max_iter):
            # TODO(ycho): Figure out why compare to []?
            # if belief == []:
            if (belief is None) or len(belief) <= 0:
                s = np.random.choice(self._states)
            else:
                s = np.random.choice(belief)  # Belief@node
            self.simulate(s, node, 0)
        # action, _ = self.search_best_action(-1, use_ucb=True)
        action, _ = self.search_best_action(node, explore=False, debug=True)
        return action

    def _posterior(self, belief, action, observation):
        if (belief is None) or len(belief) <= 0:
            s = np.random.choice(self._states)
        else:
            s = np.random.choice(belief)

        #Sample from transition distribution
        while True:
            s_next, o_next, _ = self._generate(int(s), action)
            if o_next == observation:
                return s_next
        # result = self._posterior(belief, action, observation)
        # return result

    def _prune(self, node, del_nodes, depth: int = 0):
        # Accumulate nodes to delete within the tree.
        del_nodes.append(node)
        for e in node.out_edges():
            self._prune(e.target(), del_nodes, depth + 1)

        # NOTE(ycho): Cannot remove while iterating.
        es = [e for e in node.out_edges()]
        for e in es:
            self._tree.remove_edge(e)

        # Finalize pruning at the root level.
        if depth == 0:
            self._tree.remove_vertex(del_nodes)

    def update(self, action, observation):
        """stateful update, current node & belief."""

        # (1) find action node
        action_node = None
        for e in self._root.out_edges():
            a = self._prop['action'][e]
            if (a == action):
                action_node = e.target()
                break
        else:
            raise ValueError('Target action node not found')

        # (2) find observation node
        # print('get_observation_node')
        obs_node = self.get_observation_node(action_node, observation)

        # print(F'from = {self._root}')
        # (3) TODO(ycho): prune
        self._tree.remove_edge(
            self._tree.edge(action_node, obs_node))  # root - action -/- obs
        # node_id = int(obs_node)
        node_id = self._tree.vertex_index[obs_node]
        self._prop['is_root'][obs_node] = True
        self._prune(self._root, [])

        # (4) update root
        # obs_node = self._tree.vertex(node_id)
        # FIXME(ycho): If you think about it, it's pretty "obvious"
        # that the root node will correspond to index==0...
        # this is, of course, the most robust way to do it...
        self._root = None
        if self._prop['is_root'][0]:
            self._root = self._tree.vertex(0)
            # print('zero is root')
        if self._root is None:
            roots = find_vertex(self._tree, self._prop['is_root'], True)
            # print(roots)
            self._root = roots[0]

        # obs_node =
        # print('valid?', obs_node.is_valid())
        # self._root = obs_node
        # print(F'to   = {self._root}')

        # (5) update belief
        prior = np.copy(self._prop['belief'][self._root])
        posterior = []
        for _ in range(self._cfg.num_particle):
            posterior.append(self._posterior(prior, action, observation))
        self._prop['belief'][self._root] = posterior
        print(np.mean(posterior), np.var(posterior))


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
        # print('s', s)
        # print('act', act)
        # print('a', a)
        # print('prob', a[:,s,act])
        ss = multinomial(1, a[s, act, :])
        # print('ss', ss)
        ss = int(np.nonzero(ss)[0])
        o = multinomial(1, b[ss, act, :])
        o = int(np.nonzero(o)[0])
        # print('o', o)
        rw = r[s, act]
        return ss, o, rw
    agent = POMCP(POMCP.Config(), _generate)
    agent.initialize([0, 1], [0, 1], [0, 1])

    for i in range(10):
        action = agent.search()
        print('action = ', action)
        observation = np.random.choice([0, 1])
        agent.update(action, observation)
        # (1) prune
        # (2) update_belief(action,observation)
        # break


if __name__ == '__main__':
    main()
