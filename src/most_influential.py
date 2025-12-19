from __future__ import annotations

from .influence_game import InfluenceGame
from .psne import PSNESolver


class MostInfluential:
    '''
    Used to find the most influential nodes in a graph based on our definition,
    where the most influential nodes are a set of nodes that when playing action 1
    make every other node's best response to be action 1
    '''

    def __init__(self, game: InfluenceGame):
        self.game = game
        self._nodes, self._index = self.game.canonical_order()
        self.psne_solver = PSNESolver(game)

    def generate_forced_profile(self, forced):
        '''
        Returns a profile where each node in @forced plays action 1, every other node plays 0

        :param forced: a list of nodes that are forced to play action 1
        '''
        res = {}

        for node in self._nodes:
            if node in forced:
                res[node] = 1
            else:
                res[node] = 0
        
        return res
        
    def check_forcing_set(self, forced):
        '''
        Returns whether @forced forces all other nodes to play action 1

        :param forced: the forced set of nodes
        '''
        remaining_nodes = [node for node in self._nodes if node not in forced]
        profile = self.generate_forced_profile(forced)

        update = True
        while update and len(remaining_nodes) > 0:
            update = False

            for i in range(len(remaining_nodes)):
                if self.psne_solver.best_response(remaining_nodes[i], profile) == 1:
                    update = True
                    profile[remaining_nodes[i]] = 1
                    remaining_nodes.pop(i)
                    break

        return len(remaining_nodes) == 0
    
    def get_subsequences(self, lst):
        '''
        Gets subsequences of a list

        :param lst: the list we want the subsequences of
        '''
        if not lst:
            return [[]]
        
        remaining = self.get_subsequences(lst[1:])
        with_first = [[lst[0]] + r for r in remaining]
        return remaining + with_first

    def get_most_influential(self):
        '''
        Returns a list of all sets of most influential nodes of minimum length
        '''
        res = []
        forced_nodes_list = self.get_subsequences(self._nodes)

        for forced_nodes in forced_nodes_list:
            if not res or len(res[0]) >= len(forced_nodes):
                if self.check_forcing_set(forced_nodes):
                    if not res or len(res[0]) > len(forced_nodes):
                        res = [forced_nodes]
                    elif len(res[0]) == len(forced_nodes):
                        res.append(forced_nodes)

        return res

if __name__ == "__main__":
    from .influence_game import InfluenceGame

    game = InfluenceGame(directed=False)
    for node in ["A", "B", "C", "D"]:
        game.add_node(node, threshold=1, label=node)

    edges = [("A", "B"), ("A", "C"), ("B", "C")]
    game.add_edges_from(edges, default_weight=1.0)

    solver = MostInfluential(game)
    print(solver.get_most_influential())
