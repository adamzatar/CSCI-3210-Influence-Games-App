from __future__ import annotations

from .influence_game import InfluenceGame
from .psne import PSNEResult, PSNESolver


class IrfanMostInfluential:
    '''
    Used to find the most influential nodes in a graph as defined by
    Professor Irfan
    '''

    def __init__(self, game: InfluenceGame):
        self.game = game
        self._nodes, self._index = self.game.canonical_order()
        self.psne = self.generate_psne().profiles

    def generate_psne(self):
        '''
        Returns all PSNE in the game
        '''
        solver = PSNESolver(self.game)
        return solver.enumerate_psne_bruteforce()
    
    def check_force_psne(self, desired_psne, list_profile):
        '''
        Returns whether the given action profile forces solely the desired psne

        :param desired_psne: the psne we want
        :param list_profile: the action profile being selected
        '''
        is_desired_valid = False

        # Iterate over each PSNE
        for eq in self.psne:
            forces_eq = True

            # Check if @list_profile forces the current PSNE
            for node, action in list_profile:
                if eq.get(node, None) != action:
                    forces_eq = False
            
            # Handle cases where the PSNE is the desired case
            if eq == desired_psne:
                if forces_eq:
                    is_desired_valid = True
                else:
                    return False
            # If we force a non desired equilibria, return False
            elif forces_eq:
                return False

        return is_desired_valid

    def get_subsequences(self, lst):
        '''
        Gets subsequences of a list

        :param list: the list we want the subsequences of
        '''
        if not lst:
            return [[]]
        
        remaining = self.get_subsequences(lst[1:])
        with_first = [[lst[0]] + r for r in remaining]
        return remaining + with_first

    def profile_to_list(self, profile):
        '''
        Returns a list of tuples representing an action profile
        
        :param profile: the profile being converted to a list
        '''
        res = []

        for key, value in profile.items():
            res.append((key, value))

        return res

    def get_most_influential(self, desired_psne):
        '''
        Returns a list of all combinations of most influential nodes based on
        Professor Irfan's definition
        '''
        res = []

        list_psne = self.profile_to_list(desired_psne)
        list_profiles = self.get_subsequences(list_psne)

        # Iterate over each potential list profile
        for list_profile in list_profiles:

            # Only check profiles that could be smaller or equal to our current best
            if not res or len(res[0]) >= len(list_profile):
                # If our profile forces the desired psne
                if self.check_force_psne(desired_psne, list_profile):
                    # If we found a smaller profile, make it our new res
                    if not res or len(res[0]) > len(list_profile):
                        res = [list_profile]
                    # If we found a profile of equal length to our current best, add it to our results
                    elif len(res[0]) == len(list_profile):
                        res.append(list_profile)

        return res
    
if __name__ == "__main__":
    from .influence_game import InfluenceGame

    game = InfluenceGame(directed=False)
    for node in ["A", "B", "C", "D"]:
        game.add_node(node, threshold=1.5, label=node)

    edges = [("A", "B"), ("A", "C"), ("B", "C")]
    game.add_edges_from(edges, default_weight=1.0)

    solver = IrfanMostInfluential(game)
    desired_psne = solver.generate_psne().profiles[0]

    profile = {'A': 0, 'B': 1, 'C': 0}

    profile = solver.get_subsequences(solver.profile_to_list({'A': 0, 'B': 1, 'C': 1}))[1]
    print(solver.get_most_influential(desired_psne))
