# src/__init__.py
from __future__ import annotations

"""
Top level package for the CSCI 3210 influence games project.

This package exposes the main building blocks of the codebase so that
notebooks, scripts, and tests can import them with:

    from src import (
        InfluenceGame,
        CascadeSimulator,
        PSNESolver,
        ForcingSetFinder,
    )

and so on.
"""

from .influence_game import InfluenceGame, Action
from .dynamics import CascadeSimulator, CascadeResult
from .psne import PSNESolver, PSNEResult
from .forcing import ForcingSetFinder, ForcingSetResult
from .irfan_most_influential import IrfanMostInfluential, IrfanResult
from .viz import (
    LayoutCache,
    draw_profile_matplotlib,
    draw_cascade_history_matplotlib,
    build_pyvis_network,
)
from .utils import (
    profile_to_string,
    build_complete_symmetric_game,
    build_line_graph_game,
    build_random_threshold_game,
    build_custom_game,
    kuran_style_star_example,
)

__all__ = [
    "Action",
    "InfluenceGame",
    "CascadeSimulator",
    "CascadeResult",
    "PSNESolver",
    "PSNEResult",
    "ForcingSetFinder",
    "ForcingSetResult",
    "IrfanMostInfluential",
    "IrfanResult",
    "LayoutCache",
    "draw_profile_matplotlib",
    "draw_cascade_history_matplotlib",
    "build_pyvis_network",
    "profile_to_string",
    "build_complete_symmetric_game",
    "build_line_graph_game",
    "build_random_threshold_game",
    "build_custom_game",
    "kuran_style_star_example",
]
