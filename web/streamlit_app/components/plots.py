from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import streamlit as st

from src.dynamics import CascadeResult
from src.influence_game import Action, InfluenceGame
from src.viz import (
    LayoutCache,
    build_pyvis_network,
    draw_cascade_history_matplotlib,
    draw_profile_matplotlib,
)


def show_profile_plot(
    game: InfluenceGame,
    profile: Mapping[Any, Action],
    forcing_set: Optional[Iterable[Any]] = None,
    highlight_nodes: Optional[Iterable[Any]] = None,
    title: Optional[str] = None,
    layout_cache: Optional[LayoutCache] = None,
    layout: str = "spring",
    percent_baseline: str = "incoming",
    use_container_width: bool = True,
) -> LayoutCache:
    """Render a single profile with matplotlib inside Streamlit."""
    fig, _, layout_cache_out = draw_profile_matplotlib(
        game,
        profile=profile,
        forcing_set=forcing_set,
        highlight_nodes=highlight_nodes,
        title=title,
        layout_cache=layout_cache,
        layout=layout,
        percent_baseline=percent_baseline,
        ax=None,
    )

    st.pyplot(fig, use_container_width=use_container_width)
    plt.close(fig)

    return layout_cache_out


def show_cascade_history_plot(
    game: InfluenceGame,
    cascade: CascadeResult,
    forcing_set: Optional[Iterable[Any]] = None,
    layout_cache: Optional[LayoutCache] = None,
    layout: str = "spring",
    max_steps_to_plot: int = 4,
    title: Optional[str] = None,
    percent_baseline: str = "incoming",
    use_container_width: bool = True,
) -> LayoutCache:
    """Render several cascade frames inside Streamlit."""
    fig, layout_cache_out = draw_cascade_history_matplotlib(
        game,
        cascade=cascade,
        forcing_set=forcing_set,
        layout_cache=layout_cache,
        layout=layout,
        max_steps_to_plot=max_steps_to_plot,
        percent_baseline=percent_baseline,
    )

    if title is not None:
        fig.suptitle(title)

    st.pyplot(fig, use_container_width=use_container_width)
    plt.close(fig)

    return layout_cache_out


def show_pyvis_network(
    game: InfluenceGame,
    profile: Optional[Mapping[Any, Action]] = None,
    forcing_set: Optional[Iterable[Any]] = None,
    highlight_nodes: Optional[Iterable[Any]] = None,
    directed: Optional[bool] = None,
    height: int = 600,
    scrolling: bool = True,
) -> None:
    """Render a pyvis interactive network in Streamlit."""
    from streamlit.components.v1 import html as st_html

    try:
        net = build_pyvis_network(
            game,
            profile=profile,
            forcing_set=forcing_set,
            highlight_nodes=highlight_nodes,
            directed=directed,
        )
    except Exception as exc:
        st.warning(
            f"PyVis network could not be built: {exc}. "
            "Install 'pyvis' and make sure compatible versions are used."
        )
        return

    if net is None:
        st.warning(
            "PyVis network builder returned None. "
            "Check that 'pyvis' is installed and up to date."
        )
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        tmp_path = Path(tmp.name)
        try:
            net.write_html(str(tmp_path))
        except Exception as exc:
            st.warning(
                f"PyVis network could not be written to HTML: {exc}. "
                "The interactive view is disabled for this session."
            )
            return

    try:
        html_content = tmp_path.read_text(encoding="utf-8")
    except Exception as exc:
        st.warning(
            f"PyVis HTML file could not be read: {exc}. "
            "The interactive view is disabled."
        )
        return

    st_html(html_content, height=height, scrolling=scrolling)
