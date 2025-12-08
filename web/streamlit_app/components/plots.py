# web/streamlit_app/components/plots.py
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
    use_container_width: bool = True,
) -> LayoutCache:
    """
    Render a single profile using draw_profile_matplotlib inside Streamlit.

    Parameters
    ----------
    game:
        InfluenceGame to visualize.
    profile:
        Mapping from node to action in {0, 1}.
    forcing_set:
        Optional nodes that belong to the forcing set. They will be
        outlined with thicker borders.
    highlight_nodes:
        Optional nodes drawn slightly larger to emphasize them.
    title:
        Optional title added above the plot.
    layout_cache:
        Optional LayoutCache to reuse node positions so multiple plots
        stay visually aligned.
    layout:
        Layout algorithm name if a new layout must be computed.
    use_container_width:
        Whether to allow Streamlit to expand the plot horizontally.

    Returns
    -------
    LayoutCache
        The layout cache used for this plot. You can pass this back in
        on a later call to reuse the positions.
    """
    fig, ax, layout_cache_out = draw_profile_matplotlib(
        game,
        profile=profile,
        forcing_set=forcing_set,
        highlight_nodes=highlight_nodes,
        title=title,
        layout_cache=layout_cache,
        layout=layout,
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
    use_container_width: bool = True,
) -> LayoutCache:
    """
    Render a cascade trajectory inside Streamlit.

    This uses draw_cascade_history_matplotlib to build a multi panel
    figure that shows several time steps from the cascade history.

    Parameters
    ----------
    game:
        InfluenceGame whose cascade is being visualized.
    cascade:
        CascadeResult produced by CascadeSimulator.
    forcing_set:
        Optional forcing set, nodes outlined in every frame.
    layout_cache:
        Optional LayoutCache to reuse positions. This is helpful if you
        also showed a single profile before and want node locations to
        match.
    layout:
        Layout algorithm name if a new layout must be computed.
    max_steps_to_plot:
        Maximum number of time steps to show.
    title:
        Optional title string added as suptitle.
    use_container_width:
        Whether to let Streamlit expand the figure horizontally.

    Returns
    -------
    LayoutCache
        The layout cache used for this plot.
    """
    fig, layout_cache_out = draw_cascade_history_matplotlib(
        game,
        cascade=cascade,
        forcing_set=forcing_set,
        layout_cache=layout_cache,
        layout=layout,
        max_steps_to_plot=max_steps_to_plot,
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
    """
    Render a pyvis interactive network inside Streamlit.

    This builds a pyvis.Network from the InfluenceGame and optional
    profile, then embeds the HTML in the Streamlit app.

    Parameters
    ----------
    game:
        InfluenceGame to visualize.
    profile:
        Optional mapping from node to action. If provided, node color
        and hover tooltips will reflect the current actions.
    forcing_set:
        Optional forcing set, nodes will have thicker borders.
    highlight_nodes:
        Optional nodes drawn larger.
    directed:
        Override the directed flag for visualization. If None, uses
        game.directed.
    height:
        Height of the embedded HTML in pixels.
    scrolling:
        Whether to allow scrolling inside the HTML frame.
    """
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
            # Write HTML to disk without trying to open a browser.
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