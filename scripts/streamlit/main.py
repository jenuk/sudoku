import argparse
import math
import random
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from einops import rearrange
from PIL import Image
from scipy.stats import binomtest

import streamlit as st

from tqdm import tqdm


def set_style():
    st.markdown("""
    <style> 
        .sudoku_block {
            font-family:monospace;
            font-size: 14px;
            line-height: 16px;
            border: 3px solid;
            border-radius: 10px;
            padding: 10px;
            width: fit-content;
        }
    </style>
    """, unsafe_allow_html=True)


def get_icon():
    return Image.open("icon.png")


def get_num_sudokus(filename: Path) -> int:
    num_bytes = filename.stat().st_size
    # one line contains a partial and a filled field with each 81 characters, a
    # space and a linebreak -> 2*81+2 = 164 characters per line
    return num_bytes // 164


@st.cache_data(max_entries=10)
def load_dataset(filename: Path, max_lines: Optional[int]=None) -> np.ndarray:
    """ Loads a file containing a file of the form:
        <puzzle> <solution>\n
    with lines in raster-scan order and empty spots represented by a zero.
    Returns a numpy array of the shape (N, 2, 9, 9) of dtype "uint8", where
    N is the number of sudokus in the file, out[:, 0] are puzzles and out[:, 1]
    are solutions.

    This function is re-implementing a caching mechanism to avoid replaying
    of the progress bar
    """
    def iter_file(file):
        whitespace = {" ", "\n"}
        # read file in small 1MB buffers
        while (line := file.read(1024*1024)) != "":
            for ch in line:
                if ch not in whitespace:
                    yield ch

    num_lines = filename.stat().st_size // (2 * (81 + 1))
    if max_lines is not None:
        num_lines = min(num_lines, max_lines)
    count = 162 * num_lines
    print(f"Starting to load {filename}")
    with open(filename) as file:
        out = np.fromiter(tqdm(iter_file(file), total=count), dtype=np.uint8, count=count)
    out = rearrange(out, "(n c h w) -> n c h w", n=num_lines, c=2, h=9, w=9)
    return out


@st.cache_data()
def summarize_data(key: str, _sudokus: Dict[str, np.ndarray]) -> str:
    """Gives very general information of a dataset, doesn't change over time
    so it is cached."""

    body = ""
    avg_empties = np.mean(np.sum(_sudokus[key][:, 0] == 0, axis=(-1, -2)).astype(float))
    body += f"- Average number of hints: {81 - avg_empties:.1f}\n"
    body += f"- loaded {_sudokus[key].shape[0]:,} sudokus.\n"
    flattened = rearrange(_sudokus[key], "n c h w -> n c (h w)")
    uniques = np.unique(flattened[:, 1], axis=0)
    if uniques.shape[0] == flattened.shape[0]:
        body += "- has :green[no duplicate solutions].\n"
    else:
        unique_puzzles = np.unique(flattened[:, 0], axis=0)
        body += f"- has :red[{flattened.shape[0] - uniques.shape[0]:,} duplicate solutions]\n"
        if unique_puzzles.shape[0] == flattened.shape[0]:
            body += "    - has :green[no duplicate puzzles].\n"
        else:
            body += f"    - of this there are :red[{flattened.shape[0] - unique_puzzles.shape[0]:,} duplicate puzzles]\n"
    return body


@st.cache_data(max_entries=1)
def get_num_hints_dataframe(selected: List[str], _sudokus: Dict[str, np.ndarray]):
    dataframe = {"Source": [], "Number of Hints": [], "Count": [], "Frequency": []}
    for key in selected:
        num_hints, counts = np.unique(np.sum(_sudokus[key][:, 0] != 0,
                                             axis=(1,2)),
                                      return_counts=True)
        freqs = (counts.astype(float) / _sudokus[key].shape[0])
        dataframe["Source"].extend([key]*num_hints.shape[0])
        dataframe["Number of Hints"].extend(num_hints.tolist())
        dataframe["Count"].extend(counts.tolist())
        dataframe["Frequency"].extend(freqs.tolist())

    return pd.DataFrame(dataframe)


@st.cache_data(max_entries=1)
def get_spatial_distribution(
        selected: List[str],
        _sudokus: Dict[str, np.ndarray],
        ) -> Tuple[np.ndarray, np.ndarray]:
    # returns (len(selected), 10, h, w) with dtype int32, and float64

    count = np.zeros((len(selected), 10, 9, 9), dtype=np.int32)
    freq = np.zeros((len(selected), 10, 9, 9), dtype=np.float64)
    for l, key in enumerate(selected):
        for k in range(10):
            t = 0 if k == 0 else 1
            count[l, k] = np.sum((_sudokus[key][:, t] == k), axis=0)
        freq[l] = count[l].astype(float) / _sudokus[key].shape[0]

    return count, freq


def display_sudoku(sudoku: np.ndarray):
    """Displays a sudoku using html
    
    Expects sudoku of shape (2, 9, 9)
    """
    assert type(sudoku) == np.ndarray and sudoku.dtype == np.uint8
    # -> no unexpected data is passed, html probably safe
    html = "<div class='sudoku_block'>"
    for i in range(9):
        for j in range(9):
            if sudoku[0, i, j] != 0:
                html += f"<b>{sudoku[1, i, j]}</b> "
            else:
                html += f"<span style='color:green'>{sudoku[1, i, j]}</span> "
            if j == 2 or j == 5:
                html += "│ "
        html += "<br>\n"
        if i == 2 or i == 5:
            html += (("─"*(2*3) + "┼" + "─")*3)[:-3] + "<br>\n"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def get_pvals(counts, total_count):
    pvals = np.zeros((9, 9))
    prob = np.mean(counts)/total_count
    for i in range(9):
        for j in range(9):
            res = binomtest(counts[i, j], total_count, prob)
            pvals[i, j] = res.pvalue

    return pvals


def main(demo_mode: bool=False):
    st.set_page_config(
        page_title="Sudoku Analyzer • Streamlit",
        page_icon=get_icon(),
        layout="wide",
    )
    set_style()
    with st.columns([0.25, 0.5, 0.25])[1]: # centered layout for text
        st.write(
            """
            # Analyze Sudoku Datasets

            Creating an unbiased selection of sudoku puzzles is a
            (surprisingly) hard task. With this Streamlit script I want to
            provide an option to check and compare existing datasets against
            one another without too much trouble.

            To tackle this problem we will start with loading one or more
            dataset files. Loading and parsing many sudokus into an appropriate
            data structure takes a moment, and at least some RAM. Additionally
            it slows down the computation of all statistics. So to speed up the
            experience you can set a maximum number of sudokus loaded. Changing
            this number later will completely reload and recompute all data
            with the new limit. For reference the number of sudokus in
            available data files is listed below.
            """
        )
        if demo_mode:
            st.write(
                """
                This app runs in **demo mode**. With this enabled there are
                only a few small data sources provided and there is no
                possibility to change number of loaded sudokus.

                Demo mode should run automatically if you just clone the
                repository and start streamlit. To run in normal mode create a
                new folder `data` and fill it with some data.
                """
            )
            with st.expander("Click to see list of available data"):
                st.write(
                    """
                    There are some public datasets from kaggle, each is limited
                    to the first 100,000 sudokus from their original size. I
                    thank each of them for providing this data!
                    - `park`: originally 1,000,000 sudokus by
                      [Kyubyong Park](https://github.com/Kyubyong), provided
                      [here](https://www.kaggle.com/datasets/bryanpark/sudoku).
                    - `radcliffe`: originally 3,000,000 sudokus by
                      [David Radcliffe](https://github.com/Radcliffe), provided
                      [here](https://www.kaggle.com/datasets/radcliffe/3-million-sudoku-puzzles-with-ratings).
                    - `rao`: originally 9,000,000 sudokus by
                      [Rohan Rao](https://github.com/vopani), provided
                      [here](https://www.kaggle.com/datasets/rohanrao/sudoku).

                    In addition I provide some new datasets, to showcase some
                    specific functionality of this app:
                    - `jenuk` [My](https://github.com/jenuk) current best
                      effort. No final set published yet, but see
                      [here](https://github.com/jenuk/sudoku) for code to
                      generate more.
                    - `few_hints` subsample of large dataset with only few
                      hints.
                    - `duplicate_puzzle` each row is contained two times, i.e.
                      duplicate puzzles.
                    - `duplicate_solutions` each solution is contained two
                      times but no repetition of the puzzles, i.e. duplicate
                      solutions.
                    - `single_bias_0` subsample of large dataset where a
                      specific puzzle location is always empty.
                    - `single_bias_3` subsample of large dataset where a
                      specific puzzle location is always `3`.
                    - `single_bias_3` subsample of large dataset where the
                      first and last cell are identical in both puzzle and
                      solution.
                    """
                )

        if demo_mode:
            data_root = Path("demo")
        else:
            data_root = Path("data")
        file_lengths = dict()
        expander = st.expander("Expand to see list of files and lengths")
        info = ""
        for filename in sorted(data_root.glob("*.txt")):
            key = filename.stem
            file_lengths[key] = get_num_sudokus(filename)
            info += f"- {key} with {file_lengths[key]:,} sudokus.\n"
        if not demo_mode:
            expander.write(info)
            st.write(
                """
                To provide additional data, create a file `data/<foo>.txt`
                in the format of `<puzzle> <solution>\\n`, where 
                - puzzle: raster-scan order of an unfinished sudoku, empty cell
                  should be indicated using `0`.
                - solution: version of puzzle with no empty cells.
                """
            )

        col1, col2 = st.columns([8, 1])
        max_lines = col1.number_input(
            "How many sudokus to load per file? (`0` to disable)",
            min_value=0,
            max_value=max(file_lengths.values()),
            value=100_000,
            disabled=demo_mode,
        )
        max_lines = None if max_lines == 0 else int(max_lines)
        col2.markdown("##") # Align button
        if col2.button("Apply", disabled=demo_mode):
            print("Clearing cache")
            st.cache_data.clear()
            st.session_state.max_lines = max_lines
        else:
            if "max_lines" not in st.session_state:
                st.session_state.max_lines = max_lines
            max_lines = st.session_state.max_lines
        if max_lines != 0:
            st.write(f"Currently `{max_lines:,}` sudokus per file will be loaded")
        else:
            st.write("Currently `all` sudokus per file will be loaded")
        selected = st.multiselect(
            "Datasets to load and compare",
            options=sorted(f.stem for f in data_root.glob("*.txt")),
        )
        if len(selected) == 0:
            st.write("Select some files to continue")
            return

        sudokus = {
            fn: load_dataset(data_root / (fn + ".txt"), max_lines)
            for fn in selected
        }

        st.write(
            "## General Information\n"
            "This section contains some basic information about the loaded "
            "datasets that might be nice to know. In addition there is a random "
            "example from each dataset: Bold numbers are given hints (the "
            "puzzle), all remaining numbers are green (the solution)."
        )
        if len(selected) > 1:
            tabs = st.tabs(selected)
        else:
            tabs = [st.container()]
        for i, key in enumerate(selected):
            with tabs[i]:
                st.markdown(f"### {key.capitalize()}\n")
                col_text, col_sudoku = st.columns([7, 3])
                with col_text:
                    summary = summarize_data(key, sudokus)
                    st.markdown(summary)
                with col_sudoku:
                    idx = random.randint(0, sudokus[key].shape[0]-1)
                    display_sudoku(sudokus[key][idx])

        st.write("## Number of Hints")
        log_scale = st.checkbox("Logarithmic Scale")
    num_hints = get_num_hints_dataframe(selected, sudokus)
    fig = px.bar(num_hints, x="Number of Hints", y="Frequency", color="Source", hover_data=num_hints.columns, log_y=log_scale, barmode="group")
    st.plotly_chart(fig, use_container_width=True)


    with st.columns([0.25, 0.5, 0.25])[1]: # centered layout for text
        st.write("## Spatial Distribution of Numbers")

        num = int(st.number_input("Spatial distribution of number",
                                  value=0, min_value=0, max_value=9))
        display_text = st.checkbox("Display numbers on top of plot")
        col1, col2 = st.columns([1, 6])
        set_p = col1.checkbox("Set p")
        p = col2.number_input(
            "Reject with probability (1/81 ≈ 0.0123)",
            min_value=1e-4,
            max_value=1.,
            value=1/81,
            format="%.4f",
            disabled=not set_p,
        )
        facet_col_wrap = st.number_input(
            "Number of plots per row (hit fullscreen on each plot after "
            "change, so that plotly fixes the layout)",
            min_value=1,
            max_value=len(selected),
            value=min(3, len(selected)),
        )
    spatial_count, spatial_freq = get_spatial_distribution(selected, sudokus)
    zmin = np.min(spatial_freq[:, num])
    zmax = np.max(spatial_freq[:, num])
    spatial_container, pvals_container = st.columns(2)
    pvals = []
    for k, key in enumerate(selected):
        total_count = sudokus[key].shape[0]
        pvals.append(get_pvals(spatial_count[k, num], total_count))
    pvals = np.stack(pvals)
    if set_p:
        pvals = (pvals <= p).astype(float)
    for k in range(math.ceil(spatial_freq.shape[0] / facet_col_wrap)):
        sl = slice(k*facet_col_wrap, (k+1)*facet_col_wrap)
        fig = px.imshow(
            spatial_freq[sl, num],
            labels=dict(color="Frequency"),
            facet_col=0,
            # facet_col_wrap=3, # this breaks the annotation for some reason
            aspect="equal",
            text_auto=".2%" if display_text else False, # pyright: ignore # wrong type hint from plotly
            zmin=zmin,
            zmax=zmax,
        )
        for i, key in enumerate(selected[sl]):
            fig.layout.annotations[i]['text'] = key
        for kx in range(3):
            for ky in range(3):
                x0 = -0.5 + 3*kx
                y0 = -0.5 + 3*ky
                fig.add_shape(
                    type='rect',
                    x0=x0, x1=x0+3, y0=y0, y1=y0+3,
                    xref='x', yref='y',
                    line_color='black',
                    col="all",
                    row="all",
                )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        spatial_container.plotly_chart(fig, use_container_width=True)


        fig = px.imshow(
            pvals[sl],
            labels=dict(color=("Rejected" if set_p else "p")),
            facet_col=0,
            # facet_col_wrap=3, # this breaks the annotation for some reason
            aspect="equal",
            text_auto=".2%" if display_text else False, # pyright: ignore # wrong type hint from plotly
            zmin=0 if set_p else np.min(pvals),
            zmax=1. if set_p else np.max(pvals),
        )
        for i, key in enumerate(selected[sl]):
            fig.layout.annotations[i]['text'] = key
        for kx in range(3):
            for ky in range(3):
                x0 = -0.5 + 3*kx
                y0 = -0.5 + 3*ky
                fig.add_shape(
                    type='rect',
                    x0=x0, x1=x0+3, y0=y0, y1=y0+3,
                    xref='x', yref='y',
                    line_color='black',
                    col="all",
                    row="all",
                )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        pvals_container.plotly_chart(fig, use_container_width=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--demo",
        action="store_true",
    )
    if Path("scripts/streamlit").exists():
        os.chdir(Path("scripts/streamlit"))
    args = parser.parse_args()
    demo_mode = args.demo or not Path("data").exists()
    main(demo_mode)
