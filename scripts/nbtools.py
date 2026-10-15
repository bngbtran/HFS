"""Tiện ích sinh notebook: gom cell markdown/code rồi ghi ra .ipynb."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

SETUP = r"""
import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium

from src import cleaning as cl, geo, features as ft, viz

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 80)
viz.set_style()
PROC = ROOT / "data" / "processed"
TABLES = viz.TABLE_DIR
"""


class NB:
    def __init__(self):
        self.cells = []

    def md(self, s):
        self.cells.append(nbf.v4.new_markdown_cell(s.strip()))

    def code(self, s):
        self.cells.append(nbf.v4.new_code_cell(s.strip()))

    def write(self, name):
        nb = nbf.v4.new_notebook()
        nb["cells"] = self.cells
        nb["metadata"]["kernelspec"] = {"name": "dam", "display_name": "Python 3 (dam)",
                                        "language": "python"}
        path = ROOT / "notebooks" / name
        path.write_text(nbf.writes(nb), encoding="utf-8")
        print("written", path.name)
