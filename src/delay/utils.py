""" Contains the functions used across the project.

    @author: mikhail.galkin
"""

#%% Import needed python libraryies and project config info
import time
import numpy as np
import pandas as pd

from IPython.display import display
from pprint import pprint

# ------------------------------------------------------------------------------
# ----------------------------- P A R A M E T E R S ----------------------------
# ------------------------------------------------------------------------------
#%% Set up: Pandas options
def pd_set_options():
    """Set parameters for PANDAS to InteractiveWindow"""

    display_settings = {
        "max_columns": 40,
        "max_rows": 400,  # default: 60
        "width": 500,
        "max_info_columns": 500,
        "expand_frame_repr": True,  # Wrap to multiple pages
        "float_format": lambda x: "%.5f" % x,
        "pprint_nest_depth": 4,
        "precision": 4,
        "show_dimensions": True,
    }
    print("\nPandas options established are:")
    for op, value in display_settings.items():
        pd.set_option(f"display.{op}", value)
        option = pd.get_option(f"display.{op}")
        print(f"\tdisplay.{op}: {option}")


#%% Set up: Reset Pandas options
def pd_reset_options():
    """Set parameters for PANDAS to InteractiveWindow"""
    pd.reset_option("all")
    print("Pandas all options re-established.")


#%% Set up: Matplotlib params
def matlotlib_set_params():
    """Set parameters for MATPLOTLIB to InteractiveWindow"""
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    plt.style.use(["ggplot"])
    rcParams["figure.figsize"] = 10, 8
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["xtick.labelsize"] = 12
    rcParams["ytick.labelsize"] = 12


# ------------------------------------------------------------------------------
# -------------------------- U T I L I T I E S ---------------------------------
# ------------------------------------------------------------------------------
def timing(tic):
    min, sec = divmod(time.time() - tic, 60)
    return f"for: {int(min)}min {int(sec)}sec"
