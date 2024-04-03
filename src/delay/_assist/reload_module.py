"""
    Helps to reload project's module and get its inspections
    w\o reloading working space

    @author: mikhail.galkin
"""

#%% Import libs
import sys
import inspect
import importlib

sys.path.extend([".", "./.", "././.", "../..", "../../.."])
import src/delay

#%% ------------------------------ CONFIG --------------------------------------
import src/delay.config

importlib.reload(src/delay.config)
from src/delay.config import project_dir
from src/delay.config import final_data_folder_path

print(project_dir)
print(final_data_folder_path)

#%% ------------------------------ UTILS ---------------------------------------
import src/delay.utils

importlib.reload(src/delay.utils)
print(inspect.getsource(src/delay.utils.pd_set_options))

#%% -------------------------- UTILS SPECIAL -----------------------------------
