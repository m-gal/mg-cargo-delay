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
import ftdelay

#%% ------------------------------ CONFIG --------------------------------------
import ftdelay.config

importlib.reload(ftdelay.config)
from ftdelay.config import project_dir
from ftdelay.config import final_data_folder_path

print(project_dir)
print(final_data_folder_path)

#%% ------------------------------ UTILS ---------------------------------------
import ftdelay.utils

importlib.reload(ftdelay.utils)
print(inspect.getsource(ftdelay.utils.pd_set_options))

#%% -------------------------- UTILS SPECIAL -----------------------------------
