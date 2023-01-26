"""Hook script for streamlit."""
from PyInstaller.utils.hooks import copy_metadata

datas = copy_metadata('streamlit')
