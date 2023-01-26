"""The booting up script."""
import streamlit.web.cli


if __name__ == "__main__":
    streamlit.web.cli._main_run_clExplicit('interface.py', 'streamlit run')
