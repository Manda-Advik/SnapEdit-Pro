import streamlit as st
import sys

if 'ar_filters_custom' in sys.modules:
    del sys.modules['ar_filters_custom']
if 'ar_filters' in st.session_state:
    st.session_state.ar_filters = None

from styles import get_dark_theme_css
from session_manager import initialize_session_state
from ui_components import (
    render_header,
    render_sidebar,
    render_image_preview,
    render_basic_editing_tab,
    render_ar_filters_tab,
    render_gesture_control_tab,
    render_download_section,
    render_welcome_screen,
    render_footer
)

st.set_page_config(
    page_title="Snap Edit Pro",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(get_dark_theme_css(), unsafe_allow_html=True)

initialize_session_state()

render_header()

mode = render_sidebar()

if st.session_state.image is not None:
    render_image_preview()
    st.divider()
    
    if mode == "Basic Editing":
        render_basic_editing_tab()
    elif mode == "AR Filters":
        render_ar_filters_tab()
    elif mode == "Gesture Control":
        render_gesture_control_tab()
    
    render_download_section()
else:
    render_welcome_screen()

render_footer()
