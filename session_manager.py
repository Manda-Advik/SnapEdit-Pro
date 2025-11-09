import streamlit as st
from music_controller import MusicController

def initialize_session_state():
    if 'image' not in st.session_state:
        st.session_state.image = None
    if 'edited_image' not in st.session_state:
        st.session_state.edited_image = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'ar_filters' not in st.session_state:
        st.session_state.ar_filters = None
    if 'gesture_controller' not in st.session_state:
        st.session_state.gesture_controller = None
    if 'emotion_reactor' not in st.session_state:
        st.session_state.emotion_reactor = None
    if 'music_controller' not in st.session_state:
        st.session_state.music_controller = MusicController()
    if 'current_filter' not in st.session_state:
        st.session_state.current_filter = None
    if 'active_basic_filter' not in st.session_state:
        st.session_state.active_basic_filter = None
    if 'volume' not in st.session_state:
        st.session_state.volume = 50

def get_ar_filters():
    if st.session_state.ar_filters is None:
        try:
            from ar_filters_custom import ARFiltersCustom
            st.session_state.ar_filters = ARFiltersCustom()
            st.success("✅ AR Filters loaded successfully (Custom Overlay mode)")
        except ImportError:
            try:
                from ar_filters_simple import ARFiltersSimple
                st.session_state.ar_filters = ARFiltersSimple()
                st.success("✅ AR Filters loaded successfully (Programmatic mode)")
            except Exception as e:
                st.error(f"⚠️ AR Filters unavailable: {str(e)[:100]}")
                st.info("💡 You can still use Basic Editing features!")
                return None
        except Exception as e:
            st.error(f"⚠️ AR Filters unavailable: {str(e)[:100]}")
            st.info("💡 You can still use Basic Editing features!")
            return None
    return st.session_state.ar_filters

def get_gesture_controller():
    if st.session_state.gesture_controller is None:
        try:
            from gesture_control import GestureController
            st.session_state.gesture_controller = GestureController()
        except Exception as e:
            st.error(f"⚠️ Gesture Control unavailable due to MediaPipe compatibility issue: {str(e)[:100]}")
            st.info("💡 You can still use Basic Editing features! Gesture control requires compatible MediaPipe version.")
            return None
    return st.session_state.gesture_controller

def get_emotion_reactor():
    if st.session_state.emotion_reactor is None:
        try:
            from emotion_reactor import EmotionReactor
            st.session_state.emotion_reactor = EmotionReactor()
            return st.session_state.emotion_reactor
        except Exception as e:
            st.error(f"⚠️ Emotion Reactor unavailable: {str(e)[:100]}")
            return None
    return st.session_state.emotion_reactor
