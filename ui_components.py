"""
Snap Edit Pro - UI Components Module
Contains all UI rendering functions
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

from image_editor import ImageEditor
from session_manager import get_ar_filters, get_gesture_controller
from template_loader import load_template


def render_header():
    """Render the main header and welcome message"""
    st.title("📸 Snap Edit Pro")
    st.markdown(load_template('header.html'), unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with upload and mode selection"""
    with st.sidebar:
        st.header("🎨 Control Panel")
        
        # Image upload
        st.subheader("📁 Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Check if this is a new file
            if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
                # Load image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if image is not None:
                    st.session_state.image = image
                    st.session_state.edited_image = image.copy()
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.success("✅ Image loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load image. Please try another file.")
            else:
                st.info(f"📷 Current image: {uploaded_file.name}")
        
        # Sample image option
        if st.button("📷 Use Sample Image", key="btn_sample_image"):
            # Create a colorful sample image
            sample = np.zeros((600, 800, 3), dtype=np.uint8)
            
            # Create colored blocks for easy testing
            sample[0:200, 0:400] = [255, 0, 0]      # Red
            sample[0:200, 400:800] = [0, 255, 0]    # Green
            sample[200:400, 0:400] = [0, 0, 255]    # Blue
            sample[200:400, 400:800] = [255, 255, 0] # Yellow
            sample[400:600, 0:400] = [255, 0, 255]  # Magenta
            sample[400:600, 400:800] = [0, 255, 255] # Cyan
            
            st.session_state.image = sample
            st.session_state.edited_image = sample.copy()
            st.session_state.uploaded_file_name = None
            st.success("✅ Sample image loaded!")
            st.rerun()
        
        st.divider()
        
        # Mode selection
        st.subheader("🎯 Select Mode")
        mode = st.radio(
            "Choose editing mode:",
            ["Basic Editing", "AR Filters", "Gesture Control"],
            help="Select the type of editing you want to perform"
        )
    
    return mode


def render_image_preview():
    """Render the before/after image preview"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        original_rgb = cv2.cvtColor(st.session_state.image, cv2.COLOR_BGR2RGB)
        st.image(original_rgb, width='stretch')
        # Display resolution
        height, width = st.session_state.image.shape[:2]
        st.caption(f"Resolution: {width} × {height} pixels")
    
    with col2:
        st.subheader("✨ Edited Image")
        if st.session_state.edited_image is not None:
            edited_rgb = cv2.cvtColor(st.session_state.edited_image, cv2.COLOR_BGR2RGB)
            st.image(edited_rgb, width='stretch')
            # Display resolution
            height, width = st.session_state.edited_image.shape[:2]
            st.caption(f"Resolution: {width} × {height} pixels")
            
            # Reset button directly under edited image
            st.markdown("---")
            col_reset1, col_reset2 = st.columns(2)
            with col_reset1:
                if st.button("🔄 Reset to Original", key="btn_reset_preview"):
                    st.session_state.edited_image = st.session_state.image.copy()
                    st.session_state.active_basic_filter = None
                    st.rerun()
            with col_reset2:
                if st.button("💾 Save as Original", key="btn_save_preview"):
                    st.session_state.image = st.session_state.edited_image.copy()
                    st.session_state.active_basic_filter = None
                    st.success("✅ Saved!")
                    st.rerun()


def render_basic_editing_tab():
    """Render the basic editing tools"""
    st.header("✂️ Basic Editing Tools")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Transform", "🎨 Adjust", "🌈 Filters"])
    
    with tab1:
        render_transform_tools()
    
    with tab2:
        render_adjust_tools()
    
    with tab3:
        render_filter_tools()


def render_transform_tools():
    """Render transform operations (resize, rotate, flip)"""
    st.subheader("Transform Operations")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Resize
        st.write("**Resize Image**")
        scale = st.slider("Scale Factor", 0.1, 2.0, 1.0, 0.1)
        if st.button("Apply Resize", key="btn_resize"):
            # Always apply scale to original image
            st.session_state.edited_image = ImageEditor.resize_image(
                st.session_state.image, scale=scale
            )
            st.rerun()
        
        # Rotate
        st.write("**Rotate Image**")
        angle = st.slider("Rotation Angle", -180, 180, 0, 15)
        if st.button("Apply Rotation", key="btn_rotate"):
            # Always apply rotation to original image
            st.session_state.edited_image = ImageEditor.rotate_image(
                st.session_state.image, angle
            )
            st.rerun()
    
    with col_b:
        # Flip
        st.write("**Flip Image**")
        flip_direction = st.selectbox("Direction", ["horizontal", "vertical"])
        if st.button("Apply Flip", key="btn_flip"):
            st.session_state.edited_image = ImageEditor.flip_image(
                st.session_state.edited_image, flip_direction
            )
            st.rerun()


def render_adjust_tools():
    """Render adjustment tools (brightness, contrast)"""
    st.subheader("Adjust Image Properties")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Brightness
        st.write("**Brightness**")
        brightness = st.slider("Brightness Factor", 0.0, 2.0, 1.0, 0.1)
        if st.button("Apply Brightness", key="btn_brightness"):
            # Always apply brightness to original image
            st.session_state.edited_image = ImageEditor.adjust_brightness(
                st.session_state.image, brightness
            )
            st.rerun()
    
    with col_b:
        # Contrast
        st.write("**Contrast**")
        contrast = st.slider("Contrast Factor", 0.0, 3.0, 1.0, 0.1)
        if st.button("Apply Contrast", key="btn_contrast"):
            # Always apply contrast to original image
            st.session_state.edited_image = ImageEditor.adjust_contrast(
                st.session_state.image, contrast
            )
            st.rerun()


def render_filter_tools():
    """Render filter buttons (toggleable)"""
    st.subheader("Apply Filters")
    
    filters = ["Grayscale", "Sepia", "Blur", "Sharpen", "Edge Detection", 
              "Vintage", "Cool", "Warm"]
    
    # Show active filter info
    if st.session_state.active_basic_filter:
        st.info(f"🎨 Active Filter: **{st.session_state.active_basic_filter}** (Click again to remove)")
    
    cols = st.columns(4)
    for idx, filter_name in enumerate(filters):
        with cols[idx % 4]:
            # Check if this filter is active
            is_active = st.session_state.active_basic_filter == filter_name
            button_label = f"✓ {filter_name}" if is_active else f"🎨 {filter_name}"
            
            if st.button(button_label, key=f"filter_{filter_name.lower().replace(' ', '_')}"):
                if is_active:
                    # Toggle off - return to original
                    st.session_state.edited_image = st.session_state.image.copy()
                    st.session_state.active_basic_filter = None
                    st.success(f"✅ Removed {filter_name} filter!")
                else:
                    # Apply filter to original image
                    filtered = ImageEditor.apply_filter(
                        st.session_state.image, filter_name
                    )
                    st.session_state.edited_image = filtered
                    st.session_state.active_basic_filter = filter_name
                    st.success(f"✅ Applied {filter_name} filter!")
                st.rerun()


def render_reset_tools():
    """Render reset options"""
    st.subheader("Reset Options")
    col_a, col_b = st.columns(2)
    
    with col_a:
        if st.button("🔄 Reset to Original", key="btn_reset"):
            st.session_state.edited_image = st.session_state.image.copy()
            st.session_state.active_basic_filter = None
            st.rerun()
    
    with col_b:
        if st.button("💾 Save Current as Original", key="btn_save_as_original"):
            st.session_state.image = st.session_state.edited_image.copy()
            st.session_state.active_basic_filter = None
            st.success("✅ Saved!")


def render_ar_filters_tab():
    """Render AR filters section with dynamic filter loading"""
    st.header("😎 AR Filters")
    
    st.info("👤 These filters use face detection to apply fun overlays!")
    
    # Get AR filters instance
    ar_filters = get_ar_filters()
    if ar_filters is None:
        st.warning("⚠️ AR Filters are currently unavailable. Please use Basic Editing features.")
        return
    
    # Get all available filters dynamically
    available_filters = ar_filters.get_available_filters()
    
    if not available_filters:
        st.warning("⚠️ No AR filter images found in assets/ar_filters/ directory.")
        st.info("💡 Add PNG images to the assets/ar_filters/ folder to see them here!")
        return
    
    # Display info about loaded filters
    st.success(f"✅ {len(available_filters)} filters loaded from assets/ar_filters/")
    
    # Emoji mapping for common filter names
    emoji_map = {
        'sunglasses': '😎',
        'chef_hat': '👨‍🍳',
        'graduation_hat': '🎓',
        'party_hat': '🎉',
        'dog': '🐶',
        'fox': '🦊',
        'crown': '👑',
        'pumpkin': '🎃',
        'headphone': '🎧',
        'heart': '❤️',
        'cat': '🐱',
        'bunny': '🐰',
        'bear': '🐻',
        'hearts': '😍',
        'sparkles': '✨'
    }
    
    # Create buttons dynamically for all loaded filters
    cols = st.columns(3)
    for idx, filter_name in enumerate(sorted(available_filters)):
        # Get emoji for this filter (or use default)
        emoji = emoji_map.get(filter_name, '🎨')
        
        # Format display name (replace underscores with spaces, capitalize)
        display_name = filter_name.replace('_', ' ').title()
        
        with cols[idx % 3]:
            if st.button(f"{emoji} {display_name}", key=f"ar_filter_{filter_name}"):
                st.session_state.edited_image = ar_filters.apply_filter(
                    st.session_state.edited_image,
                    filter_name
                )
                st.session_state.current_filter = filter_name
                st.rerun()
    
    st.divider()
    
    # Reset button
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Remove Filter", key="btn_remove_ar_filter"):
            st.session_state.edited_image = st.session_state.image.copy()
            st.session_state.current_filter = None
            st.rerun()
    
    with col_b:
        if st.button("💾 Save with Filter", key="btn_save_with_filter"):
            st.session_state.image = st.session_state.edited_image.copy()
            st.success("✅ Saved!")
    
    # Display current filter info
    if st.session_state.current_filter:
        st.success(f"🎭 Current Filter: {st.session_state.current_filter}")



def render_gesture_control_tab():
    """Render gesture control section"""
    st.header("✋ Gesture-Based Music Control")
    
    st.markdown(load_template('gesture_info.html'), unsafe_allow_html=True)
    
    # Music controls
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        if st.button("▶️ Start Music", key="btn_start_music"):
            st.session_state.music_controller.set_volume(st.session_state.volume)
            st.success("🎵 Music started! (Simulated)")
    
    with col_b:
        if st.button("⏸️ Pause Music", key="btn_pause_music"):
            st.session_state.music_controller.pause()
            st.info("⏸️ Music paused")
    
    with col_c:
        if st.button("⏹️ Stop Music", key="btn_stop_music"):
            st.session_state.music_controller.stop()
            st.warning("⏹️ Music stopped")
    
    st.divider()
    
    # Gesture detection
    st.subheader("👋 Detect Hand Gesture")
    
    if st.button("🔍 Analyze Hand Gesture", key="btn_analyze_gesture"):
        gesture_controller = get_gesture_controller()
        if gesture_controller is None:
            st.warning("⚠️ Gesture Control is currently unavailable. Please use Basic Editing features.")
        else:
            volume, annotated_image = gesture_controller.get_volume_from_gesture(
                st.session_state.edited_image
            )
            
            if volume is not None:
                st.session_state.volume = volume
                st.session_state.edited_image = annotated_image
                st.session_state.music_controller.set_volume(volume)
                st.success(f"🎚️ Volume set to: {volume}%")
                st.rerun()
            else:
                st.warning("⚠️ No hand detected in the image. Please show your hand clearly!")
    
    # Manual volume control
    st.subheader("🎚️ Manual Volume Control")
    manual_volume = st.slider("Volume", 0, 100, st.session_state.volume)
    if st.button("Set Volume", key="btn_set_volume"):
        st.session_state.volume = manual_volume
        st.session_state.music_controller.set_volume(manual_volume)
        st.success(f"✅ Volume set to {manual_volume}%")
    
    # Display current volume
    st.metric("Current Volume", f"{st.session_state.volume}%")
    
    # Gesture type detection
    st.divider()
    st.subheader("🤚 Gesture Recognition")
    if st.button("🔍 Identify Gesture", key="btn_identify_gesture"):
        gesture_controller = get_gesture_controller()
        if gesture_controller is None:
            st.warning("⚠️ Gesture Control is currently unavailable. Please use Basic Editing features.")
        else:
            gesture, annotated_image = gesture_controller.detect_gesture_type(
                st.session_state.edited_image
            )
            st.session_state.edited_image = annotated_image
            st.info(f"🤚 Detected Gesture: **{gesture}**")
            st.rerun()


def render_download_section():
    """Render the download section"""
    st.divider()
    st.header("💾 Download Edited Image")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Convert to RGB for download
        download_image = cv2.cvtColor(st.session_state.edited_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(download_image)
        
        # Save to bytes
        buf = io.BytesIO()
        pil_image.save(buf, format='PNG')
        byte_im = buf.getvalue()
        
        st.download_button(
            label="⬇️ Download Edited Image",
            data=byte_im,
            file_name="edited_image.png",
            mime="image/png",
            width='stretch'
        )


def render_welcome_screen():
    """Render welcome screen when no image is loaded"""
    st.markdown(load_template('welcome.html'), unsafe_allow_html=True)


def render_footer():
    """Render the footer"""
    st.divider()
    st.markdown(load_template('footer.html'), unsafe_allow_html=True)
