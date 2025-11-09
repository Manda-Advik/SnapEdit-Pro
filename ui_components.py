import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from image_editor import ImageEditor
from session_manager import get_ar_filters, get_gesture_controller, get_emotion_reactor
from template_loader import load_template

def render_header():
    st.title("📸 Snap Edit Pro")
    st.markdown(load_template('header.html'), unsafe_allow_html=True)
def render_sidebar():
    with st.sidebar:
        st.header("🎨 Control Panel")
        
        st.subheader("📁 Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
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
        
        if st.button("📷 Use Sample Image", key="btn_sample_image"):
            sample = np.zeros((600, 800, 3), dtype=np.uint8)
            
            sample[0:200, 0:400] = [255, 0, 0]
            sample[0:200, 400:800] = [0, 255, 0]
            sample[200:400, 0:400] = [0, 0, 255]
            sample[200:400, 400:800] = [255, 255, 0]
            sample[400:600, 0:400] = [255, 0, 255]
            sample[400:600, 400:800] = [0, 255, 255]
            
            st.session_state.image = sample
            st.session_state.edited_image = sample.copy()
            st.session_state.uploaded_file_name = None
            st.success("✅ Sample image loaded!")
            st.rerun()
        
        st.divider()
        
        st.subheader("🎯 Select Mode")
        mode = st.radio(
            "Choose editing mode:",
            ["Basic Editing", "AR Filters", "Gesture Control", "Emotion Reactor"],
            help="Select the type of editing you want to perform"
        )
    
    return mode
def render_image_preview():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        original_rgb = cv2.cvtColor(st.session_state.image, cv2.COLOR_BGR2RGB)
        st.image(original_rgb, width='stretch')
        height, width = st.session_state.image.shape[:2]
        st.caption(f"Resolution: {width} × {height} pixels")
    
    with col2:
        st.subheader("✨ Edited Image")
        if st.session_state.edited_image is not None:
            edited_rgb = cv2.cvtColor(st.session_state.edited_image, cv2.COLOR_BGR2RGB)
            st.image(edited_rgb, width='stretch')
            height, width = st.session_state.edited_image.shape[:2]
            st.caption(f"Resolution: {width} × {height} pixels")
            
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
    st.header("✂️ Basic Editing Tools")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Transform", "🎨 Adjust", "🌈 Filters"])
    
    with tab1:
        render_transform_tools()
    
    with tab2:
        render_adjust_tools()
    
    with tab3:
        render_filter_tools()
def crop_with_mouse():
    """Open OpenCV window to select crop area with mouse drag"""
    try:
        # Work with the current edited image if it exists, otherwise use original
        source_img = st.session_state.edited_image if st.session_state.edited_image is not None else st.session_state.image
        display_img = source_img.copy()
        
        # Show info in Streamlit
        st.info("🖱️ **Crop Window Opening...** Drag your mouse to select the area to crop.")
        
        # Create window with AUTOSIZE flag to make it pop up properly
        window_name = "Select Crop Area - Press ENTER to confirm, ESC to cancel"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        
        # Move window to front (Windows-specific)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        
        # Select ROI (returns x, y, width, height)
        roi = cv2.selectROI(window_name, 
                           display_img, 
                           fromCenter=False,
                           showCrosshair=True)
        
        # Close the window properly
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        x, y, w, h = roi
        
        # Check if valid selection was made
        if w > 0 and h > 0:
            # Crop the current image (edited or original)
            cropped = ImageEditor.crop_image(source_img, x, y, w, h)
            
            # Update only the edited image, keep original unchanged
            st.session_state.edited_image = cropped
            
            st.success(f"✅ Image cropped to {w}×{h} pixels!")
            st.rerun()
        else:
            st.warning("❌ Crop cancelled - No area selected")
            
    except Exception as e:
        st.error(f"❌ Error during crop: {str(e)}")
        cv2.destroyAllWindows()
        cv2.waitKey(1)

def render_crop_tool():
    """Interactive crop tool with mouse drag option"""
    st.write("**Crop Image**")
    h, w = st.session_state.image.shape[:2]
    
    # Add mouse-based crop button at the top
    st.markdown("### 🖱️ Mouse Drag to Crop")
    st.caption("Click the button below to open a window where you can drag with your mouse to select the crop area")
    
    if st.button("🖱️ Open Crop Tool", key="btn_mouse_crop", width="stretch"):
        crop_with_mouse()
    
    st.info("📌 **How to use:**\n- A new window will open with your image\n- Drag with your mouse to select the crop area\n- Press **ENTER** or **SPACE** to apply the crop\n- Press **ESC** to cancel")

def render_transform_tools():
    st.subheader("Transform Operations")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Resize Image**")
        scale = st.slider("Scale Factor", 0.1, 2.0, 1.0, 0.1)
        if st.button("Apply Resize", key="btn_resize"):
            st.session_state.edited_image = ImageEditor.resize_image(
                st.session_state.image, scale=scale
            )
            st.rerun()
        
        st.write("**Rotate Image**")
        angle = st.slider("Rotation Angle", -180, 180, 0, 15)
        if st.button("Apply Rotation", key="btn_rotate"):
            st.session_state.edited_image = ImageEditor.rotate_image(
                st.session_state.image, angle
            )
            st.rerun()
        
        render_crop_tool()
    
    with col_b:
        st.write("**Flip Image**")
        flip_direction = st.selectbox("Direction", ["horizontal", "vertical"])
        if st.button("Apply Flip", key="btn_flip"):
            st.session_state.edited_image = ImageEditor.flip_image(
                st.session_state.edited_image, flip_direction
            )
            st.rerun()

def render_adjust_tools():
    st.subheader("Adjust Image Properties")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("**Brightness**")
        brightness = st.slider("Brightness Factor", 0.0, 2.0, 1.0, 0.1)
        if st.button("Apply Brightness", key="btn_brightness"):
            st.session_state.edited_image = ImageEditor.adjust_brightness(
                st.session_state.image, brightness
            )
            st.rerun()
    
    with col_b:
        st.write("**Contrast**")
        contrast = st.slider("Contrast Factor", 0.0, 3.0, 1.0, 0.1)
        if st.button("Apply Contrast", key="btn_contrast"):
            st.session_state.edited_image = ImageEditor.adjust_contrast(
                st.session_state.image, contrast
            )
            st.rerun()
def render_filter_tools():
    st.subheader("Apply Filters")
    
    filters = ["Grayscale", "Sepia", "Blur", "Sharpen", "Edge Detection", 
              "Vintage", "Cool", "Warm"]
    
    if st.session_state.active_basic_filter:
        st.info(f"🎨 Active Filter: **{st.session_state.active_basic_filter}** (Click again to remove)")
    
    cols = st.columns(4)
    for idx, filter_name in enumerate(filters):
        with cols[idx % 4]:
            is_active = st.session_state.active_basic_filter == filter_name
            button_label = f"✓ {filter_name}" if is_active else f"🎨 {filter_name}"
            
            if st.button(button_label, key=f"filter_{filter_name.lower().replace(' ', '_')}"):
                if is_active:
                    st.session_state.edited_image = st.session_state.image.copy()
                    st.session_state.active_basic_filter = None
                    st.success(f"✅ Removed {filter_name} filter!")
                else:
                    filtered = ImageEditor.apply_filter(
                        st.session_state.image, filter_name
                    )
                    st.session_state.edited_image = filtered
                    st.session_state.active_basic_filter = filter_name
                    st.success(f"✅ Applied {filter_name} filter!")
                st.rerun()
def render_reset_tools():
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
    st.header("😎 AR Filters")
    
    st.info("👤 These filters use face detection to apply fun overlays!")
    
    ar_filters = get_ar_filters()
    if ar_filters is None:
        st.warning("⚠️ AR Filters are currently unavailable. Please use Basic Editing features.")
        return
    
    available_filters = ar_filters.get_available_filters()
    
    if not available_filters:
        st.warning("⚠️ No AR filter images found in assets/ar_filters/ directory.")
        st.info("💡 Add PNG images to the assets/ar_filters/ folder to see them here!")
        return
    
    st.success(f"✅ {len(available_filters)} filters loaded from assets/ar_filters/")
    
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
    
    cols = st.columns(3)
    for idx, filter_name in enumerate(sorted(available_filters)):
        emoji = emoji_map.get(filter_name, '🎨')
        
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
    
    if st.session_state.current_filter:
        st.success(f"🎭 Current Filter: {st.session_state.current_filter}")
def render_gesture_control_tab():
    st.header("🎮 Live Gesture Control - Music & Frames")
    
    st.markdown("""
    <div style='background-color: #1e1e1e; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: #00d4ff;'>🖐️ Hand Gesture Controls</h3>
        <p style='color: #ffaa00; font-size: 18px; font-weight: bold;'>👌 PINCH your thumb and index finger together, then move!</p>
        <ul style='color: #ffffff; font-size: 16px;'>
            <li><b>PINCH + Move hand LEFT ←</b> : Previous song & frame (with transparent overlay)</li>
            <li><b>PINCH + Move hand RIGHT →</b> : Next song & frame (with transparent overlay)</li>
            <li><b>PINCH + Move hand UP ↑</b> : Increase volume</li>
            <li><b>PINCH + Move hand DOWN ↓</b> : Decrease volume</li>
        </ul>
        <p style='color: #888888; font-size: 14px;'>💡 Tip: Keep your fingers pinched together while moving for best results!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for gesture control
    if 'gesture_active' not in st.session_state:
        st.session_state.gesture_active = False
    if 'gesture_controller' not in st.session_state:
        st.session_state.gesture_controller = None
    
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🚀 Start Gesture Control", key="btn_start_gesture", width="stretch"):
            st.session_state.gesture_active = True
            if st.session_state.gesture_controller is None:
                from gesture_control import GestureController
                st.session_state.gesture_controller = GestureController()
    
    with col2:
        if st.button("⏸️ Stop Gesture Control", key="btn_stop_gesture", width="stretch"):
            st.session_state.gesture_active = False
            # Stop music when gesture control stops
            if st.session_state.gesture_controller:
                import pygame
                pygame.mixer.music.stop()
    
    with col3:
        # System info
        frames_dir = "assets/frames"
        music_dir = "assets/music"
        if os.path.exists(frames_dir):
            frame_count = len([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        else:
            frame_count = 0
        if os.path.exists(music_dir):
            music_count = len([f for f in os.listdir(music_dir) if f.lower().endswith(('.mp3', '.wav', '.ogg'))])
        else:
            music_count = 0
        st.metric("📊 Frames/Songs", f"{frame_count}/{music_count}")
    
    st.divider()
    
    # Create stable placeholders before video starts
    st.subheader("🎥 Live Webcam with Transparent Frame Overlay")
    
    # Create two columns: video on left, info on right
    col_video, col_info = st.columns([2, 1])
    
    with col_video:
        # Standard video placeholder
        video_placeholder = st.empty()
    
    with col_info:
        st.markdown("### 📊 Control Info")
        info_placeholder1 = st.empty()
        info_placeholder2 = st.empty()
        info_placeholder3 = st.empty()
        st.markdown("---")
        st.markdown("### 🎮 Instructions")
        st.markdown("""
        **👌 Pinch your fingers together:**
        - 🟢 **Green** = Pinching (Active)
        - 🟡 **Yellow** = Not pinching
        
        **While pinching, move hand:**
        - **← LEFT**: Previous song/frame
        - **→ RIGHT**: Next song/frame
        - **↑ UP**: Increase volume
        - **↓ DOWN**: Decrease volume
        """)
    
    # Show placeholder image before video starts
    if not st.session_state.gesture_active:
        with col_video:
            # Create a placeholder image matching webcam resolution (640x480)
            placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
            placeholder_img[:] = (30, 30, 30)  # Dark gray background
            
            # Add text to placeholder (centered)
            cv2.putText(placeholder_img, "Gesture Control Ready", (120, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(placeholder_img, "Click 'Start Gesture Control' to begin", (80, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Add icon/visual element (centered)
            cv2.circle(placeholder_img, (320, 140), 40, (0, 255, 255), -1)
            cv2.putText(placeholder_img, "?", (305, 155), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 3)
            
            video_placeholder.image(placeholder_img, channels="BGR", width=640)
        
        # Set default info
        info_placeholder1.metric("🖼️ Frame", "0/0")
        info_placeholder2.metric("🎵 Song", "0/0")
        info_placeholder3.metric("🔊 Volume", "50%")
    
    # Live video feed with gesture control
    if st.session_state.gesture_active and st.session_state.gesture_controller:
        # Initialize webcam in session state if needed
        if 'webcam_cap' not in st.session_state or st.session_state.webcam_cap is None:
            st.session_state.webcam_cap = cv2.VideoCapture(0)
            # Set webcam to capture at 640x480 resolution (VGA standard)
            st.session_state.webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            st.session_state.webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            st.session_state.webcam_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        cap = st.session_state.webcam_cap
        
        if not cap.isOpened():
            st.error("❌ Could not open webcam")
            st.session_state.gesture_active = False
            st.session_state.webcam_cap = None
        else:
            # Capture and process frames in a continuous loop
            import time
            frame_count = 0
            
            while st.session_state.gesture_active and frame_count < 30:  # Process 30 frames per page load
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame with gesture detection (without text overlay)
                controller = st.session_state.gesture_controller
                result_frame, frame_overlay = controller.process_frame_with_gestures(frame, show_text=False)
                
                # Overlay transparent frame on camera feed
                if frame_overlay is not None:
                    # Resize overlay to match camera frame
                    h, w = result_frame.shape[:2]
                    frame_overlay_resized = cv2.resize(frame_overlay, (w, h))
                    
                    # Check if overlay has alpha channel
                    if frame_overlay_resized.shape[2] == 4:
                        # Extract alpha channel
                        overlay_bgr = frame_overlay_resized[:, :, :3]
                        alpha = frame_overlay_resized[:, :, 3] / 255.0
                        
                        # Blend with camera frame using vectorized operations
                        alpha_3d = alpha[:, :, np.newaxis]
                        result_frame = (alpha_3d * overlay_bgr + (1 - alpha_3d) * result_frame).astype(np.uint8)
                    else:
                        # If no alpha, blend with 50% transparency
                        result_frame = cv2.addWeighted(result_frame, 0.7, frame_overlay_resized, 0.3, 0)
                
                # Convert BGR to RGB for Streamlit
                result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                # Update video in the same placeholder with fixed width
                video_placeholder.image(result_frame_rgb, channels="RGB", width=640)
                
                # Update info displays less frequently
                if frame_count % 5 == 0:
                    info_placeholder1.metric("🖼️ Frame", f"{controller.current_frame_index + 1}/{len(controller.frames)}")
                    info_placeholder2.metric("🎵 Song", f"{controller.current_music_index + 1}/{len(controller.music_files)}")
                    info_placeholder3.metric("🔊 Volume", f"{controller.volume}%")
                
                frame_count += 1
                time.sleep(0.033)  # ~30 fps
            
            # Rerun after processing batch of frames
            if st.session_state.gesture_active:
                st.rerun()
    else:
        # Clean up webcam and music if stopping
        if 'webcam_cap' in st.session_state and st.session_state.webcam_cap is not None:
            st.session_state.webcam_cap.release()
            st.session_state.webcam_cap = None
        
        # Ensure music is stopped when not active
        if st.session_state.gesture_controller:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
    
def render_download_section():
    st.divider()
    st.header("💾 Download Edited Image")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        download_image = cv2.cvtColor(st.session_state.edited_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(download_image)
        
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
    st.markdown(load_template('welcome.html'), unsafe_allow_html=True)

def render_emotion_reactor_tab():
    st.header("😊 Emotion Reactor - Live Video")
    st.markdown("Real-time emotion detection from your webcam with live emoji reactions!")
    
    emotion_reactor = get_emotion_reactor()
    
    if emotion_reactor is None:
        st.warning("⚠️ Emotion Reactor is currently unavailable.")
        return
    
    if not emotion_reactor.has_mediapipe:
        st.info("ℹ️ Running with OpenCV Haar Cascades (MediaPipe not available)")
        st.markdown("**Detection capabilities:**")
        st.markdown("- 😊 **Smiling** - Detects smile")
        st.markdown("- 😐 **Straight Face** - No smile detected")
        st.markdown("- ⚠️ **Hands Up detection unavailable** (requires MediaPipe)")
    else:
        st.success("✅ Running with MediaPipe (Full features enabled)")
    
    available_reactions = emotion_reactor.get_available_reactions()
    
    if len(available_reactions) == 0:
        st.warning("⚠️ No reaction videos found in assets/cr/ folder.")
        return
    
    st.info(f"📦 {len(available_reactions)} reactions available: {', '.join(available_reactions)}")
    
    if 'reactor_running' not in st.session_state:
        st.session_state.reactor_running = False
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if not st.session_state.reactor_running:
            if st.button("🎥 Start Emotion Reactor", key="btn_start_reactor"):
                st.session_state.reactor_running = True
                st.session_state.video_capture = cv2.VideoCapture(0)
                st.session_state.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                st.session_state.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                st.session_state.current_emotion = None
                st.rerun()
        else:
            if st.button("⏹️ Stop Emotion Reactor", key="btn_stop_reactor"):
                st.session_state.reactor_running = False
                if 'video_capture' in st.session_state and st.session_state.video_capture is not None:
                    st.session_state.video_capture.release()
                    st.session_state.video_capture = None
                st.rerun()
    
    with col2:
        if st.session_state.reactor_running:
            st.markdown("**Status:** 🟢 **Running**")
        else:
            st.markdown("**Status:** 🔴 **Stopped**")
    
    if st.session_state.reactor_running:
        st.divider()
        
        # Create layout with better spacing
        video_col1, video_col2 = st.columns(2, gap="medium")
        
        # Get webcam frame
        cap = st.session_state.video_capture
        
        if cap and cap.isOpened():
            success, frame = cap.read()
            if success:
                frame = cv2.flip(frame, 1)
                current_emotion, annotated_frame = emotion_reactor.process_video_frame(frame)
                
                # Display camera feed
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                cam_image = Image.fromarray(annotated_rgb)
                
                with video_col1:
                    st.markdown("#### 📹 Live Camera Feed")
                    st.image(cam_image, width='stretch')
                
                # Display video reaction
                with video_col2:
                    st.markdown("#### 🎭 Emotion Reaction")
                    
                    # Always update video path based on current emotion
                    video_path = emotion_reactor.get_reaction_video(current_emotion)
                    
                    if video_path and os.path.exists(video_path):
                        # Use HTML video tag to hide controls and autoplay
                        # Add a unique key based on emotion to force reload when emotion changes
                        with open(video_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                        
                        import base64
                        video_base64 = base64.b64encode(video_bytes).decode()
                        
                        video_html = f"""
                        <video key="{current_emotion}" width="100%" height="auto" autoplay loop muted playsinline style="border-radius: 10px;">
                            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                        </video>
                        """
                        st.markdown(video_html, unsafe_allow_html=True)
                    else:
                        st.warning(f"No video for {current_emotion}")
                
                # Update current emotion in session state
                st.session_state.current_emotion = current_emotion
                
                # Show emotion info with better styling
                emotion_emoji = {"smiling": "😊", "straight_face": "😐", "hands_up": "🙌"}
                emotion_name = current_emotion.replace('_', ' ').title()
                
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                            padding: 15px; 
                            border-radius: 10px; 
                            text-align: center; 
                            color: white; 
                            font-size: 20px; 
                            font-weight: bold; 
                            margin-top: 20px;">
                    Detected Emotion: {emotion_emoji.get(current_emotion, '😐')} {emotion_name}
                </div>
                """, unsafe_allow_html=True)
                
                # Auto-refresh
                import time
                time.sleep(0.03)
                st.rerun()
        else:
            st.error("❌ Could not open webcam")
    else:
        st.divider()
        st.subheader("📋 How It Works")
        
        if emotion_reactor.has_mediapipe:
            st.markdown("""
            **With MediaPipe (Full Features):**
            - **Smiling 😊**: Detects when mouth is open/smiling
            - **Straight Face 😐**: Detects neutral facial expression  
            - **Hands Up 🙌**: Detects when hands are raised above shoulders
            """)
        else:
            st.markdown("""
            **With OpenCV Haar Cascades (Basic Features):**
            - **Smiling 😊**: Detects smile using Haar Cascade classifier
            - **Straight Face 😐**: No smile detected
            - **Note**: Hand detection not available without MediaPipe
            """)
        
        st.markdown("""
        **Side-by-side display:**
        1. **Camera Feed**: Your live webcam with emotion detection overlay
        2. **Emoji Reaction**: Corresponding emoji/GIF reaction in real-time
        
        Click **Start Emotion Reactor** to begin!
        """)

def render_footer():
    st.divider()
    st.markdown(load_template('footer.html'), unsafe_allow_html=True)