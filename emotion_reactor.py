import cv2
import numpy as np
import os
from pathlib import Path

class EmotionReactor:
    
    def __init__(self, assets_dir="assets/cr"):
        self.assets_dir = assets_dir
        self.videos = {}
        self.video_frames = {}
        self._load_videos()
        self._load_video_frames()
        
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_hands = mp.solutions.hands
            self.has_mediapipe = True
            print("✓ MediaPipe loaded successfully")
        except ImportError:
            self.has_mediapipe = False
            print("⚠️ MediaPipe not available - emotion detection will not work")
    
    def _load_videos(self):
        if not os.path.exists(self.assets_dir):
            print(f"⚠️ Assets directory not found: {self.assets_dir}")
            return
        
        video_files = {}
        for filename in os.listdir(self.assets_dir):
            if filename.lower().endswith('.mp4'):
                filepath = os.path.join(self.assets_dir, filename)
                video_files[filename.lower()] = filepath
        
        if 'smiling.mp4' in video_files:
            self.videos['smiling'] = video_files['smiling.mp4']
        
        if 'handsonface.mp4' in video_files:
            self.videos['straight_face'] = video_files['handsonface.mp4']
        
        if 'handsup.mp4' in video_files:
            self.videos['hands_up'] = video_files['handsup.mp4']
        
        if 'two.mp4' in video_files:
            self.videos['peace_sign'] = video_files['two.mp4']
        
        if 'hello.mp4' in video_files:
            self.videos['waving'] = video_files['hello.mp4']
        
        if 'love.mp4' in video_files:
            self.videos['heart_gesture'] = video_files['love.mp4']
        
        if 'dab.mp4' in video_files:
            self.videos['dab'] = video_files['dab.mp4']
        
        if 'thumbsup.mp4' in video_files:
            self.videos['thumbs_up'] = video_files['thumbsup.mp4']
        
        print(f"✓ Loaded {len(self.videos)} reaction videos: {list(self.videos.keys())}")
        for emotion, path in self.videos.items():
            print(f"  - {emotion}: {os.path.basename(path)}")
    
    def _load_video_frames(self):
        if not os.path.exists(self.assets_dir):
            return
        
        for emotion, video_path in self.videos.items():
            try:
                cap = cv2.VideoCapture(video_path)
                frames = []
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                
                cap.release()
                
                if len(frames) > 0:
                    self.video_frames[emotion] = frames
                    print(f"✓ Loaded {emotion} video ({len(frames)} frames)")
                else:
                    print(f"⚠️ No frames loaded for {emotion}")
            except Exception as e:
                print(f"✗ Error loading {emotion}: {e}")
    
    def get_available_reactions(self):
        return list(self.videos.keys())
    
    def get_reaction_video(self, emotion):
        if emotion in self.videos:
            return self.videos[emotion]
        return None
    
    def get_total_frames(self, emotion):
        if emotion in self.video_frames:
            return len(self.video_frames[emotion])
        return 0
    
    def get_video_frame(self, emotion, target_size=(720, 450), frame_index=0):
        if emotion in self.video_frames:
            frames = self.video_frames[emotion]
            
            if len(frames) > 0:
                idx = frame_index % len(frames)
                video_frame = frames[idx].copy()  # Make a copy to avoid reference issues
                return cv2.resize(video_frame, target_size)
        
        blank = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        cv2.putText(blank, f"{emotion}", (target_size[0]//4, target_size[1]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        return blank
    

    
    def _detect_hand_gesture(self, hand_landmarks):
        landmarks = hand_landmarks.landmark
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        index_base = landmarks[5]
        middle_base = landmarks[9]
        ring_base = landmarks[13]
        pinky_base = landmarks[17]
        thumb_base = landmarks[2]
        wrist = landmarks[0]
        
        def is_finger_extended(tip, base):
            return tip.y < base.y - 0.02
        
        def is_thumb_extended(tip, base, wrist):
            horizontal_ext = abs(tip.x - base.x) > 0.04
            vertical_ext = tip.y < wrist.y - 0.02
            return horizontal_ext or vertical_ext
        
        extended_fingers = []
        if is_finger_extended(index_tip, index_base):
            extended_fingers.append('index')
        if is_finger_extended(middle_tip, middle_base):
            extended_fingers.append('middle')
        if is_finger_extended(ring_tip, ring_base):
            extended_fingers.append('ring')
        if is_finger_extended(pinky_tip, pinky_base):
            extended_fingers.append('pinky')
        
        thumb_extended = is_thumb_extended(thumb_tip, thumb_base, wrist)
        
        if len(extended_fingers) >= 3 and thumb_extended:
            return "waving"
        
        if len(extended_fingers) == 2 and 'index' in extended_fingers and 'middle' in extended_fingers:
            return "peace_sign"
        
        thumb_index_distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
        if thumb_index_distance < 0.08 and len(extended_fingers) <= 1:
            return "heart_gesture"
        
        return None
    
    def process_video_frame(self, frame):
        if not self.has_mediapipe:
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, 'MediaPipe not available', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return "straight_face", annotated_frame
        
        try:
            with self.mp_pose.Pose(min_detection_confidence=0.5) as pose, \
                 self.mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh, \
                 self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                current_state = "straight_face"
                annotated_frame = frame.copy()
                
                results_hands = hands.process(image_rgb)
                hand_gesture = None
                detected_gestures = []
                hands_visible = False
                
                if results_hands.multi_hand_landmarks:
                    hands_visible = True
                    import mediapipe as mp
                    mp_drawing = mp.solutions.drawing_utils
                    mp_drawing_styles = mp.solutions.drawing_styles
                    
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        gesture = self._detect_hand_gesture(hand_landmarks)
                        if gesture:
                            detected_gestures.append(gesture)
                    
                    if detected_gestures.count("waving") >= 2:
                        hand_gesture = "hands_up"
                        current_state = "hands_up"
                    elif len(detected_gestures) > 0:
                        hand_gesture = detected_gestures[0]
                        current_state = hand_gesture
                
                if not hand_gesture:
                    results_pose = pose.process(image_rgb)
                    if results_pose.pose_landmarks:
                        if 'mp_drawing' not in locals():
                            import mediapipe as mp
                            mp_drawing = mp.solutions.drawing_utils
                            mp_drawing_styles = mp.solutions.drawing_styles
                        
                        landmarks = results_pose.pose_landmarks.landmark
                        
                        mp_drawing.draw_landmarks(
                            image=annotated_frame,
                            landmark_list=results_pose.pose_landmarks,
                            connections=self.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                        
                        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
                        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
                        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                        left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE]
                        right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE]

                        left_hand_to_face = ((left_wrist.x - nose.x)**2 + (left_wrist.y - nose.y)**2)**0.5
                        right_hand_to_face = ((right_wrist.x - nose.x)**2 + (right_wrist.y - nose.y)**2)**0.5
                        
                        if left_hand_to_face < 0.15 and right_hand_to_face < 0.15:
                            current_state = "straight_face"
                        
                        if (left_wrist.y < left_shoulder.y) and (right_wrist.y < right_shoulder.y):
                            current_state = "hands_up"
                        
                        elif (left_wrist.x > right_shoulder.x - 0.1 and left_elbow.y < left_shoulder.y + 0.1) or \
                             (right_wrist.x < left_shoulder.x + 0.1 and right_elbow.y < right_shoulder.y + 0.1):
                            current_state = "dab"
                
                if current_state == "straight_face":
                    results_face = face_mesh.process(image_rgb)
                    if results_face.multi_face_landmarks:
                        if 'mp_drawing' not in locals():
                            import mediapipe as mp
                            mp_drawing = mp.solutions.drawing_utils
                            mp_drawing_styles = mp.solutions.drawing_styles
                        
                        for face_landmarks in results_face.multi_face_landmarks:
                            mp_drawing.draw_landmarks(
                                image=annotated_frame,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                            )
                            
                            mp_drawing.draw_landmarks(
                                image=annotated_frame,
                                landmark_list=face_landmarks,
                                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                            )
                            
                            left_mouth_corner = face_landmarks.landmark[61]
                            right_mouth_corner = face_landmarks.landmark[291]
                            upper_lip = face_landmarks.landmark[13]
                            lower_lip = face_landmarks.landmark[14]
                            
                            mouth_width = abs(right_mouth_corner.x - left_mouth_corner.x)
                            mouth_height = abs(lower_lip.y - upper_lip.y)
                            
                            if mouth_height > 0.001:
                                mouth_ratio = mouth_width / mouth_height
                            else:
                                mouth_ratio = 0
                            
                            if mouth_ratio > 2.5:
                                current_state = "smiling"
                            
                            cv2.putText(annotated_frame, f'Mouth Ratio: {mouth_ratio:.2f}', (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.putText(annotated_frame, f'STATE: {current_state.upper()}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                return current_state, annotated_frame
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            return "straight_face", frame
    
    def apply_reaction_overlay(self, image, emotion):
        gif_path = self.get_reaction_gif(emotion)
        if gif_path is None:
            return image
        
        try:
            cap = cv2.VideoCapture(gif_path)
            ret, gif_frame = cap.read()
            cap.release()
            
            if ret:
                h, w = image.shape[:2]
                overlay_size = min(w, h) // 3
                gif_resized = cv2.resize(gif_frame, (overlay_size, overlay_size))
                
                x = w - overlay_size - 20
                y = 20
                
                result = image.copy()
                result[y:y+overlay_size, x:x+overlay_size] = gif_resized
                return result
        except Exception as e:
            print(f"Error applying reaction overlay: {e}")
        
        return image
