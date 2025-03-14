import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque
import sys
import traceback
import os  # For file operations

# Add error handling to see why the program might be closing
try:
    # Initialize MediaPipe Hand solutions
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Initialize MediaPipe Face detection
    mp_face_detection = mp.solutions.face_detection
    
    # Configure PyAutoGUI settings
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0  # Remove default pause between pyautogui commands
    
    # Print some debug info
    print("PyAutoGUI screen size:", pyautogui.size())
    print("MediaPipe version:", mp.__version__)
    print("OpenCV version:", cv2.__version__)
    
    # Initialize variables for gesture detection
    prev_finger_pos = None
    swipe_threshold = 40  # Reduced threshold - easier to trigger swipes
    swipe_cooldown = 0.1  # Reduced cooldown time between swipes
    last_swipe_time = 0
    is_dragging = False   # Track if we're currently dragging (slicing)
    
    # Variables for smooth cursor movement
    cursor_smoothing = 0.3  # Lower = more responsive, Higher = smoother (between 0 and 1)
    prev_cursor_x, prev_cursor_y = 0, 0
    dead_zone = 3  # Ignore very small movements (pixels in webcam space)
    
    # Variables for swipe effect
    finger_trail = deque(maxlen=20)  # Stores recent finger positions
    is_swiping = False
    swipe_color = (0, 255, 255)  # Yellow swipe trail
    swipe_thickness = 5
    
    # Variables for face recognition using MediaPipe
    registered_face_center = None  # Store the center position of registered face
    registered_face_area = 0       # Store the area of registered face
    face_detected = False
    registered_user = False
    face_detection_cooldown = 0    # Counter to run face detection less frequently
    face_distance_threshold = 150  # Increased from 100 for better distance detection
    face_size_ratio_min = 0.5      # More lenient size ratio (was 0.7)
    face_size_ratio_max = 1.5      # More lenient size ratio (was 1.3)
    
    # Variables for saved face image
    face_image_path = "registered_face.jpg"
    saved_face_image = None
    face_save_margin = 20  # Extra pixels around the face for better context
    
    # Variables for peace sign detection
    peace_sign_frames = 0
    required_peace_frames = 15  # Hold peace sign for ~0.5 seconds to register
    peace_sign_active = False
    
    # Variables for two-step authentication
    auth_step = 0  # 0: not started, 1: peace sign complete, 2: fist complete
    fist_sign_frames = 0
    required_fist_frames = 15  # Hold fist for ~0.5 seconds to complete registration
    auth_cooldown = 0  # Cooldown between steps
    
    # Check if we have a previously saved face
    if os.path.exists(face_image_path):
        # Load the saved face image
        saved_face_image = cv2.imread(face_image_path)
        print(f"Loaded existing registered face from {face_image_path}")
        # Initialize face parameters (these will be overwritten later)
        h, w, _ = saved_face_image.shape
        center_x, center_y = w // 2, h // 2
        registered_face_center = (center_x, center_y)
        registered_face_area = (w * h) / (640 * 480)  # Normalized area
        registered_user = True
    else:
        print("No registered face found. Please register using the peace sign and fist gestures.")
    
    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    
    # Capture video from webcam
    print("Attempting to open webcam...")
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    print("Webcam opened successfully!")
    
    # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create a window to display the camera feed
    cv2.namedWindow("Fruit Ninja Controller", cv2.WINDOW_NORMAL)
    
    # Function to calculate distance between two points
    def calculate_distance(p1, p2):
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    # Function to detect if index finger is extended while others are curled
    def is_index_finger_only(landmarks, image_width, image_height):
        # Check if index finger is extended (tip is above pip)
        index_tip_y = landmarks.landmark[8].y * image_height
        index_pip_y = landmarks.landmark[6].y * image_height
        
        # Check if other fingers are curled (tips are below pips)
        middle_tip_y = landmarks.landmark[12].y * image_height
        middle_pip_y = landmarks.landmark[10].y * image_height
        
        ring_tip_y = landmarks.landmark[16].y * image_height
        ring_pip_y = landmarks.landmark[14].y * image_height
        
        pinky_tip_y = landmarks.landmark[20].y * image_height
        pinky_pip_y = landmarks.landmark[18].y * image_height
        
        # Index finger should be extended (tip above pip)
        index_extended = index_tip_y < index_pip_y
        
        # Other fingers should be curled (tips below pips)
        other_fingers_curled = (
            middle_tip_y > middle_pip_y and
            ring_tip_y > ring_pip_y and
            pinky_tip_y > pinky_pip_y
        )
        
        # Optionally check thumb as well
        thumb_tip_y = landmarks.landmark[4].y * image_height
        thumb_ip_y = landmarks.landmark[3].y * image_height
        thumb_curled = thumb_tip_y > thumb_ip_y
        
        # Return true only if index is extended and others are curled
        return index_extended and other_fingers_curled
    
    # Function to detect a peace sign (index and middle fingers extended)
    def is_peace_sign(landmarks, image_width, image_height):
        # Check if index finger is extended
        index_tip_y = landmarks.landmark[8].y * image_height
        index_pip_y = landmarks.landmark[6].y * image_height
        index_extended = index_tip_y < index_pip_y
        
        # Check if middle finger is extended
        middle_tip_y = landmarks.landmark[12].y * image_height
        middle_pip_y = landmarks.landmark[10].y * image_height
        middle_extended = middle_tip_y < middle_pip_y
        
        # Check if ring finger is curled
        ring_tip_y = landmarks.landmark[16].y * image_height
        ring_pip_y = landmarks.landmark[14].y * image_height
        ring_curled = ring_tip_y > ring_pip_y
        
        # Check if pinky is curled
        pinky_tip_y = landmarks.landmark[20].y * image_height
        pinky_pip_y = landmarks.landmark[18].y * image_height
        pinky_curled = pinky_tip_y > pinky_pip_y
        
        # Return true if index and middle are extended, others are curled
        return index_extended and middle_extended and ring_curled and pinky_curled
    
    # Function to detect if a fist is made (all fingers curled)
    def is_fist(landmarks, image_width, image_height):
        # Check if all fingers are curled
        index_tip_y = landmarks.landmark[8].y * image_height
        index_pip_y = landmarks.landmark[6].y * image_height
        index_curled = index_tip_y > index_pip_y
        
        middle_tip_y = landmarks.landmark[12].y * image_height
        middle_pip_y = landmarks.landmark[10].y * image_height
        middle_curled = middle_tip_y > middle_pip_y
        
        ring_tip_y = landmarks.landmark[16].y * image_height
        ring_pip_y = landmarks.landmark[14].y * image_height
        ring_curled = ring_tip_y > ring_pip_y
        
        pinky_tip_y = landmarks.landmark[20].y * image_height
        pinky_pip_y = landmarks.landmark[18].y * image_height
        pinky_curled = pinky_tip_y > pinky_pip_y
        
        # Check thumb
        thumb_tip_x = landmarks.landmark[4].x * image_width
        thumb_ip_x = landmarks.landmark[3].x * image_width
        thumb_curled = thumb_tip_x < thumb_ip_x  # For fist, thumb is usually tucked in
        
        # All fingers should be curled for a fist
        return index_curled and middle_curled and ring_curled and pinky_curled
    
    print("Starting hand tracking. Press 'q' to quit...")
    print("For face registration: First make a peace sign ✌️, then make a fist ✊")
    
    with mp_hands.Hands(
        model_complexity=1,  # Use medium model for better distance detection
        min_detection_confidence=0.4,  # Lower threshold (was 0.5)
        min_tracking_confidence=0.4,  # Lower threshold (was 0.5)
        max_num_hands=1) as hands, \
        mp_face_detection.FaceDetection(
            model_selection=1,  # Use the LONG-RANGE model (0 was short-range)
            min_detection_confidence=0.4) as face_detection:  # Lower threshold (was 0.5)
        
        frame_count = 0
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Process every other frame to reduce CPU load
            frame_count += 1
            
            # Flip the image horizontally for a more intuitive mirror view - ALWAYS do this
            image = cv2.flip(image, 1)
            
            if frame_count % 2 != 0:
                # Skip detailed processing but still show the flipped frame
                cv2.imshow("Fruit Ninja Controller", image)
                key = cv2.waitKey(5) & 0xFF
                if key == ord('r'):
                    print("Registration reset requested by user")
                    registered_user = False
                    if os.path.exists(face_image_path):
                        try:
                            os.remove(face_image_path)
                            print(f"Deleted saved face: {face_image_path}")
                        except Exception as e:
                            print(f"Error deleting file: {e}")
                    auth_step = 0
                    face_detected = False
                    saved_face_image = None
                elif key == ord('q'):
                    break
                continue
            
            # Create a canvas for drawing the swipe effect
            swipe_canvas = np.zeros_like(image)
            
            # Convert the image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image for face detection (less frequently for performance)
            face_detection_cooldown -= 1
            if face_detection_cooldown <= 0 or not face_detected:
                face_results = face_detection.process(image_rgb)
                face_detection_cooldown = 5  # Check face every 5 processed frames
                
                # Reset face detection status
                face_detected = False
                
                if face_results.detections:
                    for detection in face_results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = image.shape
                        x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), \
                                             int(bbox.width * w), int(bbox.height * h)
                        
                        # Draw face rectangle
                        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
                        
                        # If we have a registered face, check if this is that face
                        if registered_user:
                            # Calculate current face center and area
                            current_face_center_x = int(x + width/2)
                            current_face_center_y = int(y + height/2)
                            current_face_area = bbox.width * bbox.height
                            
                            # Calculate similarity metrics
                            center_distance = calculate_distance(
                                (current_face_center_x, current_face_center_y), 
                                registered_face_center
                            )
                            
                            area_ratio = current_face_area / registered_face_area if registered_face_area > 0 else 0
                            
                            # If the face is similar enough to the registered face
                            # (close position and similar size) - more lenient parameters
                            if center_distance < face_distance_threshold and face_size_ratio_min < area_ratio < face_size_ratio_max:
                                face_detected = True
                                # Mark this as the registered face
                                cv2.putText(image, "Registered User", (x, y - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            # If no user is registered, treat any face as valid
                            face_detected = True
            
            # Process the image and detect hands - only if face is detected or no user registered
            if face_detected or not registered_user:
                results = hands.process(image_rgb)
                
                # Draw hand annotations on the image
                image_height, image_width, _ = image.shape
                
                # Check if hands are detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw the hand landmarks
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))
                        
                        # Check for peace sign for first step of face registration
                        if auth_step == 0 and is_peace_sign(hand_landmarks, image_width, image_height):
                            peace_sign_frames += 1
                            
                            # Display peace sign detection progress - MOVED TO BOTTOM OF SCREEN
                            progress = min(peace_sign_frames / required_peace_frames, 1.0)
                            bar_width = int(200 * progress)
                            
                            # Position at bottom of screen instead of middle
                            bar_y = image_height - 60  # Bottom position
                            cv2.rectangle(image, (220, bar_y), (220 + bar_width, bar_y + 20), (0, 255, 0), -1)
                            cv2.rectangle(image, (220, bar_y), (420, bar_y + 20), (255, 255, 255), 2)
                            cv2.putText(image, "Step 1: Hold Peace Sign ✌️", (160, bar_y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            # If peace sign held long enough, move to step 2
                            if peace_sign_frames >= required_peace_frames:
                                auth_step = 1
                                peace_sign_frames = 0
                                auth_cooldown = 30  # Give some frames to transition to fist
                                print("Step 1 complete: Peace sign recognized. Now make a fist.")
                        
                        # Check for fist for second step of face registration
                        elif auth_step == 1 and is_fist(hand_landmarks, image_width, image_height):
                            if auth_cooldown > 0:
                                auth_cooldown -= 1
                                # Show transition message at bottom of screen
                                bar_y = image_height - 60
                                cv2.putText(image, "Now make a fist ✊", (160, bar_y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            else:
                                fist_sign_frames += 1
                                
                                # Display fist detection progress at bottom of screen
                                progress = min(fist_sign_frames / required_fist_frames, 1.0)
                                bar_width = int(200 * progress)
                                
                                # Position at bottom of screen
                                bar_y = image_height - 60
                                cv2.rectangle(image, (220, bar_y), (220 + bar_width, bar_y + 20), (0, 255, 0), -1)
                                cv2.rectangle(image, (220, bar_y), (420, bar_y + 20), (255, 255, 255), 2)
                                cv2.putText(image, "Step 2: Hold Fist ✊", (160, bar_y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                
                                # If fist held long enough, register face
                                if fist_sign_frames >= required_fist_frames:
                                    # Register the user's face using MediaPipe
                                    print("Step 2 complete: Fist recognized. Registering face...")
                                    # Get face detection results
                                    face_results = face_detection.process(image_rgb)
                                    
                                    if face_results.detections:
                                        # Get the first face detected
                                        detection = face_results.detections[0]
                                        bbox = detection.location_data.relative_bounding_box
                                        h, w, _ = image.shape
                                        
                                        # Calculate face center and area
                                        face_center_x = int((bbox.xmin + bbox.width/2) * w)
                                        face_center_y = int((bbox.ymin + bbox.height/2) * h)
                                        face_area = bbox.width * bbox.height
                                        
                                        # Store the registered face information
                                        registered_face_center = (face_center_x, face_center_y)
                                        registered_face_area = face_area
                                        registered_user = True
                                        
                                        # Save the face image to a file
                                        # Calculate the face bounding box with margin
                                        x = max(0, int(bbox.xmin * w) - face_save_margin)
                                        y = max(0, int(bbox.ymin * h) - face_save_margin)
                                        width = min(int(bbox.width * w) + face_save_margin * 2, w - x)
                                        height = min(int(bbox.height * h) + face_save_margin * 2, h - y)
                                        
                                        # Extract and save the face region
                                        face_img = image[y:y+height, x:x+width]
                                        cv2.imwrite(face_image_path, face_img)
                                        saved_face_image = face_img.copy()
                                        
                                        print(f"Face registered and saved to {face_image_path}!")
                                        print(f"Face center: {registered_face_center}, Area: {registered_face_area:.4f}")
                                        
                                        # Draw a green rectangle around the registered face
                                        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 3)
                                        cv2.putText(image, "Face Registered & Saved!", (x, y - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    else:
                                        print("No face detected for registration.")
                                        cv2.putText(image, "No face detected for registration!", (160, 280),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    
                                    # Reset authentication state
                                    auth_step = 0
                                    fist_sign_frames = 0
                        
                        # If neither the correct peace sign nor fist is detected, reset progress
                        elif auth_step == 0:
                            peace_sign_frames = 0
                        elif auth_step == 1 and auth_cooldown <= 0 and not is_fist(hand_landmarks, image_width, image_height):
                            fist_sign_frames = 0
                            # If too much time passes without completing step 2, reset to step 0
                            if fist_sign_frames == 0 and not is_fist(hand_landmarks, image_width, image_height):
                                auth_step = 0
                                cv2.putText(image, "Authentication reset. Start over.", (160, 280),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Check if only index finger is extended
                        using_index_finger = is_index_finger_only(hand_landmarks, image_width, image_height)
                        
                        # Only allow finger tracking for registered face or if no face registered yet
                        if using_index_finger and (face_detected or not registered_user):
                            # Get index finger tip position
                            index_finger_tip = hand_landmarks.landmark[8]
                            finger_x = int(index_finger_tip.x * image_width)
                            finger_y = int(index_finger_tip.y * image_height)
                            
                            # Draw a circle at the finger tip
                            cv2.circle(image, (finger_x, finger_y), 8, (0, 255, 0), -1)
                            
                            # Store the current position for trail effect
                            finger_trail.append((finger_x, finger_y))
                            
                            # Scale the finger position to screen coordinates
                            screen_x = min(max(0, int(finger_x * screen_width / image_width)), screen_width-1)
                            screen_y = min(max(0, int(finger_y * screen_height / image_height)), screen_height-1)
                            
                            # Apply smoothing to reduce jitter
                            if prev_cursor_x == 0 and prev_cursor_y == 0:
                                # First frame, no smoothing
                                prev_cursor_x, prev_cursor_y = screen_x, screen_y
                            else:
                                # Apply smoothing formula: new_pos = prev_pos * smoothing + current_pos * (1-smoothing)
                                smoothed_x = int(prev_cursor_x * cursor_smoothing + screen_x * (1 - cursor_smoothing))
                                smoothed_y = int(prev_cursor_y * cursor_smoothing + screen_y * (1 - cursor_smoothing))
                                screen_x, screen_y = smoothed_x, smoothed_y
                                prev_cursor_x, prev_cursor_y = screen_x, screen_y
                            
                            # Move the cursor to the scaled and smoothed position
                            try:
                                pyautogui.moveTo(screen_x, screen_y, _pause=False)
                                
                                # SIMPLIFIED APPROACH: Always keep mouse button down while finger is tracking
                                # This is ideal for Fruit Ninja where we want continuous slicing
                                if not is_dragging:
                                    # Press mouse button down when finger is first detected
                                    pyautogui.mouseDown(button='left', _pause=False)
                                    is_dragging = True
                                    is_swiping = True
                                    print("Mouse DOWN - starting slice")
                                
                                # Update previous finger position
                                prev_finger_pos = (finger_x, finger_y)
                                
                            except Exception as e:
                                print(f"Error moving cursor: {e}")
                        else:
                            # If no longer using index finger or face is not recognized, release mouse button
                            if is_dragging:
                                try:
                                    pyautogui.mouseUp(button='left', _pause=False)
                                    is_dragging = False
                                    is_swiping = False
                                    print("Mouse UP - ending slice")
                                except Exception as e:
                                    print(f"Error releasing mouse: {e}")
                            
                            # If finger is detected but face is not the registered one, show a warning
                            if using_index_finger and registered_user and not face_detected:
                                # Get index finger tip position to show warning
                                index_finger_tip = hand_landmarks.landmark[8]
                                finger_x = int(index_finger_tip.x * image_width)
                                finger_y = int(index_finger_tip.y * image_height)
                                
                                # Draw a red X at the finger tip to indicate tracking is disabled
                                cv2.line(image, (finger_x-10, finger_y-10), (finger_x+10, finger_y+10), (0, 0, 255), 3)
                                cv2.line(image, (finger_x+10, finger_y-10), (finger_x-10, finger_y+10), (0, 0, 255), 3)
                                
                                # Add warning text and registration instructions
                                cv2.putText(image, "FACE NOT RECOGNIZED - TRACKING LOCKED", (60, 400),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                cv2.putText(image, "Make peace sign ✌️ then fist ✊ to register", (60, 430),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Process fewer effects when drawing to reduce lag
            # Simplify the swipe effect rendering when performance is an issue
            if is_swiping and len(finger_trail) > 1:
                # Draw a simpler trail for performance
                for i in range(1, len(finger_trail), 2):  # Skip every other point
                    if finger_trail[i-1] is None or finger_trail[i] is None:
                        continue
                    cv2.line(swipe_canvas, finger_trail[i-1], finger_trail[i], 
                             swipe_color, swipe_thickness)
                             
                # Only apply blur if performance allows
                if len(finger_trail) < 10:  # Apply blur for shorter trails only
                    swipe_canvas = cv2.GaussianBlur(swipe_canvas, (15, 15), 0)
            
            # Blend swipe canvas with the main image
            image = cv2.addWeighted(image, 1.0, swipe_canvas, 0.7, 0)
            
            # If a face is saved, display it in a small window in corner
            if saved_face_image is not None and registered_user:
                # Resize the saved face to a smaller size for display
                display_height = 100
                display_width = int(saved_face_image.shape[1] * (display_height / saved_face_image.shape[0]))
                display_face = cv2.resize(saved_face_image, (display_width, display_height))
                
                # Determine the position (top-right corner)
                y_offset = 10
                x_offset = image.shape[1] - display_width - 10
                
                # Create a region of interest in the main image
                roi = image[y_offset:y_offset+display_height, x_offset:x_offset+display_width]
                
                # Create a semi-transparent overlay
                overlay = display_face.copy()
                cv2.rectangle(image, (x_offset-5, y_offset-5), 
                             (x_offset+display_width+5, y_offset+display_height+5), 
                             (255, 255, 255), 2)
                cv2.putText(image, "Registered Face", (x_offset, y_offset-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Only overlay the face where there's room
                try:
                    # Create a mask for blending
                    alpha = 0.7
                    cv2.addWeighted(display_face, alpha, roi, 1-alpha, 0, roi)
                except:
                    # If there's a size mismatch, just skip this frame's overlay
                    pass
            
            # Display status and instructions
            cv2.putText(image, "Extend ONLY your index finger to control", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, "Auth: Peace sign ✌️ then Fist ✊", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Smoothing: {cursor_smoothing:.1f} (Press 'r' to reset)", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add distance indicator
            if not face_detected and registered_user:
                cv2.putText(image, "MOVE CLOSER TO CAMERA!", (170, 350), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                # Add registration instruction for new users
                cv2.putText(image, "New user? Make peace sign ✌️ then fist ✊", (120, 380),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Show face recognition status
            face_status = "REGISTERED" if registered_user else "NOT REGISTERED"
            face_lock = "UNLOCKED" if face_detected or not registered_user else "LOCKED"
            cv2.putText(image, f"Face: {face_status} - Control: {face_lock}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, "Press 'q' to quit", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show the image with hand landmarks and swipe effect
            cv2.imshow("Fruit Ninja Controller", image)
            
            # Exit the program when 'q' is pressed
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
    
    print("Cleanup: releasing camera and closing windows")
    cap.release()
    cv2.destroyAllWindows()
    print("Program completed successfully")

    # Just before exiting, ensure mouse button is released to prevent stuck buttons
    if is_dragging:
        try:
            pyautogui.mouseUp(button='left', _pause=False)
            print("Mouse UP - releasing before exit")
        except Exception as e:
            print(f"Error releasing mouse button: {e}")

except Exception as e:
    print("ERROR: An exception occurred!")
    print(f"Exception: {e}")
    print("Traceback:")
    traceback.print_exc()
    input("Press Enter to exit...")
