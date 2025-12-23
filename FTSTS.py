import cv2
import numpy as np
import time
import winsound  # For audio cue (Windows specific)
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, BodyTrackingMode

# --- Configuration & Constants ---
class AppConfig:
    # Thresholds (Vertical distance in millimeters to consider "Standing")
    # This is a heuristic: if head moves up by ~30cm from sitting position
    STAND_THRESHOLD_MM = 300 
    
    # Risk Criteria
    AGE_THRESHOLD = 75
    TIME_LIMIT_YOUNG = 11.5  # Age <= 74
    TIME_LIMIT_OLD = 12.1    # Age >= 75

class TestState:
    IDLE = 0
    CALIBRATING = 1  # 3 seconds sit still
    TESTING = 2      # Counting reps
    FINISHED = 3

class FiveTimesSitToStandApp:
    def __init__(self):
        # Initialize Kinect
        self.k4a = PyK4A(
            Config(
                color_resolution=ColorResolution.RES_720P,
                depth_mode=DepthMode.NFOV_UNBINNED,
                camera_fps=30,
            ),
            device_id=0,
        )
        self.k4a.start()
        
        # Initialize Body Tracker
        # Ensure 'pyk4a.body_tracker' is accessible or configured correctly
        from pyk4a import PyK4ABodyTracker
        self.body_tracker = PyK4ABodyTracker()
        self.body_tracker.start()

        # Test Variables
        self.age = 0
        self.time_limit = 0.0
        self.state = TestState.IDLE
        self.start_time = 0
        self.end_time = 0
        self.reps_count = 0
        self.is_standing = False
        self.sitting_head_height = 0.0 # Baseline height when sitting
        
        # UI Text
        self.message = "Press 'SPACE' to start input."

    def get_user_age(self):
        """Console input for Age to set criteria"""
        try:
            print("\n--- NEW TEST SESSION ---")
            val = input("Please enter the patient's Age: ")
            self.age = int(val)
            
            # Set Criteria based on requirements
            if self.age <= 74:
                self.time_limit = AppConfig.TIME_LIMIT_YOUNG
                print(f"Age {self.age}: Cut-off time is {self.time_limit} sec.")
            else:
                self.time_limit = AppConfig.TIME_LIMIT_OLD
                print(f"Age {self.age}: Cut-off time is {self.time_limit} sec.")
                
            self.state = TestState.CALIBRATING
            self.start_time = time.time() # Start 3s countdown
            self.message = "Sit Still... Calibrating"
            
        except ValueError:
            print("Invalid input. Please enter a number.")
            self.message = "Invalid Age. Press SPACE to try again."

    def play_sound(self):
        """Simple beep for start cue"""
        # Frequency 1000Hz, Duration 500ms
        winsound.Beep(1000, 500) 

    def process_logic(self, body):
        """Core logic for counting Sit-to-Stand reps"""
        
        # Get Head Joint Position (Joint ID 26 in K4A usually, or use index)
        # Using PELVIS (0) or HEAD (26) usually works. Let's use HEAD for height diff.
        # body.joints is a structure. Assuming we access position via .position (x, y, z)
        # Note: In K4A, Y-axis is vertical (usually positive down, but depends on calibration).
        # We will use relative change, so absolute direction matters less if we take diff.
        
        # Using Pelvis is often more stable than Head for sit/stand
        # Joint 0 = PELVIS, Joint 3 = HEAD (check specific SDK mapping)
        # Here assuming body.joints[0] is PELVIS. 
        # K4A coordinate: Y is DOWN. So smaller Y = Higher physical position.
        
        current_height = body.joints[0, 1] # Pelvis Y (mm)
        
        # 1. Calibration Phase (3 Seconds)
        if self.state == TestState.CALIBRATING:
            elapsed = time.time() - self.start_time
            remaining = 3 - elapsed
            self.message = f"Sit Still... {remaining:.1f}s"
            
            # Continuously update baseline "sitting height"
            self.sitting_head_height = current_height
            
            if elapsed >= 3.0:
                self.play_sound()
                self.state = TestState.TESTING
                self.start_time = time.time() # Reset timer for actual test
                self.reps_count = 0
                self.is_standing = False
                self.message = "GO! Stand up and Sit down."

        # 2. Testing Phase
        elif self.state == TestState.TESTING:
            test_duration = time.time() - self.start_time
            
            # Logic: Determine Stand vs Sit based on displacement from baseline
            # If Pelvis moves UP (Y decreases in K4A) by threshold -> Standing
            # Note: In K4A, Y is down, so Standing Y < Sitting Y
            displacement = self.sitting_head_height - current_height 
            
            # Check Stand Condition
            if not self.is_standing and displacement > AppConfig.STAND_THRESHOLD_MM:
                self.is_standing = True
                # User has stood up
                
            # Check Sit Condition (Cycle Complete)
            elif self.is_standing and displacement < (AppConfig.STAND_THRESHOLD_MM / 2):
                # Using a slightly lower threshold for hysteresis/debounce
                self.is_standing = False
                self.reps_count += 1
                winsound.Beep(600, 100) # Short beep for rep count
            
            self.message = f"Reps: {self.reps_count}/5 | Time: {test_duration:.2f}s"

            # Check Completion
            if self.reps_count >= 5:
                self.end_time = test_duration
                self.state = TestState.FINISHED
                self.evaluate_risk()

    def evaluate_risk(self):
        """Determine Low vs High Risk"""
        status = "LOW RISK"
        color = (0, 255, 0) # Green
        
        if self.end_time > self.time_limit:
            status = "HIGH FALL RISK"
            color = (0, 0, 255) # Red
            
        self.message = f"Result: {status} ({self.end_time:.2f}s)"
        self.result_color = color

    def run(self):
        while True:
            # 1. Capture
            capture = self.k4a.get_capture()
            if capture is not None:
                # Get Color Image for display
                img = capture.color_image_as_numpy_array()
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Get Body Tracking
                body_frame = self.body_tracker.update()
                
                # 2. Process Logic if body detected
                if body_frame.get_num_bodies() > 0:
                    # Get the first person detected (closest usually index 0)
                    body = body_frame.get_body(0)
                    
                    if self.state != TestState.IDLE and self.state != TestState.FINISHED:
                        self.process_logic(body)
                    
                    # Visualization (Draw skeleton - optional simple visualization)
                    # Simple circle on Pelvis
                    pelvis_x = int(body.joints[0, 0]) # 2D projection needed for perfect mapping
                    # For simplicity in this snippet, we just draw text mostly.
                    # Getting 2D coordinates requires k4a calibration map, usually handled by SDK helper.
                    
                else:
                    if self.state == TestState.TESTING:
                        self.message = "No body detected! Please stay in frame."

                # 3. Draw UI
                # Overlay Text
                cv2.rectangle(img, (0, 0), (1280, 100), (0, 0, 0), -1)
                
                if self.state == TestState.FINISHED:
                    cv2.putText(img, self.message, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.result_color, 3)
                    cv2.putText(img, "Press SPACE to restart", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                else:
                    cv2.putText(img, self.message, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                cv2.imshow("Azure Kinect - 5STS Fall Risk", img)

            # 4. Input Handling
            key = cv2.waitKey(10) & 0xFF
            if key == 27: # ESC to quit
                break
            elif key == 32: # SPACE
                if self.state == TestState.IDLE or self.state == TestState.FINISHED:
                    self.get_user_age()

        # Cleanup
        self.body_tracker.stop()
        self.k4a.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FiveTimesSitToStandApp()
    app.run()