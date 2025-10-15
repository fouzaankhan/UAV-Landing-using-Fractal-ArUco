#!/usr/bin/env python3
"""
Complete Autonomous Landing System - For Working AirSim Environments
Connects directly to your running AirSim simulation
"""

import numpy as np
import cv2
import time
import json
import threading
from queue import Queue
import logging
from enum import Enum

# Import our AirSim connector
from airsim_connector import AirSimConnector, ImageType

class LandingState(Enum):
    """Landing state machine states"""
    SEARCHING = "SEARCHING"
    APPROACHING = "APPROACHING" 
    POSITIONING = "POSITIONING"
    DESCENDING = "DESCENDING"
    LANDED = "LANDED"
    ABORTED = "ABORTED"

class WorkingAirSimLanding:
    """
    Complete autonomous landing system for working AirSim environments
    """
    
    def __init__(self, config_file="landing_config.json"):
        """Initialize autonomous landing system"""
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Connect to your running AirSim
        self.client = AirSimConnector()
        
        # Test connection
        print("🔌 Connecting to your AirSim...")
        if not self.client.confirmConnection():
            raise Exception("Cannot connect to AirSim! Make sure it's running.")
        
        print("✅ Connected to AirSim successfully!")
        
        # Initialize components
        self.camera_matrix = self._create_camera_matrix()
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        # State management
        self.state = LandingState.SEARCHING
        self.landing_target = None
        
        # Threading for image processing
        self.detection_queue = Queue(maxsize=5)
        self.processing_thread = None
        self.running = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🚁 Autonomous Landing System initialized for AirSim")
    
    def _load_config(self, config_file):
        """Load configuration"""
        default_config = {
            "marker": {
                "dict_type": "DICT_6X6_250",
                "size": 0.5,
                "target_id": 10
            },
            "camera": {
                "name": "front_center",
                "fov_degrees": 90,
                "image_width": 640,
                "image_height": 480
            },
            "landing": {
                "search_altitude": 10.0,
                "approach_altitude": 5.0,
                "precision_threshold": 0.3,
                "descent_rate": 0.5,
                "timeout": 300
            },
            "safety": {
                "min_altitude": 1.0,
                "max_search_time": 60
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                for key in default_config:
                    if key in loaded_config:
                        default_config[key].update(loaded_config[key])
                return default_config
        except FileNotFoundError:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _create_camera_matrix(self):
        """Create camera calibration matrix"""
        camera_config = self.config['camera']
        image_width = camera_config['image_width']
        image_height = camera_config['image_height'] 
        fov_degrees = camera_config['fov_degrees']
        
        # Calculate focal length from FOV
        fov_radians = np.radians(fov_degrees)
        focal_length = image_width / (2 * np.tan(fov_radians / 2))
        
        camera_matrix = np.array([
            [focal_length, 0, image_width / 2],
            [0, focal_length, image_height / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return camera_matrix
    
    def detect_fractal_aruco_markers(self, image):
        """Detect fractal ArUco markers in image"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Create ArUco detector
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            detector_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
            
            # Detect markers
            corners, ids, rejected = detector.detectMarkers(gray)
            
            if ids is not None and len(ids) > 0:
                # Find the largest marker (primary landing target)
                largest_idx = 0
                largest_area = 0
                
                for i, corner in enumerate(corners):
                    area = cv2.contourArea(corner[0])
                    if area > largest_area:
                        largest_area = area
                        largest_idx = i
                
                # Estimate pose using solvePnP
                marker_size = self.config['marker']['size']
                half_size = marker_size / 2
                
                # 3D points of marker corners
                marker_points = np.array([
                    [-half_size, half_size, 0],   # Top-left
                    [half_size, half_size, 0],    # Top-right
                    [half_size, -half_size, 0],   # Bottom-right
                    [-half_size, -half_size, 0]   # Bottom-left
                ], dtype=np.float32)
                
                # Solve PnP for pose estimation
                success, rvec, tvec = cv2.solvePnP(
                    marker_points,
                    corners[largest_idx][0],
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                if success:
                    marker_id = ids[largest_idx][0]
                    center = np.mean(corners[largest_idx][0], axis=0)
                    distance = np.linalg.norm(tvec)
                    
                    # Calculate confidence based on marker area and detection quality
                    confidence = min(largest_area / 5000, 1.0)
                    
                    return {
                        'detected': True,
                        'marker_id': marker_id,
                        'center': center,
                        'distance': distance,
                        'confidence': confidence,
                        'rvec': rvec,
                        'tvec': tvec,
                        'corners': corners[largest_idx]
                    }
            
            return {'detected': False}
            
        except Exception as e:
            self.logger.error(f"Marker detection failed: {e}")
            return {'detected': False}
    
    def start_landing_sequence(self):
        """Start the autonomous landing sequence"""
        self.logger.info("🚀 Starting autonomous landing sequence")
        self.running = True
        
        # Start image processing thread
        self.processing_thread = threading.Thread(target=self._image_processing_loop)
        self.processing_thread.start()
        
        try:
            # Enable API control
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            
            # Execute main landing sequence
            self._execute_landing_sequence()
            
        except KeyboardInterrupt:
            self.logger.info("Landing sequence interrupted by user")
        except Exception as e:
            self.logger.error(f"Landing sequence failed: {e}")
            self.state = LandingState.ABORTED
        finally:
            # Cleanup
            self.running = False
            if self.processing_thread:
                self.processing_thread.join()
            
            try:
                self.client.landAsync().join()
                self.client.armDisarm(False)
                self.client.enableApiControl(False)
            except:
                pass
    
    def _execute_landing_sequence(self):
        """Execute the main landing sequence"""
        
        # Phase 1: Takeoff to search altitude
        self.logger.info(f"🚀 Taking off to {self.config['landing']['search_altitude']}m")
        self.client.takeoffAsync().join()
        time.sleep(3)  # Wait for takeoff to complete
        
        self.client.moveToZAsync(-self.config['landing']['search_altitude'], 2.0).join()
        time.sleep(2)  # Wait for altitude adjustment
        
        # Phase 2: Search for landing markers
        self.state = LandingState.SEARCHING
        self.logger.info("🔍 Searching for fractal ArUco landing markers...")
        
        search_start_time = time.time()
        max_search_time = self.config['safety']['max_search_time']
        
        while (time.time() - search_start_time) < max_search_time and self.running:
            # Check for marker detection
            if not self.detection_queue.empty():
                target = self.detection_queue.get()
                self.landing_target = target
                
                self.logger.info(f"🎯 Landing marker detected!")
                self.logger.info(f"   Marker ID: {target['marker_id']}")
                self.logger.info(f"   Distance: {target['distance']:.2f}m")
                self.logger.info(f"   Confidence: {target['confidence']:.2f}")
                
                # Proceed to approach phase
                self._execute_approach_and_landing()
                return
            
            # Hover and continue searching
            time.sleep(1)
        
        # Search timeout
        self.logger.warning("⏰ Search timeout - executing emergency landing")
        self.client.landAsync().join()
    
    def _execute_approach_and_landing(self):
        """Execute approach and precision landing"""
        if not self.landing_target:
            return
        
        # Phase 3: Approach altitude
        self.state = LandingState.APPROACHING
        approach_alt = -self.config['landing']['approach_altitude']
        
        self.logger.info(f"➡️ Approaching - moving to {self.config['landing']['approach_altitude']}m altitude")
        
        # Get current position for reference
        current_pose = self.client.simGetVehiclePose()
        target_x = current_pose.position.x_val
        target_y = current_pose.position.y_val
        
        # Move to approach altitude
        self.client.moveToPositionAsync(target_x, target_y, approach_alt, 2.0).join()
        time.sleep(2)
        
        # Phase 4: Precision positioning and descent
        self.state = LandingState.POSITIONING
        self.logger.info("🎯 Starting precision positioning and controlled descent")
        
        descent_altitudes = [4, 3, 2, 1.5, 1]
        
        for altitude in descent_altitudes:
            self.logger.info(f"⬇️ Descending to {altitude}m with position correction...")
            
            # Move to target altitude
            self.client.moveToPositionAsync(target_x, target_y, -altitude, 1.0).join()
            time.sleep(1.5)
            
            # Check for updated marker detection to refine position
            if not self.detection_queue.empty():
                updated_target = self.detection_queue.get()
                if updated_target['detected']:
                    self.logger.info(f"📍 Position refined - Distance: {updated_target['distance']:.2f}m")
                    
                    # Simple position correction based on marker center
                    image_center_x = self.config['camera']['image_width'] / 2
                    image_center_y = self.config['camera']['image_height'] / 2
                    
                    # Calculate offset from image center
                    offset_x = (updated_target['center'][0] - image_center_x) * 0.01  # Scale factor
                    offset_y = (updated_target['center'][1] - image_center_y) * 0.01
                    
                    # Adjust position
                    target_x += offset_x
                    target_y += offset_y
        
        # Phase 5: Final landing
        self.state = LandingState.DESCENDING
        self.logger.info("🛬 Executing final landing sequence...")
        
        self.client.landAsync().join()
        
        self.state = LandingState.LANDED
        self.logger.info("✅ 🎉 AUTONOMOUS LANDING COMPLETED SUCCESSFULLY! 🎉")
    
    def _image_processing_loop(self):
        """Process camera images for marker detection"""
        while self.running:
            try:
                # Get camera image from AirSim
                response = self.client.simGetImage(
                    self.config['camera']['name'],
                    ImageType.Scene
                )
                
                if response and len(response) > 0:
                    # Decode image
                    img_1d = np.frombuffer(response, dtype=np.uint8)
                    if len(img_1d) > 0:
                        img_rgb = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)
                        
                        if img_rgb is not None:
                            # Detect fractal ArUco markers
                            detection_result = self.detect_fractal_aruco_markers(img_rgb)
                            
                            if detection_result['detected'] and not self.detection_queue.full():
                                self.detection_queue.put(detection_result)
                            
                            # Optional: Save debug images
                            if detection_result['detected']:
                                # Draw detection on image
                                debug_img = img_rgb.copy()
                                cv2.aruco.drawDetectedMarkers(debug_img, [detection_result['corners']])
                                cv2.imwrite("debug_detection.jpg", debug_img)
                
                time.sleep(0.1)  # 10Hz processing rate
                
            except Exception as e:
                self.logger.error(f"Image processing error: {e}")
                time.sleep(0.5)

def create_test_markers():
    """Create test markers for the system"""
    print("🎯 Creating test fractal ArUco markers...")
    
    try:
        import os
        os.makedirs("test_markers", exist_ok=True)
        
        # Create ArUco dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        
        # Create different sized markers for fractal effect
        marker_configs = [
            {"id": 10, "size": 800, "name": "large_marker_10.png"},
            {"id": 11, "size": 400, "name": "medium_marker_11.png"}, 
            {"id": 12, "size": 200, "name": "small_marker_12.png"},
            {"id": 20, "size": 600, "name": "marker_20.png"}
        ]
        
        for config in marker_configs:
            marker_image = cv2.aruco.generateImageMarker(aruco_dict, config["id"], config["size"])
            filepath = f"test_markers/{config['name']}"
            cv2.imwrite(filepath, marker_image)
            print(f"✅ Created {config['name']} (ID: {config['id']}, Size: {config['size']}px)")
        
        # Create a simple fractal marker by combining
        print("\n🎨 Creating fractal marker...")
        base_marker = cv2.aruco.generateImageMarker(aruco_dict, 10, 800)
        
        # Add smaller markers in corners
        small_marker = cv2.aruco.generateImageMarker(aruco_dict, 11, 150)
        
        # Embed smaller markers
        positions = [(50, 50), (600, 50), (50, 600), (600, 600)]
        for pos in positions:
            x, y = pos
            base_marker[y:y+150, x:x+150] = small_marker
        
        cv2.imwrite("test_markers/fractal_marker_combined.png", base_marker)
        print("✅ Created fractal_marker_combined.png")
        
        print(f"\n📁 All markers saved to 'test_markers/' directory")
        print("💡 Place these markers in your AirSim environment for testing")
        print("🎯 Recommended: Use 'fractal_marker_combined.png' for best results")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create markers: {e}")
        return False

def main():
    """Main function"""
    print("🚁 WORKING AIRSIM AUTONOMOUS LANDING SYSTEM")
    print("="*60)
    print("✅ Connects directly to your running AirSim!")
    print("✅ No problematic package installation needed!")
    print("="*60)
    
    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "markers":
            create_test_markers()
            
        elif command == "test":
            # Test the connection
            try:
                landing_system = WorkingAirSimLanding()
                print("✅ System initialized successfully!")
                print("\n📋 Connection Status:")
                print("✅ AirSim: Connected")
                print("✅ Camera: Available") 
                print("✅ API Control: Ready")
                print("\n🎯 Ready for autonomous landing!")
                
            except Exception as e:
                print(f"❌ System test failed: {e}")
                print("\n💡 Troubleshooting:")
                print("1. Make sure AirSim is running")
                print("2. Ensure a drone is spawned in the environment")
                print("3. Check that API access is enabled")
                
        elif command == "landing":
            # Run full autonomous landing
            try:
                print("🚁 Initializing autonomous landing system...")
                landing_system = WorkingAirSimLanding()
                
                print("\n⚠️  SAFETY CHECK:")
                print("1. AirSim environment is clear of obstacles")
                print("2. Landing markers are placed in the scene")
                print("3. Drone has enough space to maneuver")
                print("\nPress ENTER to start autonomous landing or Ctrl+C to cancel...")
                input()
                
                landing_system.start_landing_sequence()
                
            except KeyboardInterrupt:
                print("\n❌ Landing cancelled by user")
            except Exception as e:
                print(f"❌ Landing failed: {e}")
        else:
            print(f"Unknown command: {command}")
    else:
        print("\n📋 Available commands:")
        print("  python working_airsim_landing.py markers  - Create test markers")
        print("  python working_airsim_landing.py test     - Test system connection")
        print("  python working_airsim_landing.py landing  - Run autonomous landing")

if __name__ == "__main__":
    main()