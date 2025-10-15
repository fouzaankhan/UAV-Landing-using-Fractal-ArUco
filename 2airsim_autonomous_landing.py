import airsim
import numpy as np
import cv2
import time
import math
import json
from enum import Enum
import threading
from queue import Queue
import logging

from fractal_aruco_detector import FractalArucoDetector
from camera_calibration import CameraCalibration
from pid_controller import LandingController

class LandingState(Enum):
    """Landing state machine states"""
    SEARCHING = "SEARCHING"
    APPROACHING = "APPROACHING" 
    POSITIONING = "POSITIONING"
    DESCENDING = "DESCENDING"
    LANDED = "LANDED"
    ABORTED = "ABORTED"

class AirSimAutonomousLanding:
    """
    Autonomous landing system for AirSim using fractal ArUco markers
    """
    
    def __init__(self, config_file="landing_config.json"):
        """
        Initialize the autonomous landing system
        
        Args:
            config_file (str): Configuration file path
        """
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize AirSim client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # Initialize components
        self.detector = FractalArucoDetector(
            dict_type=self.config['marker']['dict_type'],
            marker_size=self.config['marker']['size']
        )
        
        self.calibration = CameraCalibration()
        self._setup_camera_calibration()
        
        self.controller = LandingController()
        self._setup_controller_gains()
        
        # State management
        self.state = LandingState.SEARCHING
        self.landing_target = None
        self.detection_history = []
        self.max_detection_history = 5
        
        # Safety parameters
        self.min_altitude = self.config['safety']['min_altitude']
        self.max_search_time = self.config['safety']['max_search_time']
        self.landing_precision = self.config['landing']['precision_threshold']
        self.descent_rate = self.config['landing']['descent_rate']
        
        # Threading for image processing
        self.image_queue = Queue(maxsize=10)
        self.detection_queue = Queue(maxsize=5)
        self.processing_thread = None
        self.running = False
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("AirSim Autonomous Landing System initialized")
    
    def _load_config(self, config_file):
        """Load configuration from JSON file"""
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
                "precision_threshold": 0.2,
                "descent_rate": 0.3,
                "timeout": 300
            },
            "safety": {
                "min_altitude": 1.0,
                "max_search_time": 120,
                "obstacle_avoidance": True
            },
            "pid_gains": {
                "x": {"kp": 0.8, "ki": 0.1, "kd": 0.15},
                "y": {"kp": 0.8, "ki": 0.1, "kd": 0.15},
                "z": {"kp": 0.5, "ki": 0.05, "kd": 0.1},
                "yaw": {"kp": 0.5, "ki": 0.02, "kd": 0.08}
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key in loaded_config:
                        default_config[key].update(loaded_config[key])
                return default_config
        except FileNotFoundError:
            self.logger.info(f"Config file {config_file} not found, using defaults")
            # Save default config
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _setup_camera_calibration(self):
        """Setup camera calibration parameters"""
        camera_config = self.config['camera']
        self.camera_matrix, self.dist_coeffs = self.calibration.create_airsim_camera_params(
            image_width=camera_config['image_width'],
            image_height=camera_config['image_height'],
            fov_degrees=camera_config['fov_degrees']
        )
        self.logger.info("Camera calibration setup complete")
    
    def _setup_controller_gains(self):
        """Setup PID controller gains from config"""
        gains = self.config['pid_gains']
        for axis in ['x', 'y', 'z', 'yaw']:
            if axis in gains:
                g = gains[axis]
                self.controller.tune_gains(axis, g['kp'], g['ki'], g['kd'])
        self.logger.info("PID controller gains configured")
    
    def start_landing_sequence(self):
        """Start the autonomous landing sequence"""
        self.logger.info("Starting autonomous landing sequence")
        self.running = True
        
        # Start image processing thread
        self.processing_thread = threading.Thread(target=self._image_processing_loop)
        self.processing_thread.start()
        
        try:
            # Main control loop
            self._main_control_loop()
        except KeyboardInterrupt:
            self.logger.info("Landing sequence interrupted by user")
        except Exception as e:
            self.logger.error(f"Landing sequence failed: {e}")
            self.state = LandingState.ABORTED
        finally:
            self.running = False
            if self.processing_thread:
                self.processing_thread.join()
            self.client.landAsync().join()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
    
    def _main_control_loop(self):
        """Main control loop for landing sequence"""
        start_time = time.time()
        
        # Takeoff to search altitude
        self.logger.info("Taking off to search altitude")
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-self.config['landing']['search_altitude'], 2).join()
        
        while self.running and (time.time() - start_time) < self.config['landing']['timeout']:
            # State machine
            if self.state == LandingState.SEARCHING:
                self._handle_searching_state()
            elif self.state == LandingState.APPROACHING:
                self._handle_approaching_state()
            elif self.state == LandingState.POSITIONING:
                self._handle_positioning_state()
            elif self.state == LandingState.DESCENDING:
                self._handle_descending_state()
            elif self.state == LandingState.LANDED:
                self.logger.info("Landing completed successfully!")
                break
            elif self.state == LandingState.ABORTED:
                self.logger.error("Landing sequence aborted!")
                break
            
            time.sleep(0.1)  # 10Hz control loop
    
    def _image_processing_loop(self):
        """Image processing loop running in separate thread"""
        while self.running:
            try:
                # Get camera image from AirSim
                response = self.client.simGetImage(
                    self.config['camera']['name'], 
                    airsim.ImageType.Scene
                )
                
                if response:
                    # Convert to opencv format
                    img_1d = np.frombuffer(response, dtype=np.uint8)
                    img_rgb = cv2.imdecode(img_1d, cv2.IMREAD_COLOR)
                    
                    if img_rgb is not None:
                        # Detect fractal ArUco markers
                        detection_result = self.detector.detect_markers(img_rgb)
                        
                        if detection_result['ids'] is not None:
                            # Estimate poses
                            poses = self.detector.estimate_pose(
                                detection_result['corners'],
                                self.camera_matrix,
                                self.dist_coeffs
                            )
                            
                            # Get landing target information
                            target_info = self.detector.get_landing_target(
                                detection_result, poses, self.camera_matrix
                            )
                            
                            if target_info and not self.detection_queue.full():
                                self.detection_queue.put(target_info)
                
                time.sleep(0.05)  # 20Hz image processing
                
            except Exception as e:
                self.logger.error(f"Image processing error: {e}")
                time.sleep(0.1)
    
    def _handle_searching_state(self):
        """Handle searching state - look for landing markers"""
        # Check for detections
        if not self.detection_queue.empty():
            detection = self.detection_queue.get()
            self.landing_target = detection
            self.state = LandingState.APPROACHING
            self.logger.info(f"Marker detected! ID: {detection['marker_id']}, "
                           f"Distance: {detection['distance']:.2f}m")
        else:
            # Perform search pattern (simple hover for now)
            current_pose = self.client.simGetVehiclePose()
            # TODO: Implement spiral search pattern
    
    def _handle_approaching_state(self):
        """Handle approaching state - move to approach altitude above marker"""
        if self.landing_target:
            # Calculate target position above marker
            current_pose = self.client.simGetVehiclePose()
            
            # Move to approach altitude
            approach_alt = -self.config['landing']['approach_altitude']
            
            # Simple approach: move to marker location at approach altitude
            marker_world_pos = self._calculate_marker_world_position(self.landing_target)
            
            if marker_world_pos:
                self.client.moveToPositionAsync(
                    marker_world_pos[0],
                    marker_world_pos[1], 
                    approach_alt,
                    2.0
                ).join()
                
                self.state = LandingState.POSITIONING
                self.logger.info("Transitioning to positioning state")
    
    def _handle_positioning_state(self):
        """Handle positioning state - precise positioning above marker"""
        if not self.detection_queue.empty():
            self.landing_target = self.detection_queue.get()
        
        if self.landing_target:
            # Calculate control commands using PID
            current_pose = self.client.simGetVehiclePose()
            current_pos = [current_pose.position.x_val, current_pose.position.y_val, current_pose.position.z_val]
            
            # Target position (above marker)
            marker_world_pos = self._calculate_marker_world_position(self.landing_target)
            
            if marker_world_pos:
                target_pos = [marker_world_pos[0], marker_world_pos[1], current_pos[2]]
                
                # Update PID controller
                commands = self.controller.update(target_pos, current_pos)
                
                # Apply control commands
                self.client.moveByVelocityAsync(
                    commands['vx'],
                    commands['vy'],
                    0,  # Don't descend yet
                    1.0
                )
                
                # Check if positioned accurately enough
                if commands['in_position'] and commands['horizontal_distance'] < self.landing_precision:
                    self.state = LandingState.DESCENDING
                    self.logger.info("Positioned accurately, starting descent")
    
    def _handle_descending_state(self):
        """Handle descending state - controlled descent to landing"""
        if not self.detection_queue.empty():
            self.landing_target = self.detection_queue.get()
        
        current_pose = self.client.simGetVehiclePose()
        current_altitude = -current_pose.position.z_val
        
        if current_altitude <= self.min_altitude:
            # Land
            self.client.landAsync().join()
            self.state = LandingState.LANDED
            return
        
        if self.landing_target:
            # Continue positioning while descending
            current_pos = [current_pose.position.x_val, current_pose.position.y_val, current_pose.position.z_val]
            marker_world_pos = self._calculate_marker_world_position(self.landing_target)
            
            if marker_world_pos:
                target_pos = [marker_world_pos[0], marker_world_pos[1], current_pos[2]]
                commands = self.controller.update(target_pos, current_pos)
                
                # Apply positioning + descent
                self.client.moveByVelocityAsync(
                    commands['vx'],
                    commands['vy'],
                    self.descent_rate,  # Positive = descend
                    1.0
                )
        else:
            # Lost marker during descent - emergency hover
            self.client.hoverAsync()
            self.state = LandingState.SEARCHING
            self.logger.warning("Lost marker during descent, returning to search")
    
    def _calculate_marker_world_position(self, target_info):
        """Calculate marker position in world coordinates"""
        if target_info and target_info['tvec'] is not None:
            # Convert camera coordinates to world coordinates
            # This is a simplified calculation - in practice you'd need proper coordinate transforms
            current_pose = self.client.simGetVehiclePose()
            
            # Extract translation from marker detection (in camera frame)
            marker_cam_pos = target_info['tvec'].flatten()
            
            # Simple approximation: assume camera points down and convert to world frame
            # In practice, you'd use full rotation matrices
            marker_world_x = current_pose.position.x_val + marker_cam_pos[0]
            marker_world_y = current_pose.position.y_val - marker_cam_pos[1]  # Camera Y is opposite
            
            return [marker_world_x, marker_world_y, 0]  # Landing target at ground level
        
        return None
    
    def get_status(self):
        """Get current system status"""
        return {
            'state': self.state.value,
            'target_detected': self.landing_target is not None,
            'target_info': self.landing_target,
            'controller_status': self.controller.get_status()
        }

# Configuration setup helper
def create_default_config():
    """Create default configuration file"""
    config = {
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
            "precision_threshold": 0.2,
            "descent_rate": 0.3,
            "timeout": 300
        },
        "safety": {
            "min_altitude": 1.0,
            "max_search_time": 120,
            "obstacle_avoidance": True
        },
        "pid_gains": {
            "x": {"kp": 0.8, "ki": 0.1, "kd": 0.15},
            "y": {"kp": 0.8, "ki": 0.1, "kd": 0.15},
            "z": {"kp": 0.5, "ki": 0.05, "kd": 0.1},
            "yaw": {"kp": 0.5, "ki": 0.02, "kd": 0.08}
        }
    }
    
    with open('landing_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Default configuration created: landing_config.json")

if __name__ == "__main__":
    # Create default configuration
    create_default_config()
    
    # Initialize and run landing system
    landing_system = AirSimAutonomousLanding()
    landing_system.start_landing_sequence()