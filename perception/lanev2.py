import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy
from sensor_msgs.msg import Image,LaserScan
from std_msgs.msg import String,Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import threading

class TeknofestLaneNode(Node):
    def __init__(self):
        super().__init__('teknofest_lane_node')
        self.bridge = CvBridge()
        self.mission_active = False
        self.emergency = False
        self.lock = threading.Lock()
        self.latest_mask = None
        self.latest_scan = None
        self.STOP_DISTANCE = 0.5
        self.PIXEL_TO_METER = 0.003
        self.VIOLATION_THRESHOLD = 0.30
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST,depth=10)
        self.publisher_lane = self.create_publisher(Float32MultiArray,'/lane_detection',10)
        self.create_subscription(String,'/mission_state',self.mission_callback,10)
        self.create_subscription(Image,'/seg_mask',self.mask_callback,sensor_qos)
        self.create_subscription(LaserScan,'/scan',self.scan_callback,10)
        self.timer = self.create_timer(0.033,self.process_loop)
        self.get_logger().info("Bumblebee Lane Detection Node Stage 2 Active")

    def mission_callback(self,msg):
        self.mission_active = (msg.data == "RUN")

    def mask_callback(self,msg):
        try:
            mask = self.bridge.imgmsg_to_cv2(msg,"passthrough")
            with self.lock: self.latest_mask = mask
        except Exception as e: self.get_logger().error(str(e))

    def scan_callback(self,msg):
        ranges = msg.ranges
        n = len(ranges)
        if n == 0: return
        front_sectors = list(ranges[n*11//12:]) + list(ranges[:n//12])
        valid_ranges = [r for r in front_sectors if msg.range_min < r < msg.range_max]
        with self.lock:
            self.latest_scan = msg
            was_emergency = self.emergency
            self.emergency = bool(valid_ranges and min(valid_ranges) < self.STOP_DISTANCE)
            if self.emergency and not was_emergency: self.get_logger().warn("LIDAR GÜVENLİK: Engel var!")
            elif not self.emergency and was_emergency: self.get_logger().info("LIDAR GÜVENLİK: Yol temiz.")

    def apply_IPM(self,image):
        h,w = image.shape[:2]
        src = np.float32([[w*0.02,h*0.98],[w*0.98,h*0.98],[w*0.38,h*0.60],[w*0.62,h*0.60]])
        dst = np.float32([[w*0.15,h],[w*0.85,h],[w*0.15,0],[w*0.85,0]])
        matrix = cv2.getPerspectiveTransform(src,dst)
        return cv2.warpPerspective(image,matrix,(640,480))

    def sliding_window_search(self,binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:],axis=0)
        midpoint = int(histogram.shape[0]//2)
        if np.max(histogram) < 10: return None,None,None,None
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        nwindows,margin,minpix = 10,80,40
        window_height = int(binary_warped.shape[0]//nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy,nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        leftx_current,rightx_current = leftx_base,rightx_base
        left_lane_inds,right_lane_inds = [],[]
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0]-(window+1)*window_height
            win_y_high = binary_warped.shape[0]-window*window_height
            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= leftx_current - margin) & (nonzerox < leftx_current + margin)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= rightx_current - margin) & (nonzerox < rightx_current + margin)).nonzero()[0]
            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)
            if len(good_left) > minpix: leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > minpix: rightx_current = int(np.mean(nonzerox[good_right]))
        try: return nonzerox[np.concatenate(left_lane_inds)],nonzeroy[np.concatenate(left_lane_inds)],nonzerox[np.concatenate(right_lane_inds)],nonzeroy[np.concatenate(right_lane_inds)]
        except: return None,None,None,None

    def process_loop(self):
        with self.lock:
            if not self.mission_active or self.emergency: return
            if self.latest_mask is None: return
            mask = self.latest_mask.copy()
        mask = cv2.resize(mask,(640,480))
        h,w = mask.shape[:2]
        warped = self.apply_IPM(mask)
        _,binary_mask = cv2.threshold(warped,1,255,cv2.THRESH_BINARY)
        try:
            lx,ly,rx,ry = self.sliding_window_search(binary_mask)
            if lx is not None and len(lx) > 30 and len(rx) > 30:
                l_fit = np.polyfit(ly,lx,2)
                r_fit = np.polyfit(ry,rx,2)
                ploty = np.linspace(0,h-1,15)
                left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
                right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]
                
                left_edge = left_fitx[-1]
                right_edge = right_fitx[-1]
                lane_center = (left_edge + right_edge)/2
                car_center = w//2
                error = float(lane_center - car_center)
                angle = math.degrees(math.atan2(error,h))
                
                left_distance = abs(car_center - left_edge)*self.PIXEL_TO_METER
                right_distance = abs(right_edge - car_center)*self.PIXEL_TO_METER
                
                violation = 0.0
                if left_distance < self.VIOLATION_THRESHOLD or right_distance < self.VIOLATION_THRESHOLD: violation = 1.0
                
                output = Float32MultiArray()
                output.data = [error,angle,left_distance,right_distance,violation]
                self.publisher_lane.publish(output)
        except Exception: pass

def main(args=None):
    rclpy.init(args=args)
    node = TeknofestLaneNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()