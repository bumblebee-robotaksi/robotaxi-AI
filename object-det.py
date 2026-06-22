import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2
import time
import threading
import json
import math
from ultralytics import YOLO
from std_msgs.msg import String, Bool

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
        self.model = YOLO('best.pt')
        
        self.PERSON_CLASS_ID = 12
        self.CROSSWALK_TRIGGER_IDS = {4, 5}
        
        self.CLASS_LABELS = {0: "keep_left",1: "keep_right",2: "no_entry",3: "no_parking",4: "crosswalk_sign",5: "crosswalk",6: "go_ahead",7: "green_light",8: "left",9: "no_left_turn",10: "no_right_turn",11: "park",12: "pedestrian",13: "red_light",14: "right",15: "roundabout",16: "stop_sign",17: "tunnel",18: "yellow_light"}
        self.SIGN_IDS = set(range(19)) 
        
        self.H_FOV = math.radians(60)
        self.STOP_DISTANCE = 0.4
        self.bridge = CvBridge()
        
        self.latest_frame = None
        self.latest_mask = None
        self.latest_scan = None
        self.emergency = False
        self.lock = threading.Lock()
        
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,history=HistoryPolicy.KEEP_LAST,depth=10)
        
        self.objects_pub = self.create_publisher(String, '/objects', 10)
        self.ped_pub = self.create_publisher(Bool, '/pedestrian_detected', 10)
        
        self.running = True
        self.infer_interval = 0.1
        self.last_infer_time = time.time()
        self.crosswalk_memory = 0
        self.crosswalk_hold = 8
        
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, sensor_qos)
        self.create_subscription(Image, '/seg_mask', self.mask_callback, sensor_qos)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        self.thread = threading.Thread(target=self.infer_loop, daemon=True)
        self.thread.start()
        self.get_logger().info("Bumblebee Perception Node: ALL 19 CLASSES FULLY ACTIVATED")

    def scan_callback(self, msg):
        ranges = msg.ranges
        n = len(ranges)
        if n == 0:
            return

        front = list(ranges[n * 11 // 12:]) + list(ranges[:n // 12])
        valid = [r for r in front if msg.range_min < r < msg.range_max]
        
        with self.lock:
            self.latest_scan = msg
            was_emergency = self.emergency
            self.emergency = bool(valid and min(valid) < self.STOP_DISTANCE)
            
            if self.emergency and not was_emergency:
                self.get_logger().warn("LIDAR: cok yakin engel, algi askiya aliniyor.")
            elif not self.emergency and was_emergency:
                self.get_logger().info("LIDAR: engel acildi, algi devam ediyor.")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        with self.lock:
            self.latest_frame = frame

    def mask_callback(self, msg):
        mask = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        with self.lock:
            self.latest_mask = mask

    def _safe_on_road(self, mask, frame_shape, cx, cy):
        if mask is None:
            return False
        fh, fw = frame_shape[:2]
        mh, mw = mask.shape[:2]
        if (mh, mw) != (fh, fw):
            mx = int(cx * mw / fw)
            my = int(cy * mh / fh)
        else:
            mx, my = cx, cy
        if 0 <= my < mh and 0 <= mx < mw:
            return bool(mask[my, mx] > 0)
        return False

    def get_distance_for_bbox(self, scan, cx, frame_width):
        if scan is None or scan.angle_increment == 0:
            return None
        
        angle = -((cx / frame_width) - 0.5) * self.H_FOV
        idx = int((angle - scan.angle_min) / scan.angle_increment)
        idx = max(0, min(idx, len(scan.ranges) - 1))
        
        start_idx = max(0, idx - 1)
        end_idx = min(len(scan.ranges) - 1, idx + 1)
        valid_ranges = [scan.ranges[i] for i in range(start_idx, end_idx + 1) if scan.range_min < scan.ranges[i] < scan.range_max]
        
        if valid_ranges:
            return float(round(min(valid_ranges), 2)) 
        return None

    def infer_loop(self):
        while self.running:
            with self.lock:
                is_emergency = self.emergency
            
            if is_emergency:
                time.sleep(0.1)
                self.last_infer_time = time.time()
                continue
                
            now = time.time()
            remaining = self.infer_interval - (now - self.last_infer_time)
            if remaining > 0:
                time.sleep(remaining)
                continue
                
            self.last_infer_time = time.time()
            
            with self.lock:
                if self.latest_frame is None:
                    continue
                frame = self.latest_frame.copy()
                mask = self.latest_mask.copy() if self.latest_mask is not None else None
                scan = self.latest_scan

            results = self.model.predict(frame, verbose=False, imgsz=640)
            
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                empty_msg = String()
                empty_msg.data = "[]"
                self.objects_pub.publish(empty_msg)
                
                ped_msg = Bool()
                ped_msg.data = False
                self.ped_pub.publish(ped_msg)
                continue
                
            boxes = results[0].boxes
            objects = []
            pedestrian_alert = False
            
            detected_ids = [int(b.cls[0].item()) for b in boxes if float(b.conf[0]) >= 0.5]
            
            if self.CROSSWALK_TRIGGER_IDS & set(detected_ids):
                self.crosswalk_memory = self.crosswalk_hold
            elif self.crosswalk_memory > 0:
                self.crosswalk_memory -= 1
                
            crosswalk_active = self.crosswalk_memory > 0
            
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy().tolist())
                cx = (x1 + x2) // 2
                check_y = y2 if cls_id == self.PERSON_CLASS_ID else (y1 + y2) // 2
                
                on_road = self._safe_on_road(mask, frame.shape, cx, check_y)
                label = self.CLASS_LABELS.get(cls_id, f"unknown_{cls_id}")
                distance = self.get_distance_for_bbox(scan, cx, frame.shape[1])
                
                if cls_id == self.PERSON_CLASS_ID:
                    h = y2 - y1
                    if on_road and (crosswalk_active or h > 80):
                        pedestrian_alert = True
                        
               
                if on_road or cls_id in self.SIGN_IDS:
                    objects.append({"class": label,"conf": float(round(conf, 2)), "bbox": [x1, y1, x2, y2],"on_road": on_road,"distance_m": distance})
                    
            obj_msg = String()
            obj_msg.data = json.dumps(objects, separators=(',', ':'))
            self.objects_pub.publish(obj_msg)
            
            ped_msg = Bool()
            ped_msg.data = pedestrian_alert
            self.ped_pub.publish(ped_msg)

    def destroy_node(self):
        self.running = False
        self.thread.join(timeout=2.0)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()