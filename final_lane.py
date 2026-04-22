import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class TeknofestLaneNode(Node):
    def __init__(self):
        super().__init__('teknofest_lane_node')
        
        # ROS 2 Publishers & Subscribers
        self.publisher_lane = self.create_publisher(Float32MultiArray, '/lane_detection', 10)
        self.create_subscription(String, '/mission_state', self.mission_callback, 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        self.bridge = CvBridge()
        self.mission_active = False
        
        # Ubuntu pencere sabitleme
        cv2.namedWindow("Teknofest Takip", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Teknofest Takip", 640, 480)

    def mission_callback(self, msg):
        self.mission_active = (msg.data == "RUN")

    def apply_IPM(self, image):
        h, w = image.shape[:2]
        # O güzel koddaki tam koordinatlar:
        src = np.float32([[w*0.02, h*0.98], [w*0.98, h*0.98], [w*0.38, h*0.60], [w*0.62, h*0.60]])
        dst = np.float32([[w*0.15, h], [w*0.85, h], [w*0.15, 0], [w*0.85, 0]])
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, matrix, (640, 480))

    def sliding_window_search(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        midpoint = int(histogram.shape[0]//2)
        if np.max(histogram) < 10: return None, None, None, None

        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        nwindows, margin, minpix = 10, 80, 40
        window_height = int(binary_warped.shape[0]//nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
        
        leftx_current, rightx_current = leftx_base, rightx_base
        left_lane_inds, right_lane_inds = [], []

        for window in range(nwindows):
            win_y_low, win_y_high = binary_warped.shape[0]-(window+1)*window_height, binary_warped.shape[0]-window*window_height
            
            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= leftx_current - margin) & (nonzerox < leftx_current + margin)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= rightx_current - margin) & (nonzerox < rightx_current + margin)).nonzero()[0]
            
            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)
            
            if len(good_left) > minpix: leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > minpix: rightx_current = int(np.mean(nonzerox[good_right]))

        try:
            return nonzerox[np.concatenate(left_lane_inds)], nonzeroy[np.concatenate(left_lane_inds)], \
                   nonzerox[np.concatenate(right_lane_inds)], nonzeroy[np.concatenate(right_lane_inds)]
        except: return None, None, None, None

    def image_callback(self, msg):
        if not self.mission_active: return
        
        # 1. Görüntü Al ve İyileştir (CLAHE 2.5)
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]
        
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5) # O güzel kodun gizli ayarı
        frame_balanced = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
        
        # 2. IPM ve Maske
        warped = self.apply_IPM(frame_balanced)
        hls = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
        mask = cv2.inRange(hls, np.array([0, 150, 0]), np.array([180, 255, 80]))
        
        # 3. Şerit Bulma ve Görselleştirme (fillPoly)
        try:
            lx, ly, rx, ry = self.sliding_window_search(mask)
            if lx is not None and len(lx) > 30 and len(rx) > 30:
                l_fit = np.polyfit(ly, lx, 2)
                r_fit = np.polyfit(ry, rx, 2)
                
                ploty = np.linspace(0, h-1, 15) # Görselleştirme için 15 nokta yeterli
                left_fitx = l_fit[0]*ploty**2 + l_fit[1]*ploty + l_fit[2]
                right_fitx = r_fit[0]*ploty**2 + r_fit[1]*ploty + r_fit[2]
                
                # Sapma Hesabı (Videonun sonundaki mantık)
                lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
                error = float(lane_center - (w // 2))
                angle = math.degrees(math.atan2(error, h))

                # ROS Yayınla
                output = Float32MultiArray()
                output.data = [error, angle]
                self.publisher_lane.publish(output)

                # Şerit arasını yeşile boya (Senin beğendiğin o görsel etki)
                res_to_show = warped.copy()
                canvas = np.zeros_like(warped)
                ploty_full = np.linspace(0, h-1, h)
                left_full = l_fit[0]*ploty_full**2 + l_fit[1]*ploty_full + l_fit[2]
                right_full = r_fit[0]*ploty_full**2 + r_fit[1]*ploty_full + r_fit[2]
                
                pts_left = np.array([np.transpose(np.vstack([left_full, ploty_full]))])
                pts_right = np.array([np.flipud(np.transpose(np.vstack([right_full, ploty_full])))])
                pts = np.hstack((pts_left, pts_right))
                
                cv2.fillPoly(canvas, np.int_([pts]), (0, 255, 0))
                res_to_show = cv2.addWeighted(res_to_show, 0.7, canvas, 0.3, 0.0)
                
                cv2.imshow("Teknofest Takip", res_to_show)
                cv2.waitKey(1)
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = TeknofestLaneNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == "__main__":
    main()