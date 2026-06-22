import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')
        self.bridge = CvBridge()
        
        # Aracın anlık görev durumu
        self.current_mission = "UNKNOWN"

        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.mission_sub = self.create_subscription(String, '/mission_state', self.mission_callback, 10)
        
        self.mask_pub = self.create_publisher(Image, '/seg_mask', 10) # Şerit maskesi (Siyah-Beyaz)
        self.tl_pub = self.create_publisher(String, '/traffic_light_status', 10) # Trafik ışığı durumu

        self.get_logger().info("Trafik Isigi Tespiti Basladi!")

    def mission_callback(self, msg):
        # Görev durumu değiştiğinde bu kod çalışır ve anlık görevi kaydeder
        self.current_mission = msg.data

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(str(e))
            return

        height, width, _ = cv_image.shape
        
        # 1. BÖLÜM: TRAFİK IŞIĞI TESPİTİ (Görüntünün sadece üst yarısına bakar)
        top_half = cv_image[0:height//2, :]
        hsv_top = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)

        # Kırmızı renk için HSV aralıkları 
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red1 = cv2.inRange(hsv_top, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_top, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2) # İki kırmızı maskesini birleştirir

        # Yeşil renk için HSV aralığı
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv_top, lower_green, upper_green)

        # Işık durumuna karar ver
        tl_status = String()
        if cv2.countNonZero(mask_red) > 500: 
            tl_status.data = "RED"
        elif cv2.countNonZero(mask_green) > 500: 
            tl_status.data = "GREEN"
        else:
            tl_status.data = "UNKNOWN" # Işık yoksa veya belirsizse

        self.tl_pub.publish(tl_status)

        # 2. BÖLÜM: ŞERİT SEGMENTASYONU 
        
        # Görüntüyü HLS renk uzayına çevir ve sadece L (Lightness/Parlaklık) kanalını al
        hls = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        
        # CLAHE algoritması ile kontrastı bölgesel olarak artır (Gölgeli yollarda şeritleri kaybetmemek için)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl_l = clahe.apply(l_channel)

        # Çok parlak pikselleri beyaz (255), diğer her şeyi siyah (0) 
        _, seg_mask = cv2.threshold(cl_l, 200, 255, cv2.THRESH_BINARY)

        # Eğer araç "Şerit Takip" modundaysa, gökyüzünü ve ufuk çizgisini siyah yap 
        # direksiyon kodu ilerideki alakasız beyazlıklardan etkilenmez.
        if self.current_mission == "LANE_FOLLOW":
            seg_mask[0:height//2, :] = 0
            
        try:
            ros_mask = self.bridge.cv2_to_imgmsg(seg_mask, encoding="mono8")
            self.mask_pub.publish(ros_mask)
        except Exception as e:
            self.get_logger().error(str(e))

def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    # Betul
