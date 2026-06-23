import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

DEBUG_MODE = True #bilgisayarda kodu test etmek için True, aracı çalıştırırken False olmalı!!!!

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')
        self.bridge = CvBridge() # ROS2yi OpenCV formatına çevirir
        
        self.current_mission = "UNKNOWN" # Aracın görev modu

        # Dijital amartisör (ekrandaki durum yazıları için)
        self.smoothed_otsu = 150.0          
        
        self.tl_history = ["UNKNOWN"] * 5   

        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.mission_sub = self.create_subscription(String, '/mission_state', self.mission_callback, 10)
        
        self.mask_pub = self.create_publisher(Image, '/seg_mask', 10)            # Siyah/Beyaz şerit maskesi
        self.tl_pub = self.create_publisher(String, '/traffic_light_status', 10) # 'RED' / 'GREEN' / 'UNKNOWN'

        self.get_logger().info("Algilama (Perception) dugumu basariyla baslatildi.")

    def mission_callback(self, msg):
        #Görev kontrolcüsünden gelen 'LANE_FOLLOW', 'TRAFFIC_LIGHT' vb. komutları dinler
        self.current_mission = msg.data

    def image_callback(self, msg):
        #Kameradan her yeni fotoğraf karesi geldiğinde tetiklenen ana motor 
        try:
            # ROS'udan gelen veriyi [Yükseklik x Genişlik x 3] şeklinde OpenCV matrisine çevir
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            return

        height, width, _ = cv_image.shape
        
        # 1. BÖLÜM: TRAFİK IŞIĞI TESPİTİ 

        # Görüntünün sadece üst %45'lik kısmın
        top_half = cv_image[0:int(height*0.45), :] 
        hsv_top = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)

        #KIRMIZI RENK YELPAZESİ (OpenCV'de kırmızı 180'den 0'a kadar olduyğu için 2 parça yazılır)
        lower_red1 = np.array([0, 90, 60])    # Soluk ve parlayan kırmızılar dahil
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([165, 90, 60])  # Hafif pembe/mor gibi kırmızılar dahil
        upper_red2 = np.array([180, 255, 255])
        
        mask_red1 = cv2.inRange(hsv_top, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_top, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2) # İki kırmızı maskesini birleştir

        # --- YEŞİL RENK YELPAZESİ ---
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv_top, lower_green, upper_green)

        # 1. Adım: Bu karede piksel sayımı yap (Baraj: 120 piksel)
        anlik_oy = "UNKNOWN"
        if cv2.countNonZero(mask_red) > 120: 
            anlik_oy = "RED"
        elif cv2.countNonZero(mask_green) > 120: 
            anlik_oy = "GREEN"

        # 2. Adım: Oyu 5 karelik hafıza sandığına at, en eski kareyi sandıktan çöpe at
        self.tl_history.append(anlik_oy)
        self.tl_history.pop(0)

        # 3. Adım: SANDIK SONUCU (Son 5 karede en çok hangi kelime tekrar ettiyse resmi karar odur)
        resmi_isik_karari = max(set(self.tl_history), key=self.tl_history.count)

        # Kararı ROS Topic'ine fırlat
        tl_status = String()
        tl_status.data = resmi_isik_karari
        self.tl_pub.publish(tl_status)

        # 2. BÖLÜM: ŞERİT SEGMENTASYONU (CLAHE + Otsu + Low-Pass)
  
        # BGR görüntüyü HLS uzayına çevir ve sadece 'L' (Lightness/Parlaklık) kanalını çek
        hls = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        
        # CLAHE: Görüntüyü 8x8 piksellik küçük karelere bölüp her karenin kontrastını lokal patlatır (Gölgeler için)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl_l = clahe.apply(l_channel)

        # Otsu'ya "Matematiksel eşiği bul ama 150'nin altına düşersen 150 kabul et" diyo (şeritleri ayırt etmek için)
        otsu_tahmini, _ = cv2.threshold(cl_l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        guvenli_esik = max(otsu_tahmini, 150)
        
        # LOW-PASS FILTER: Eşik aniden fırlamasın; eski değerin %85'i ile yeni değerin %15'ini harmanla
        self.smoothed_otsu = (0.85 * self.smoothed_otsu) + (0.15 * guvenli_esik)

        # Gerçek siyah-beyaz kesim işlemini bu yumuşatılmış sayıyla yap
        _, seg_mask = cv2.threshold(cl_l, int(self.smoothed_otsu), 255, cv2.THRESH_BINARY)

        # Eğer görev "Şerit Takibi" ise, görüntünün üst yarısını zifiri siyaha boya
        if self.current_mission == "LANE_FOLLOW":
            seg_mask[0:height//2, :] = 0
            
        # Temizlenmiş siyah-beyaz maskeyi ROS2'ye gönder
        try:
            ros_mask = self.bridge.cv2_to_imgmsg(seg_mask, encoding="mono8")
            self.mask_pub.publish(ros_mask)
        except Exception as e:
            pass

        # 3. BÖLÜM:DEBUG_MODE = True iken çalışır
        if DEBUG_MODE:
            debug_frame = cv_image.copy() # Orijinal resmin kopyasını al
            
            # Bulunan beyaz şerit piksellerini ana ekranda fosforlu yeşile boya
            debug_frame[seg_mask == 255] = [0, 255, 0] 

            #KIRMIZI IŞIKLARI KUTUYA AL
            # Görüntüdeki kırmızı adacıkların koordinatlarını bul
            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_red:
                if cv2.contourArea(cnt) > 60: # 60 pikselden küçük uzaktaki tabelaları es geç
                    x, y, w, h = cv2.boundingRect(cnt) # Adacığın GPS en-boy verisi
                    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 2) # Kırmızı Kutu çiz
                    cv2.putText(debug_frame, "KIRMIZI", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # YEŞİL IŞIKLARI KUTUYA AL 
            contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_green:
                if cv2.contourArea(cnt) > 60:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(debug_frame, "YESIL", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Ekranın sol üst köşesine canlı telemetri yazılarını bas
            cv2.putText(debug_frame, f"Isik: {resmi_isik_karari}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(debug_frame, f"Otsu: {int(self.smoothed_otsu)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_frame, f"Gorev: {self.current_mission}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Pencereleri ekranda göster
            cv2.imshow("Perception: Canli Analiz", debug_frame)
            cv2.imshow("Perception: ROS Ciktisi", seg_mask)
            cv2.waitKey(1) # Görüntünün ekranda kalması için 1 ms bekle

def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    try: 
        rclpy.spin(node) 
    except KeyboardInterrupt: 
        pass             
    finally: 
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': 
    main()