# Bumblebee Perception Paketi

TEKNOFEST Robotaksi – Binek Otonom Araç Yarışması (Hazır Araç kategorisi) için geliştirilen ROS 2 perception (algı) paketi. Kamera, segmentasyon maskesi ve LIDAR verilerini kullanarak nesne tespiti ve şerit tespiti yapar.

## Node'lar

| Node | Dosya | Görevi |
|---|---|---|
| `inference_node` | `object_detection_node.py` | YOLOv8 ile 19 sınıf nesne tespiti (trafik işaretleri, yaya, ışıklar) |
| `teknofest_lane_node` | `lane_detection_node.py` | Segmentasyon maskesi üzerinden şerit tespiti ve şerit merkezi hatası hesaplama |

## Topic'ler

**inference_node**
- Dinler: `/camera/image_raw`, `/seg_mask`, `/scan`
- Yayınlar: `/objects` (JSON tespit listesi), `/pedestrian_detected` (Bool)

**teknofest_lane_node**
- Dinler: `/mission_state`, `/seg_mask`, `/scan`
- Yayınlar: `/lane_detection` → `[error, angle, left_distance, right_distance, violation]`

## Güvenlik

Her iki node da `/scan` üzerinden LIDAR verisini izler; tanımlı bir mesafenin altında engel algılandığında işlemeyi geçici olarak durdurur.

## Gereksinimler

```bash
pip install ultralytics opencv-python numpy
```
- ROS 2 (Humble veya üstü)
- `rclpy`, `cv_bridge`, `sensor_msgs`, `std_msgs`
- Eğitilmiş YOLOv8 ağırlık dosyası (`best.pt`)

## Çalıştırma

```bash
ros2 run perception_package object_detection_node
ros2 run perception_package lane_detection_node
```

## Ekip

TEKNOFEST Robotaksi – Binek Otonom Araç Yarışması, Hazır Araç Kategorisi
Ankara Üniversitesi
