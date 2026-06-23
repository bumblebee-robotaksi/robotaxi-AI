# Algılama Modülü (Perception)

YOLOv8n tabanlı nesne tespiti, şerit tespiti ve yol segmentasyonu.

## İçerik

| Dosya | Açıklama |
|---|---|
| `object-det.py` | YOLOv8n tabanlı 19 sınıflı nesne tespiti. LiDAR (`/scan`) ile mesafe füzyonu içerir. |
| `lanev2.py` | Şerit tespiti |
| `segmentation_node.py` | ROS 2 node — yol segmentasyon maskesini (`/seg_mask`) yayınlar |
| `best.pt` | Eğitilmiş YOLOv8n model ağırlıkları |
| `results/` | Eğitim sonuçları (grafikler, karışıklık matrisi, örnek tahminler) |

## Tespit Edilen Sınıflar (19 sınıf)

| ID | Sınıf | ID | Sınıf |
|---|---|---|---|
| 0 | keep_left | 10 | no_right_turn |
| 1 | keep_right | 11 | park |
| 2 | no_entry | 12 | pedestrian |
| 3 | no_parking | 13 | red_light |
| 4 | crosswalk_sign | 14 | right |
| 5 | crosswalk | 15 | roundabout |
| 6 | go_ahead | 16 | stop_sign |
| 7 | green_light | 17 | tunnel |
| 8 | left | 18 | yellow_light |
| 9 | no_left_turn | | |

## Model Eğitim Sonuçları

### `results/results.png`
Eğitim/doğrulama kayıpları ve precision/recall/mAP metriklerinin epoch
bazında değişimi.
*(Doldurulacak: kaç epoch, en iyi mAP hangi noktada elde edildi)*

### `results/confusion_matrix_normalized.png`
Sınıf bazlı normalize edilmiş karışıklık matrisi.
*(Doldurulacak: en çok karıştırılan sınıflar ve olası neden)*

### `results/BoxPR_curve.png`
Precision-Recall eğrisi.
*(Doldurulacak: seçilen güven eşiği ve gerekçesi — bkz. KTR 7.1.3.4, 0.5 olarak belirtilmiş)*

### `results/val_batch0_pred.jpg`
Doğrulama setinden örnek model tahminleri.

## Veri Seti

*(Doldurulacak: kaynak — örn. Roboflow Universe, görüntü sayısı, augmentasyonlar)*

## Kullanım

```bash
python object-det.py --source <görüntü_veya_video_yolu>
ros2 run robotaksi_perception segmentation_node
```

## Bağımlılıklar

```bash
pip install ultralytics opencv-python
```