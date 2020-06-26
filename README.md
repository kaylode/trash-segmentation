# Real-Time Trash Instaces Segmentation
 Project for studying

**Introduction**
![alt text](https://miro.medium.com/max/1400/0*QeOs5RvXlkbDkLOy.png)

![alt text](https://miro.medium.com/max/1400/1*8xmRyJa0Plkp0xByomvb7A.png)
- Yolact quan trọng vấn đề tốc độ hơn chính xác (độ chính xác giảm không nhiều, nhưng tốc độ vượt trội hơn so với các model khác)
## **Referenced Papers:** 

1. **YOLACT: Real-time Instance Segmentation**
    - Author: Daniel Bolya
    - Cited by 45
    - Link: https://arxiv.org/abs/1904.02689

2.	**YOLACT++: Better Real-time Instance Segmentation (2019)**
    -	Author: Daniel Bolya
    -	Cited by 2
    -	Link: https://arxiv.org/abs/1912.06218

## **Datasets:**
  -	TACO (Trash Annotations in Context)	: https://github.com/pedropro/TACO
  -	COCO 					: https://cocodataset.org/#download
## **Pretrained Weights**
	
Model | Image Size | Backbone | Dataset | Weights
--- | --- | --- | --- | --- 
YOLACT | 550 x 550 | Resnet101-FPN | COCO | [yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view)
YOLACT | 550 x 550 | Resnet101-FPN | TACO | [yolact_base_10_16500.pth](https://drive.google.com/file/d/1XuCem1VaEv4X_IZr1ccgBwoJ_pWM8yRE/view?usp=sharing)

Backbone Resnet101: [Link](https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view?usp=sharing)


## **Referenced Codes:**
  -	Yolact: https://github.com/dbolya/yolact

## **Guide Video:**
- **COCO Annotation Format Guide**
[![Watch the video](https://img.youtube.com/vi/h6s61a_pqfM/maxresdefault.jpg)](https://www.youtube.com/watch?v=h6s61a_pqfM)
	
# Hướng dẫn cài đặt và sử dụng trên colab
- Vào link này, thêm lối tắt drive của dataset và annotiation về drive của mình: 
	- [Link dataset và annotation của TACO](https://drive.google.com/file/d/1Ol3OcfjAfRul0lxKuV6ezaKP89HXqDMz/view?usp=sharing)
- Thêm lối tắt 3 file pretrained weights ở trên về drive của mình

- Tải [file colab](https://colab.research.google.com/drive/1ZVlF6K5HfcsS_G1tW_esY542n80CZOn0?authuser=1#scrollTo=fdHPLAp4xPKO) và chạy thử
