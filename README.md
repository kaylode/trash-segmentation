# Clothes Segmentation
 Project for studying

## To Do:
- [ ]  1. Analyze training data 
- [ ]  2. Add custom dataset
- [ ]  3. Preprocess data (data cleaning)
- [ ]  4. Build model
- [ ]  5. Train / Evaluate
- [ ]  6. Predict


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
  -	Deep Fashion 2: https://github.com/switchablenorms/DeepFashion2

## **Pretrained Weights (on COCO test-dev)**
	
Model | Image Size | Backbone | FPS | mAP | Weights
--- | --- | --- | --- | --- |--- 
YOLACT | 550 x 550 | Resnet101-FPN | 33.5 | 29.8 | [yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view)
YOLACT | 700 x 700 | Resnet101-FPN | 23.6 | 31.2 | [yolact_im700_54_800000.pth](https://drive.google.com/file/d/1lE4Lz5p25teiXV-6HdTiOJSnS7u7GBzg/view)
YOLACT++ | 550 x 550 | Resnet50-FPN | 33.5 | 34.1 | [yolact_plus_resnet50_54_800000.pth](https://drive.google.com/file/d/1ZPu1YR2UzGHQD0o1rEqy-j5bmEm3lbyP/view)


## **Referenced Codes:**
  -	https://github.com/dbolya/yolact

## **Guide Video:**
- **COCO Annotation Format Guide**
[![Watch the video](https://img.youtube.com/vi/h6s61a_pqfM/maxresdefault.jpg)](https://www.youtube.com/watch?v=h6s61a_pqfM)
	
# Hướng dẫn cài:
- Clone project về
- Vào link này, tải dataset và annotiation về: 
	- [Link dataset](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok) (Pass unzip: 2019Deepfashion2**) 
	- [Link annotation](https://drive.google.com/file/d/1kVBKLII2Q4KLof1DfUKAwQoo3x27b6Dp/view?usp=sharing)
- Giải nén dataset vào thư mục dataset/deepfashion, annotation vào dataset/deepfashion/train
- Tải 3 file pretrained model ở trên về, giải nén vào thư mục weights
- Mở file deep_fashion.ipynb bằng colab rồi chạy thử
