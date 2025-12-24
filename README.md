# Face Mask Detection using CNN

This project implements an end-to-end face mask detection system using
TensorFlow and MobileNetV2.

## Features
- Pascal VOC XML parsing
- CNN-based bounding box regression
- Mask classification (with_mask, without_mask, incorrect_mask)
- IoU@0.5 evaluation
- Streamlit & OpenCV deployment
- TFLite optimization

## How to Run
```bash
python prepare_data.py
python train_model.py
python evaluate_and_visualize.py
streamlit run app_streamlit.py
