import numpy as np
import tensorflow as tf
import cv2
import random

model = tf.keras.models.load_model("model/face_mask_detector.keras")
data = np.load("data/dataset.npz")

X_test = data["X_test"]
yb_test = data["yb_test"]
yl_test = data["yl_test"]

pred_boxes, pred_classes = model.predict(X_test)

def iou(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])

    return inter / (box1_area + box2_area - inter + 1e-6)

ious = [iou(pb, gb) for pb, gb in zip(pred_boxes, yb_test)]
print("Mean IoU @0.5:", np.mean(ious))

# visualize 10 images
for i in random.sample(range(len(X_test)), 10):
    img = (X_test[i] * 255).astype("uint8")
    h, w, _ = img.shape

    pb = pred_boxes[i]
    xmin, ymin, xmax, ymax = (pb * [w,h,w,h]).astype(int)

    cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
