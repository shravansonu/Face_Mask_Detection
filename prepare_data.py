import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall("object"):
        label = obj.find("name").text
        label_id = CLASSES.index(label)

        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_id)

    return boxes, labels

images = []
bboxes = []
labels = []

image_dir = "data/images"
ann_dir = "data/annotations"

for file in os.listdir(image_dir):
    if file.endswith(".png") or file.endswith(".jpg"):
        img_path = os.path.join(image_dir, file)
        xml_path = os.path.join(ann_dir, file.replace(".png", ".xml").replace(".jpg", ".xml"))

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        boxes, lbls = parse_xml(xml_path)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        for box, lbl in zip(boxes, lbls):
            xmin, ymin, xmax, ymax = box

            xmin /= w
            ymin /= h
            xmax /= w
            ymax /= h

            images.append(img)
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(lbl)

images = np.array(images)
bboxes = np.array(bboxes)
labels = np.eye(len(CLASSES))[labels]

X_train, X_temp, yb_train, yb_temp, yl_train, yl_temp = train_test_split(
    images, bboxes, labels, test_size=0.3, random_state=42)

X_val, X_test, yb_val, yb_test, yl_val, yl_test = train_test_split(
    X_temp, yb_temp, yl_temp, test_size=0.5)

np.savez("data/dataset.npz",
         X_train=X_train, X_val=X_val, X_test=X_test,
         yb_train=yb_train, yb_val=yb_val, yb_test=yb_test,
         yl_train=yl_train, yl_val=yl_val, yl_test=yl_test)

print("Dataset prepared successfully")
