import cv2
import matplotlib as plt
import numpy as np
import imutils
from PIL import Image
import time
from edgetpu.detection.engine import DetectionEngine

model = DetectionEngine(
    '/usr/share/edgetpu/examples/models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
labels = {}
for row in open('imagenet_labels.txt'):
    # unpack the row and update the labels dictionary
    (classID, label) = row.strip().split(" ", maxsplit=1)
    label = label.strip().split(",", maxsplit=1)[0]
    labels[int(classID)] = label
cam = cv2.VideoCapture(2)
while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=500)
    canvas = frame.copy()

    frame = Image.fromarray(frame[:, :, ::-1])
    start = time.time()
    results = model.detect_with_image(frame, threshold=0.3, keep_aspect_ratio=True, relative_coord=False)
    end = time.time()
    for r in results:
        box = r.bounding_box.flatten().astype("int")
        (startX, startY, endX, endY) = box
        cv2.rectangle(canvas, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        text = "{}: {:.2f}%".format(str(r.label_id), r.score * 100)
        cv2.putText(canvas, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(1 / (end - start))
    cv2.imshow("Result", canvas)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
