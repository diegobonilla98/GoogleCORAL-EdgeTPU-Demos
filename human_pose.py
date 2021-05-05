import cv2
import matplotlib as plt
import numpy as np
import imutils
from PIL import Image
import time
from edgetpu.basic.edgetpu_utils import ListEdgeTpuPaths, EDGE_TPU_STATE_UNASSIGNED
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.utils import dataset_utils, image_processing
import collections
import math
import enum

print(ListEdgeTpuPaths(EDGE_TPU_STATE_UNASSIGNED))
engine = BasicEngine('posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
_, height, width, _ = engine.get_input_tensor_shape()

cam = cv2.VideoCapture(2)
while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, (width, height), cv2.INTER_NEAREST)
    canvas = frame.copy()

    start = time.time()
    _, result = engine.run_inference(frame[:, :, ::-1].flatten())
    end = time.time()

    for i in range(0, len(result)-1, 2):
        cv2.circle(canvas, (int(result[i+1]), int(result[i])), 10, (0, 255, 0), -1)

    print(1 / (end - start))
    cv2.imshow("Result", canvas)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
