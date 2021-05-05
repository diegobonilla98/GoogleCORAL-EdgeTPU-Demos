import cv2
import matplotlib as plt
import numpy as np
import imutils
from PIL import Image
import time
from edgetpu.basic.basic_engine import BasicEngine
from edgetpu.utils import dataset_utils, image_processing


def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    indices = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((indices >> channel) & 1) << shift
        indices >>= 3
    return colormap


def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    colormap = create_pascal_label_colormap()
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
    return colormap[label]


engine = BasicEngine('/usr/share/edgetpu/examples/models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite')
_, height, width, _ = engine.get_input_tensor_shape()
algo = engine.required_input_array_size()

cam = cv2.VideoCapture(2)
while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, (width, height), cv2.INTER_NEAREST)

    start = time.time()
    _, result = engine.run_inference(frame[:, :, ::-1].flatten())
    result = np.reshape(result, (height, width))
    end = time.time()
    result = label_to_color_image(result.astype(int)).astype(np.uint8)

    print(1 / (end - start))
    cv2.imshow("Result", result)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
