# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import collections
import common
import cv2
import numpy as np
import os
from PIL import Image
import re
import tflite_runtime.interpreter as tflite
import time

from application_util import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

def get_packed_output(interpreter, score_threshold, top_k, image_scale=1.0):

    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    top_k_scores_id = np.argpartition(np.array(scores), top_k)[-top_k:]
    top_k_scores = scores[top_k_scores_id]
    threshold_pass = top_k_scores_id[top_k_scores >= score_threshold]

    boxes, scores, class_ids = boxes[threshold_pass], scores[threshold_pass], class_ids[threshold_pass]

    return [boxes, scores, class_ids, count]


def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

def convert_boxes(image, boxes):
    returned_boxes = []
    for box in boxes:
        box[0] = (box[0] * image.shape[1]).astype(int)
        box[1] = (box[1] * image.shape[0]).astype(int)
        box[2] = (box[2] * image.shape[1]).astype(int)
        box[3] = (box[3] * image.shape[0]).astype(int)
        box[2] = int(box[2]-box[0])
        box[3] = int(box[3]-box[1])
        box = box.astype(int)
        box = box.tolist()
        if box != [0,0,0,0]:
            returned_boxes.append(box)
    return returned_boxes

def convert_boxes_PIL(image, boxes):
    returned_boxes = []
    for box in boxes:
        box[0] = (box[0] * image.width).astype(int)
        box[1] = (box[1] * image.height).astype(int)
        box[2] = (box[2] * image.width).astype(int)
        box[3] = (box[3] * image.height).astype(int)
        box[2] = int(box[2]-box[0])
        box[3] = int(box[3]-box[1])
        box = box.astype(int)
        box = box.tolist()
        if box != [0,0,0,0]:
            returned_boxes.append(box)

    return returned_boxes

def get_features(interpreter, patches):
    
    #patches je lista ndarray objekata, treba pretvoriti u tenzor istih dimenzija ili sliku po sliku ubacivat
    features = []
    for patch in patches:
        if 0 in patch.shape:
            features.append(None)
            continue

        common.set_input(interpreter, Image.fromarray(patch))
        interpreter.invoke()
        feature = common.output_tensor(interpreter, 0)
        features.append(feature)

    return features


def extract_image_patches(pil_image, bboxes):
    image = np.array(pil_image)
    patches = []
    #print("bbox len is {}".format(len(bboxes)))
    for bbox in bboxes:
        bbox = np.array(bbox)

        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)

        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            patches.append(None)
            continue
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        #image = cv2.resize(image, tuple(patch_shape[::-1]))
        patches.append(image)

    return np.array(patches)


def process_objs(cv2_im, objs, labels):
    height, width, channels = cv2_im.shape

    patches = []

    for obj in objs:
        #calculate detected object bounding box
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        
        #draw rectangle to the image
        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        
        #extract image from rectangle in seperate list
        if x0<x1:
            patch = cv2_im[x0:x1, y0:y1, :]
        else:
            patch = cv2_im[x1:x0, y1:y0, :]

        patches.append(patch)

    return cv2_im, patches


def append_objs_to_img(cv2_im, objs, labels):
    height, width, channels = cv2_im.shape
    #print("appending objs, there is {} objects".format(len(objs)))
    #print(type(objs[0]))
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

def main():
    default_model_dir = './all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_feature_encoder = "./all_models/feature_encoder_mobilenet_128x128_headless_edge_tpu.tflite"
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=10,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.25,
                        help='classifier score threshold')
    parser.add_argument('--headless', help='run the script without output display', action='store_true')
    parser.add_argument('--video', help='use video instead of camera input')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = common.make_interpreter(args.model)
    interpreter_features = common.make_interpreter(default_feature_encoder)
    interpreter.allocate_tensors()
    interpreter_features.allocate_tensors()
    labels = load_labels(args.labels)

    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.8

    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.camera_idx)
    st = time.time()
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cnt += 1
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)


        common.set_input(interpreter, pil_im)
        interpreter.invoke()
        objs = get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k)

        cv2_im, patches = process_objs(cv2_im, objs, labels)
        if not args.headless:
            cv2.imshow('frame', cv2_im)
        features = get_features(interpreter_features, patches)

        detections = [Detection(obj.bbox, obj.score, obj.id, feature) for obj, feature in
              zip(objs, features) if feature is not None]

        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        tracker.predict()
        tracker.update(detections)

        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

        current_count = int(0)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    elapse = time.time()-st
    print("average fps is {}".format(  elapse/cnt  ))

if __name__ == '__main__':
    main()
