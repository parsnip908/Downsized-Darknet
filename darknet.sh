#!/bin/sh

./darknet detector test cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg

# ./darknet detector map cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights

