#! /bin/bash

# make clean
make
rm -f testtimes.txt
touch testtimes.txt
for (( i = 0; i < $1; i++ )); do
	#statements
	./darknet detector test cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg 2> /dev/null | tail -7 | head -1 >> testtimes.txt
	tail -1 testtimes.txt
done
python3 teststats.py
