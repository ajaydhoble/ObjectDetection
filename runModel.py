from detector import *
import tensorflow as tf

classFile = "coco.names"
modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x6" \
           "40_coco17_tpu-8.tar.gz"
imgPath = "images/10.jpg"


d = detector()
d.readClasses(classFile)
d.downloadModel(modelURL)
d.loadModel()

d.predictImage(imgPath)
#d.predVid(0)
