import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(100)


class detector:
    def __init__(self):
        self.classList = None
        self.colorList = None
        self.cacheDir = None
        self.fileName = None
        self.modelName = None

    def readClasses(self, classPath):
        with open(classPath, 'rb') as f:
            self.classList = f.read().splitlines()

        self.colorList = np.random.uniform(0, 255, len(self.classList))

        print(len(self.classList), len(self.colorList))

    def downloadModel(self, modelURL):
        self.fileName = modelURL.split("/")[-1]
        self.modelName = self.fileName.split(".")[0]

        self.cacheDir = "./pretrained_models"
        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(fname=self.fileName, origin=modelURL, extract=True, cache_dir=self.cacheDir, cache_subdir="checkpoints"
                 )

    def loadModel(self):
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName,
                                                      "saved_model"))
        print("Model Loaded")

    def boundingBox(self, image):
        imgArr = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inpTens = tf.convert_to_tensor(imgArr, dtype=tf.uint8)
        inpTens = inpTens[tf.newaxis, ...]

        detections = self.model(inpTens)
        print(detections)
        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape
        bboxIDs = tf.image.non_max_suppression(bboxs, classScores, max_output_size=20, iou_threshold=0.5,
                                               score_threshold=0.5)

        if len(bboxIDs) != 0:
            for i in bboxIDs:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(classScores[i] * 100)
                classIndex = classIndexes[i]

                classLabelText = self.classList[classIndex]
                classColor = self.colorList[classIndex]
                dispText = "{}: {}%".format(classLabelText, classConfidence)

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                cv2.rectangle(image, (xmin, ymin), (ymin, ymax), color=classColor, thickness=2)
                cv2.putText(image, dispText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, color=classColor)

        return image

    def predictImage(self, imagePath):
        image = cv2.imread(imagePath)
        imageResized = cv2.resize(image, (480, 480))
        predictedImg = self.boundingBox(imageResized)

        cv2.imwrite(self.modelName + ".jpg", predictedImg)
        cv2.imshow("Res", predictedImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def predVid(self, vidPath):
        cap = cv2.VideoCapture(vidPath)

        if not cap.isOpened():
            print("LOL")

        ret, frame = cap.read()

        while ret:
            bboxImg = self.boundingBox(frame)
            cv2.imshow("Res", bboxImg)
            key = cv2.waitKey(1) & 0xFF
            if key == "q":
                break
            ret, frame = cap.read()
        cv2.destroyAllWindows()
