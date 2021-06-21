import cv2
import datetime
import imutils
import numpy as np
from centroid import CentroidTracker
from collections import defaultdict

proto = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=proto, caffeModel=model)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("int")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("pase")


def main():
    #cap = cv2.VideoCapture('videos/testvideo2.mp4')
    vid = cv2.VideoCapture('videos/video6.mp4')
    total_frames = 0
    cen= defaultdict(list)
    object_id_list = []

    while True:
        ret, frame = vid.read()
        frame = imutils.resize(frame, width=600)
        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        bounding = np.array(rects)
        bounding = bounding.astype(int)
        rects = non_max_suppression_fast(bounding, 0.5)

        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 255, 0), -1)

            cen[objectId].append((cX, cY))
            if objectId not in object_id_list:
                object_id_list.append(objectId)
                start_pt = (cX, cY)
                end_pt = (cX, cY)
                cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
            else:
                l = len(cen[objectId])
                for pt in range(len(cen[objectId])):
                    if not pt + 1 == l:
                        start_pt = (cen[objectId][pt][0], cen[objectId][pt][1])
                        end_pt = (cen[objectId][pt + 1][0], cen[objectId][pt + 1][1])
                        cv2.line(frame, start_pt, end_pt, (255, 0, 0), 1)
                        
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            text = "Persona {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
   
        cv2.imshow("Ventana", frame)
        key = cv2.waitKey(1)
        
        if key == 27:
            break

    cv2.destroyAllWindows()


main()
