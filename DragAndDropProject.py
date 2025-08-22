import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1200)  # width
cap.set(4, 1000)  # height
cap.set(10, 100)  # brightness

detector = HandDetector(detectionCon=0.8)


class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size
        self.color = (255, 0, 255)  # default color

    def update(self, cursor, grabbing):
        cx, cy = self.posCenter
        w, h = self.size

        # If finger inside rectangle
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            if grabbing:  # grab mode
                self.posCenter = cursor
                self.color = (0, 255, 0)  # green when grabbing
            else:
                self.color = (255, 0, 255)  # purple when not grabbing
        else:
            if not grabbing:
                self.color = (255, 0, 255)


# Create multiple draggable rectangles
rectList = []
for x in range(5):
    rectList.append(DragRect([x * 250 + 150, 150]))

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands
    hands, img = detector.findHands(img)
    lmList = []

    if hands:
        lmList = hands[0]['lmList']

    if lmList:
        cursor = lmList[8][:2]  # index fingertip [x,y]
        length, _, img = detector.findDistance(lmList[8][:2], lmList[12][:2], img)

        # Grab mode if fingers close
        grabbing = length < 40

        # Update all rectangles
        for rect in rectList:
            rect.update(cursor, grabbing)

    # Transparent overlay
    imgNew = np.zeros_like(img, np.uint8)

    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size

        # Draw filled rectangle on transparent layer
        cv2.rectangle(imgNew, (int(cx - w // 2), int(cy - h // 2)),
                      (int(cx + w // 2), int(cy + h // 2)),
                      rect.color, cv2.FILLED)

        # Draw corner rectangle on base image
        cvzone.cornerRect(img, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    # Blend images (add transparency effect)
    alpha = 0.3
    mask = imgNew.astype(bool)
    out = img.copy()
    out[mask] = cv2.addWeighted(img, 1 - alpha, imgNew, alpha, 0)[mask]

    # Show result
    cv2.imshow("Image", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
