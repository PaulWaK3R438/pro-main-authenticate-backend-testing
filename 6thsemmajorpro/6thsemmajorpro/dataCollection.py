from time import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

####################################
classID = 0  # 0 = fake, 1 = real  <-- CHANGE THIS BEFORE RUNNING
outputFolderPath = 'Dataset/DataCollect'
confidence = 0.8
save = True
blurThreshold = 35

debug = False
offsetPercentageW = 10

offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6
####################################

# Automatically set folder and label
if classID == 0:
    className = "fake"
else:
    className = "real"

savePath = f"{outputFolderPath}/{className}"
allPath = f"{outputFolderPath}/all"

# Create folders
os.makedirs(savePath, exist_ok=True)
os.makedirs(allPath, exist_ok=True)

print(f"Saving to: {savePath}  |  ClassID: {classID} ({className})")
print(f"Also saving to: {allPath}")

# Setup mediapipe face detector
model_path = "face_detector.tflite"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=confidence
)
detector = vision.FaceDetector.create_from_options(options)

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

while True:
    success, img = cap.read()
    if not success:
        break

    imgOut = img.copy()

    # Convert to mediapipe format
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    results = detector.detect(mp_image)

    listBlur = []
    listInfo = []

    if results.detections:
        for detection in results.detections:
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            score = detection.categories[0].score

            if score > confidence:

                # Offset
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)

                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)

                # Clamp
                x = max(0, x)
                y = max(0, y)
                w = max(0, w)
                h = max(0, h)

                # Blur check
                imgFace = img[y:y + h, x:x + w]
                if imgFace.size == 0:
                    continue

                cv2.imshow("Face", imgFace)

                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                listBlur.append(blurValue > blurThreshold)

                # Normalize
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2

                xcn = min(round(xc / iw, floatingPoint), 1)
                ycn = min(round(yc / ih, floatingPoint), 1)
                wn  = min(round(w / iw, floatingPoint), 1)
                hn  = min(round(h / ih, floatingPoint), 1)

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # Draw
                color = (0, 255, 0) if classID == 1 else (0, 0, 255)
                cv2.rectangle(imgOut, (x, y), (x + w, y + h), color, 3)

                cv2.putText(
                    imgOut,
                    f'{className.upper()} | Score: {int(score * 100)}% Blur: {blurValue}',
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # Save images and labels
        if save and all(listBlur) and listBlur != []:
            timeNow = str(time()).replace('.', '')

            # Save in class folder
            cv2.imwrite(f"{savePath}/{timeNow}.jpg", img)
            with open(f"{savePath}/{timeNow}.txt", 'w') as f:
                f.writelines(listInfo)

            # ALSO save in "all" folder
            cv2.imwrite(f"{allPath}/{timeNow}.jpg", img)
            with open(f"{allPath}/{timeNow}.txt", 'w') as f:
                f.writelines(listInfo)

    cv2.imshow("Image", imgOut)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()