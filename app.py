
import cv2
import numpy as np
import time
import os
from pathlib import Path
from tensorflow.keras.models import load_model

# =====================================
# CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent

CASCADE_PATH = (
    BASE_DIR / "fer2013" /
    "haarcascade_frontalface_default.xml"
)

MODEL_PATH = BASE_DIR / "training" / "emotion_model.h5"

# CAMERA MODE:
# "auto"   = Try available local cameras
# "local"  = Use a specific camera index
# "iphone" = Use iPhone IP video stream

CAMERA_MODE = "local"
CAMERA_INDEX = 1

# For iPhone streaming app:
# Replace with the actual stream URL.
IPHONE_STREAM_URL = ""

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

# =====================================
# LOAD MODEL
# =====================================

print("Loading TensorFlow model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = load_model(
    str(MODEL_PATH),
    compile=False
)

print("Model loaded successfully!")

# =====================================
# LOAD FACE DETECTOR
# =====================================

faces = cv2.CascadeClassifier(
    str(CASCADE_PATH)
)

if faces.empty():
    raise RuntimeError(
        "Face detector could not be loaded."
    )

# =====================================
# CAMERA FUNCTIONS
# =====================================

def test_camera(camera, name):
    if not camera.isOpened():
        camera.release()
        return None

    # Allow camera to initialize
    time.sleep(1)

    for attempt in range(10):
        ret, frame = camera.read()

        if (
            ret
            and frame is not None
            and frame.size > 0
        ):
            print(f"Connected to {name}")
            return camera

        time.sleep(0.2)

    camera.release()
    return None


def open_local_camera(index):
    print(f"Testing local camera {index}...")

    camera = cv2.VideoCapture(
        index,
        cv2.CAP_AVFOUNDATION
    )

    return test_camera(
        camera,
        f"camera {index}"
    )


def find_camera():

    if CAMERA_MODE == "iphone":
        if not IPHONE_STREAM_URL:
            raise ValueError(
                "Set IPHONE_STREAM_URL first."
            )

        print("Connecting to iPhone stream...")

        camera = cv2.VideoCapture(
            IPHONE_STREAM_URL
        )

        result = test_camera(
            camera,
            "iPhone IP camera"
        )

        if result is not None:
            return result

        raise RuntimeError(
            "Cannot connect to iPhone stream."
        )

    if CAMERA_MODE == "local":
        camera = open_local_camera(
            CAMERA_INDEX
        )

        if camera is not None:
            return camera

        raise RuntimeError(
            f"Camera {CAMERA_INDEX} failed."
        )

    if CAMERA_MODE == "auto":
        # Test the first two local cameras.
        # Continuity Camera may not be
        # exposed as a numbered device.
        for index in [0, 1]:
            camera = open_local_camera(index)

            if camera is not None:
                return camera

        raise RuntimeError(
            "No working local camera found. "
            "Check camera permissions and "
            "test the camera in Photo Booth."
        )

    raise ValueError(
        f"Unknown CAMERA_MODE: {CAMERA_MODE}"
    )


# =====================================
# START CAMERA
# =====================================

video = find_camera()

print("Camera initialized successfully!")
print("Press Q to exit.")

# =====================================
# EMOTION RECOGNITION
# =====================================

failed_frames = 0

try:
    while True:

        ret, frame = video.read()

        if (
            not ret
            or frame is None
            or frame.size == 0
        ):
            failed_frames += 1

            if failed_frames >= 30:
                print(
                    "Camera stopped providing frames."
                )
                break

            time.sleep(0.1)
            continue

        failed_frames = 0

        # Mirror local webcam display
        if CAMERA_MODE != "iphone":
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        detected_faces = faces.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(40, 40)
        )

        for (x, y, w, h) in detected_faces:

            roi = gray[y:y+h, x:x+w]

            if roi.size == 0:
                continue

            roi = cv2.resize(
                roi,
                (48, 48),
                interpolation=cv2.INTER_AREA
            )

            roi = (
                roi.astype("float32") / 255.0
            )

            image = roi.reshape(
                1, 48, 48, 1
            )

            prediction = model.predict(
                image,
                verbose=0
            )[0]

            label = int(np.argmax(prediction))

            emotion = EMOTIONS[label]

            confidence = (
                float(prediction[label]) * 100
            )

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            text = (
                f"{emotion}: "
                f"{confidence:.1f}%"
            )

            cv2.putText(
                frame,
                text,
                (x, max(y-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "FER2013 Emotion Recognition",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    video.release()
    cv2.destroyAllWindows()
    print("Application closed.")
