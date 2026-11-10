
import cv2
import time

for camera_index in range(4):
    print(f"\nTesting camera {camera_index}...")

    cap = cv2.VideoCapture(
        camera_index,
        cv2.CAP_AVFOUNDATION
    )

    if not cap.isOpened():
        print("Camera could not open")
        cap.release()
        continue

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Give the camera time to initialize
    time.sleep(2)

    for attempt in range(10):
        ret, frame = cap.read()

        if ret and frame is not None:
            print(f"SUCCESS! Camera {camera_index} works.")
            print("Frame shape:", frame.shape)

            cv2.imshow("Camera Test", frame)
            cv2.waitKey(3000)
            break

        print(f"Frame attempt {attempt + 1} failed")
        time.sleep(0.3)

    cap.release()

cv2.destroyAllWindows()
