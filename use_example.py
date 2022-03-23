import cv2
import hand_tracking_module as htm
import time

cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

# Framerate Variables
previous_time = 0
current_time = 0

while cap.isOpened():
    success, image = cap.read()
    image = detector.find_hands(image, draw=True)
    landmark_list = detector.find_position(image)

    # Display data
    if len(landmark_list) != 0:
        print(landmark_list[0])

    # Display framerate
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(image, f'FPS: {int(fps)}', (5, 15),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # Display to screen
    cv2.imshow("Hand Tracking", image)
    cv2.waitKey(1)