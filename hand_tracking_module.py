import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.static_image_mode,
                                         self.max_num_hands,
                                         self.model_complexity,
                                         self.min_detection_confidence,
                                         self.min_tracking_confidence)

    def find_hands(self, image, draw=True):
        self.imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        self.results = self.hands.process(self.imageRGB)

        # Draw Landmarks
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return image

    def find_position(self, image, hand_id=0, draw=True):
        landmark_list = []

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_id]

            for landmark_id, landmark in enumerate(hand.landmark):
                image_height, image_width, image_channel = image.shape
                x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                # print(f"Landmark ID: {landmark_id} \nX: {x} \nY: {y} \n")
                landmark_list.append([landmark_id, x, y])

        return landmark_list

