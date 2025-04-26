from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
from tensorflow import keras
import string
import pandas as pd
import numpy as np
import copy
from util import calc_landmark_list, pre_process_landmark

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify a list of allowed origins here
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

model = keras.models.load_model("model.h5")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet =  ['1','2','3','4','5','6','7','8','9']
alphabet += list(string.ascii_uppercase)

@app.get("/api/detect")
def read_sign_language():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            debug_image = copy.deepcopy(image)

            if results.multi_hand_landmarks:
                for hand_landmarks, _ in zip(results.multi_hand_landmarks,results.multi_handedness):
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    # Draw the landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    df = pd.DataFrame(pre_processed_landmark_list).transpose()

                    # predict the sign language
                    predictions = model.predict(df, verbose=0)
                    # get the predicted class for each sample
                    predicted_classes = np.argmax(predictions, axis=1)
                    label = alphabet[predicted_classes[0]]
                    cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                    print(alphabet[predicted_classes[0]])
                    print("------------------------")
            # output image
            cv2.imshow('Indian sign language detector', image)
            if cv2.waitKey(5) & 0xFF == 27:
                cv2.destroyAllWindows()
                break
    cap.release()
    return {"message": "Sign language detection stopped."}