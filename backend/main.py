from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import cv2
import mediapipe as mp
from tensorflow import keras
import string
import pandas as pd
import numpy as np
import copy

from util import calc_landmark_list, pre_process_landmark, generate_content_with_file

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = keras.models.load_model("model.h5")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

CONFIDENCE_THRESHOLD = 0.9
VICINITY_SIZE = 2  # how many letters on the left we check

@app.get("/api/recognize")
def read_sign_language():
    cap = cv2.VideoCapture(0)

    buffer = []

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image = cv2.flip(image, 1)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            debug_image = copy.deepcopy(image)

            if results.multi_hand_landmarks:
                for hand_landmarks, _ in zip(results.multi_hand_landmarks, results.multi_handedness):
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)

                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    df = pd.DataFrame(pre_processed_landmark_list).transpose()

                    # Predict
                    predictions = model.predict(df, verbose=0)[0]
                    predicted_class = np.argmax(predictions)
                    confidence = predictions[predicted_class]

                    if confidence >= CONFIDENCE_THRESHOLD:
                        label = alphabet[predicted_class]

                        # Check last VICINITY_SIZE letters
                        if label not in buffer[-VICINITY_SIZE:]:
                            buffer.append(label)

                        # Display
                        cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                    else:
                        cv2.putText(image, "...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

            cv2.imshow('Indian Sign Language Detector', image)

            if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                break

    cap.release()
    cv2.destroyAllWindows()

    word = ''.join(buffer)
    word = generate_content_with_file("ref/words.txt", word).rstrip("\n")
    return JSONResponse(content={"word": word})
