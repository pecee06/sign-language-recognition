import cv2
import pandas as pd
import numpy as np
import copy
from utils.recognitionReqs import mp_hands
from utils.preprocessLandmark import pre_process_landmark
from utils.calcLandmarkList import calc_landmark_list
from utils.predictWord import predict_word
from utils.recognitionReqs import mp_hands, mp_drawing, mp_drawing_styles
from fastapi.responses import JSONResponse

def initialize_camera():
    return cv2.VideoCapture(0)

def initialize_hands():
    return mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def preprocess_image(image):
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def postprocess_image(image):
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def predict_sign(model, pre_processed_landmark_list):
    df = pd.DataFrame(pre_processed_landmark_list).transpose()
    predictions = model.predict(df, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    return predicted_class, confidence

def display_label(image, label, is_confident):
    color = (0, 0, 255) if is_confident else (255, 0, 0)
    text = label if is_confident else "..."
    cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

def process_landmarks(image, hand_landmarks):
    landmark_list = calc_landmark_list(image, hand_landmarks)
    return pre_process_landmark(landmark_list)

def process_frame(image, hands):
    """Preprocess the frame, detect hands, and postprocess the frame."""
    image = preprocess_image(image)
    results = hands.process(image)
    image = postprocess_image(image)
    debug_image = copy.deepcopy(image)
    return image, debug_image, results

def process_hand(debug_image, image, hand_landmarks):
    """Process one hand's landmarks and draw them."""
    pre_processed_landmarks = process_landmarks(debug_image, hand_landmarks)

    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )

    return pre_processed_landmarks

def predict_label(model, pre_processed_landmarks, alphabet, CONFIDENCE_THRESHOLD):
    """Predict the label and check if confident enough."""
    predicted_class, confidence = predict_sign(model, pre_processed_landmarks)
    if confidence >= CONFIDENCE_THRESHOLD:
        return alphabet[predicted_class], True
    else:
        return "", False

def update_buffer(buffer, label, VICINITY_SIZE):
    """Update buffer if label is new in the last VICINITY_SIZE elements."""
    if label and (label not in buffer[-VICINITY_SIZE:]):
        buffer.append(label)

def finalize_prediction(buffer):
    """Finalize prediction by joining the buffer and finding nearest word."""
    word = ''.join(buffer)
    word = predict_word("ref/words.txt", word).rstrip("\n")
    return JSONResponse(content={"word": word})