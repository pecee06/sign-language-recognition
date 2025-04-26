from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
from dotenv import load_dotenv
load_dotenv()
from utils.recognitionReqs import alphabet, CONFIDENCE_THRESHOLD, VICINITY_SIZE
from utils.imageUtility import (
    initialize_camera,
    initialize_hands,
    process_frame,
    process_hand,
    predict_label,
    update_buffer,
    finalize_prediction,
    display_label
)
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files directory
app.mount("/assets", StaticFiles(directory="../dist/assets"), name="assets")
# Serving the static files

@app.get("/api/recognize")
def read_sign_language():
    cap = initialize_camera()
    buffer = []

    with initialize_hands() as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image, debug_image, results = process_frame(image, hands)

            if results.multi_hand_landmarks:
                for hand_landmarks, _ in zip(results.multi_hand_landmarks, results.multi_handedness):
                    pre_processed_landmarks = process_hand(debug_image, image, hand_landmarks)
                    label, is_confident = predict_label(model, pre_processed_landmarks, alphabet, CONFIDENCE_THRESHOLD)

                    update_buffer(buffer, label, VICINITY_SIZE)
                    display_label(image, label, is_confident)

            cv2.imshow('Indian Sign Language Detector', image)

            if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                break

    cap.release()
    cv2.destroyAllWindows()

    return finalize_prediction(buffer)

@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    return FileResponse("../dist/index.html")