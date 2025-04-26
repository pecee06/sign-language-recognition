import mediapipe as mp
import string

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

CONFIDENCE_THRESHOLD = 0.9
VICINITY_SIZE = 2  # how many letters on the left we check