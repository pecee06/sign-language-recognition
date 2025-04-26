import copy
import itertools
from google import genai
import os
from dotenv import load_dotenv

# functions
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def read_words_from_file(file_path):
    with open(file_path, "r") as file:
        # Read the entire file content and split by commas
        words = file.read().strip().split(",")
    return [word.strip() for word in words]  # Strip any surrounding whitespace from each word

def find_word_in_gibberish(gibberish: str, words: list[str]) -> str | None:
    gibberish = gibberish.strip()

    def lcs_length(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m):
            for j in range(n):
                if s1[i].lower() == s2[j].lower():
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

        return dp[m][n]

    for word in words:
        match_letters = lcs_length(gibberish, word)
        match_percentage = (match_letters / len(word)) * 100

        if match_percentage >= 60:
            return word

    return None



def generate_content_with_file(file_path, gibberish):
    print(gibberish)
    # Read the words from the file
    words = read_words_from_file(file_path)
    word = find_word_in_gibberish(gibberish, words)
    if word is not None:
        return word.capitalize()
    # Create a prompt using the words from the file
    prompt = f"""
    Given a gibberish string {gibberish}, find the most probable English word hidden inside it.
    Instructions:
    - Give just the word, no explanation, no method, no extra text.
    - Explicit content not allowed.
    - Search through the entire English vocabulary you know — no external word list will be provided.
    - A word is considered hidden if its letters appear in the same order (left to right) in the gibberish string.
    - Random extra letters may appear between the correct ones (gaps are allowed), but no skipping back is allowed.
    - Each character from the gibberish can be used only once in order.
    - Calculate the match percentage: (number of matched letters / total letters in the word) * 100.
    - word is considered found if at least 60% of its letters match in order.
    - Prefer longer and more complete words when multiple matches are found.
    """

    # Use client.models.generate_content to generate content
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",  # Or use the relevant model name
        contents=prompt
    )

    return response.candidates[0].content.parts[0].text.capitalize() # Return the generated content as a string