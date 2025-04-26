from google import genai
import os
from utils.readWordsFromFile import read_words_from_file
from utils.findWordInGibberish import find_word_in_gibberish

def predict_word(file_path, gibberish):
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
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",  # Or use the relevant model name
        contents=prompt
    )

    return response.candidates[0].content.parts[0].text.capitalize() # Return the generated content as a string