def read_words_from_file(file_path):
    with open(file_path, "r") as file:
        # Read the entire file content and split by commas
        words = file.read().strip().split(",")
    return [word.strip() for word in words]  # Strip any surrounding whitespace from each word