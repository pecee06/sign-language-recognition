# Sign Language Recognition

This project implements an automated system to recognize sign language gestures and translate them into text. By leveraging machine learning and computer vision, it aims to bridge communication gaps for deaf or hearing-impaired individuals. The system captures live video of hand movements, extracts landmark keypoints, and uses a trained neural network to identify the corresponding sign. Ultimately, this enables real-time sign-to-text translation to facilitate more inclusive communication.

## Overview

The repository contains a full-stack sign language recognition solution, including a frontend interface and a backend processing pipeline. The **backend** (in `backend/src`) is built with Python and FastAPI, and it uses OpenCV and Mediapipe to capture video frames and extract hand landmarks. Each frame is processed to detect hands and compute landmark keypoints, which are then fed into the trained model for classification ([sign-language-recognition/backend/src/main.py at main · pecee06/sign-language-recognition · GitHub](https://github.com/pecee06/sign-language-recognition/blob/main/backend/src/main.py#:~:text=for%20hand_landmarks%2C%20_%20in%20zip%28results,multi_handedness)). The **frontend** (in `frontend/`) is a React application (using Vite and Tailwind CSS) that provides a user interface to display video and recognized signs. The FastAPI server exposes an endpoint (`/api/recognize`) that runs the inference loop, streaming predictions to the frontend as the user signs ([sign-language-recognition/backend/src/main.py at main · pecee06/sign-language-recognition · GitHub](https://github.com/pecee06/sign-language-recognition/blob/main/backend/src/main.py#:~:text=for%20hand_landmarks%2C%20_%20in%20zip%28results,multi_handedness)) ([sign-language-recognition/backend/src/main.py at main · pecee06/sign-language-recognition · GitHub](https://github.com/pecee06/sign-language-recognition/blob/main/backend/src/main.py#:~:text=%40app.get%28)).

## Model Architecture

The sign language classifier is a deep feedforward neural network implemented in TensorFlow/Keras. The architecture consists of four hidden dense layers followed by dropout, and a softmax output layer. Specifically, the network layers are as follows (with ReLU activations):

- Dense layer with 1470 units, followed by Dropout(0.5) ([github.com](https://github.com/pecee06/sign-language-recognition/raw/refs/heads/main/backend/src/ISL_classifier.ipynb#:~:text=,))
- Dense layer with 832 units, followed by Dropout(0.5) ([github.com](https://github.com/pecee06/sign-language-recognition/raw/refs/heads/main/backend/src/ISL_classifier.ipynb#:~:text=,))
- Dense layer with 428 units, followed by Dropout(0.5) ([github.com](https://github.com/pecee06/sign-language-recognition/raw/refs/heads/main/backend/src/ISL_classifier.ipynb#:~:text=,))
- Dense layer with 264 units, followed by Dropout(0.5) ([github.com](https://github.com/pecee06/sign-language-recognition/raw/refs/heads/main/backend/src/ISL_classifier.ipynb#:~:text=,))
- Output Dense layer with 35 units (softmax) for classification ([github.com](https://github.com/pecee06/sign-language-recognition/raw/refs/heads/main/backend/src/ISL_classifier.ipynb#:~:text=,))

The model uses sparse categorical cross-entropy loss and the Adam optimizer. Dropout layers are used between hidden layers to reduce overfitting. This architecture was chosen to handle the dimensionality of the input (hand landmark vectors) and to classify among multiple sign classes.

## Dataset Details

The dataset consists of preprocessed hand landmark keypoints collected for various sign gestures. Each data sample is stored as a row in `backend/src/keypoint.csv`, where the first column is the class label (the sign) and the remaining columns are the flattened (x, y) coordinates of hand landmarks. For example, the code loads this CSV with pandas:

```python
data = pd.read_csv('keypoint.csv', header=None)  # load keypoint data ([github.com](https://github.com/pecee06/sign-language-recognition/raw/refs/heads/main/backend/src/ISL_classifier.ipynb#:~:text=,base_uri))
data[0] = data[0].astype(str)
```

Each row thus represents one sign instance (e.g., the gesture for letter 'A', 'B', 'C', etc.) by its landmarks. The notebook found unique labels (like `['A', 'B', 'C']`) in the first column ([github.com](https://github.com/pecee06/sign-language-recognition/raw/refs/heads/main/backend/src/ISL_classifier.ipynb#:~:text=,base_uri)), indicating which sign each sample corresponds to. The remaining columns (typically 42 columns for 21 hand landmarks with x and y coordinates) form the feature vector for that sample.

## Training Process

The model is trained in the provided Jupyter notebook (`ISL_classifier.ipynb`). The data is first split into training and test sets using an 80/20 split:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) ([github.com](https://github.com/pecee06/sign-language-recognition/raw/refs/heads/main/backend/src/ISL_classifier.ipynb#:~:text=%7B%20,))
```

Early stopping is used to avoid overfitting (monitoring validation loss with patience=2) ([github.com](https://github.com/pecee06/sign-language-recognition/raw/refs/heads/main/backend/src/ISL_classifier.ipynb#:~:text=,)). The model is trained for up to 50 epochs with a batch size of 128 and a 20% validation split:

```python
model.fit(X_train, y_train, epochs=50, batch_size=128, validation_split=0.2, callbacks=[es]) ([github.com](https://github.com/pecee06/sign-language-recognition/raw/refs/heads/main/backend/src/ISL_classifier.ipynb#:~:text=,accuracy))
```

During training, the model rapidly reaches very high accuracy (nearly 100% on both training and validation) ([github.com](https://github.com/pecee06/sign-language-recognition/raw/refs/heads/main/backend/src/ISL_classifier.ipynb#:~:text=,accuracy)). After training, the model weights are saved to `model.h5` for later inference.

## Results

After training, the model achieves perfect performance on the test set for the classes in the dataset. The reported metrics are:

- Accuracy: 1.0
- Precision: 1.0
- Recall: 1.0
- F1-score: 1.0

These results (shown in the training notebook output) indicate 100% accuracy and perfect precision/recall for the tested signs ([github.com](https://github.com/pecee06/sign-language-recognition/raw/refs/heads/main/backend/src/ISL_classifier.ipynb#:~:text=%7B%20,id)). In practice, such performance suggests either a simple dataset or near-separable classes; real-world usage should validate on diverse sign data to ensure robustness.

## Setup Instructions

To set up the environment:

- **Clone the repository**:

  ```bash
  git clone https://github.com/pecee06/sign-language-recognition.git
  cd sign-language-recognition
  ```

- **Backend (Python) dependencies**:  
  Ensure you have Python 3.8+ installed. Install required Python packages:

  ```bash
  pip install -r backend/requirements.txt
  ```

  This includes libraries like TensorFlow (e.g. TensorFlow 2.19.0 as listed) ([sign-language-recognition/backend/requirements.txt at main · pecee06/sign-language-recognition · GitHub](https://github.com/pecee06/sign-language-recognition/blob/main/backend/requirements.txt#:~:text=tensorflow%3D%3D2)), FastAPI, OpenCV, Mediapipe, etc.

- **Frontend (Node.js) dependencies**:  
  Ensure you have Node.js (version ~19) and npm. In the `frontend/` directory, install dependencies:

  ```bash
  cd frontend
  npm install
  ```

  The `package.json` specifies React, Vite, Tailwind CSS, and other front-end libraries ([sign-language-recognition/frontend/package.json at main · pecee06/sign-language-recognition · GitHub](https://github.com/pecee06/sign-language-recognition/blob/main/frontend/package.json#:~:text=)).

- **Environment variables**:  
  Create a `.env` file in `backend/src/` if needed (the code uses `dotenv`). For example, you might set configuration flags or API keys here. No specific keys are required by default.

- **Database or Models**:  
  If you have a pre-trained model file (`model.h5`), place it in the `backend/src/` directory. Otherwise, run the training notebook (`ISL_classifier.ipynb`) to generate `model.h5` from the dataset.

## Usage Examples

- **Training the model** (if starting from raw data): Open `backend/src/ISL_classifier.ipynb` in Jupyter or Google Colab and run all cells to train the network and save `model.h5`.

- **Starting the server**: From the `backend/src/` directory, launch the FastAPI server. For example:

  ```bash
  uvicorn main:app --reload
  ```

  This will start the server (by default on `http://localhost:8000`).

- **Recognition API**: The FastAPI backend provides an endpoint `/api/recognize`. When accessed (for example, via the frontend), it captures webcam frames, processes them, and returns the predicted sign. The relevant code is:

  ```python
  @app.get("/api/recognize")
  def read_sign_language():
      ...
      # process frames and return predicted sign
  ```

  ([sign-language-recognition/backend/src/main.py at main · pecee06/sign-language-recognition · GitHub](https://github.com/pecee06/sign-language-recognition/blob/main/backend/src/main.py#:~:text=%40app.get%28))  
  You can also call this endpoint directly (e.g., `curl http://localhost:8000/api/recognize`) to get the prediction result after signing in front of the camera.

- **Example command (frontend)**: Once both servers are running, open the browser to the frontend URL. The app will show the webcam feed and overlay the detected sign label in real time.

## Contribution Guidelines

Contributions are welcome to improve the Sign Language Recognition project. Please follow these guidelines:

- **Fork and branch**: Fork the repository and create a feature branch for any changes or enhancements.
- **Code style**: Maintain consistent coding style (e.g. PEP8 for Python, ESLint rules for JavaScript) as seen in the repository.
- **Issues and pull requests**: Report bugs or request features by opening an issue. For contributing code, submit a pull request with a clear description of changes.
- **Testing**: Ensure any new code is tested. For model changes, verify performance on a held-out set.
- **License**: This project is released under an open-source license (please check the LICENSE file). By contributing, you agree that your contributions will be licensed under the same terms.

We appreciate any feedback or contributions that help improve this system for the community. Happy coding!
