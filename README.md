# Speech Emotion Recognition (SER) with MLP Classifier

## Project Overview
This project implements a **Speech Emotion Recognition (SER)** system using machine learning, specifically a **Multi-Layer Perceptron (MLP)** classifier, to identify emotions in speech from the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset. The system can recognize emotions such as calmness, happiness, fearfulness, and disgust with an accuracy of **86.32%**.

### Key Features:
- **Emotion Prediction:** Recognizes a variety of emotions from speech such as happiness, sadness, anger, fear, calmness, etc.
- **Real-Time Recognition:** Uses the **pyaudio** library for real-time emotion detection from live audio input.
- **File-Based Recognition:** Can predict emotions from pre-recorded audio files.
- **High Accuracy:** Achieves **86.32%** accuracy using the RAVDESS dataset with feature extraction techniques like **MFCC**, **Mel Spectrogram**, and **Chroma**.

## Technologies
- **Python** (for implementation)
- **MLP Classifier** (for emotion classification)
- **pyaudio** (for real-time audio capture)
- **Librosa** (for feature extraction like MFCC, Chroma, and Mel Spectrogram)
- **RAVDESS dataset** (emotional speech and song recordings)

## Dataset
- The **RAVDESS dataset** consists of **7356 audio files**, with emotional speech and song recordings from **24 professional actors** (12 males and 12 females).
- Emotions: Happy, Sad, Calm, Angry, Fearful, Surprise, Disgust, etc.
- The dataset contains both **speech** and **song** recordings.
  ![Image](https://github.com/user-attachments/assets/ea51a248-07c9-44ff-9c32-1e217303b733)

## Installation
1. Clone the repository:
   ```bash
   git clone https://[https://github.com/Nafisa-Rumman-Sauda/Speech-Emotion-Recognition-SER-]
   cd speech-emotion-recognition
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. **Create and Train Model**  
   Run the training script to create and train the model using the RAVDESS dataset.

   ```bash
   python train_model.py
   ```

### 2. **Record and Predict Emotion (Real-Time)**  
   Use the system to record live audio and predict the emotion in real-time.

   ```bash
   python live_recognition.py
   ```

### 3. **Predict Emotion on Pre-Recorded Audio**  
   To predict emotions from a pre-recorded audio file:
   
   ```bash
   python predict_from_file.py --audio_path <path_to_audio_file>
   ```

## System Architecture

1. **Speech Processing Module**  
   - **Pre-Processing**: Removes silence, applies pre-emphasis, and normalizes audio.
   - **Feature Extraction**: Extracts features such as **MFCC**, **Mel Spectrogram**, and **Chroma**.

2. **Classification Module**  
   - **MLP Classifier**: The extracted features are fed into an MLP classifier to predict the emotional state.

## Results
- **Accuracy**: The system achieves **86.32%** accuracy for emotion recognition from speech.
- **Real-Time Recognition**: The model supports real-time speech emotion recognition using a microphone.
- 
![Image](https://github.com/user-attachments/assets/b12b643b-a527-4497-8f1a-175426552b6c)

- **Pre-Recorded Audio**: Emotion prediction can also be done from pre-recorded audio files.

 ![Image](https://github.com/user-attachments/assets/26e8ea02-1627-45a4-8a53-eee8360b8a12)

![Image](https://github.com/user-attachments/assets/85938577-1e5d-47ba-8849-b2bc59d43ce4)


| **Dataset/Technique** | **Accuracy** |
|-----------------------|--------------|
| IEMOCAP               | 73%          |
| CNN-SVM               | 70%          |
| GAN                   | 46.52%       |
| SVM                   | 60.8%        |
| RAVDESS with MLP      | **86.32%**   |

![Image](https://github.com/user-attachments/assets/061e2d6d-ca3c-4e69-849d-89dff069c967)



## Future Work
- Expand the dataset to include more emotions and languages.
- Improve performance by eliminating background noise and interference.
- Enhance accuracy by using more diverse training data.
- Implement emotion detection with context awareness at a minute level.

## Conclusion
This system provides an accurate, real-time solution for **Speech Emotion Recognition (SER)**. It can be applied to improve human-computer interaction, virtual voice assistants, customer service, and marketing strategies.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
