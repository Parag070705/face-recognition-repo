import cv2
import numpy as np
import os

# Define paths
dataset_path = 'C:/Users/DELL/OneDrive/Desktop/dataset'
model_path = 'C:/Users/DELL/OneDrive/Desktop/trained_model.xml'
cascade_path = 'C:/Users/DELL/OneDrive/Desktop/haarcascade_frontalface_default.xml'

# Initialize face classifier
face_classifier = cv2.CascadeClassifier(cascade_path)

# Prepare training data
Training_Data, labels = [], []
label_dict = {}  # Dictionary to store label mappings
label_count = 0  # Unique label counter

# Iterate over subdirectories (each person's dataset)
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_folder):
        continue  # Skip non-directory files
    
    # Assign a label to each person
    label_dict[label_count] = person_name
    label_count += 1

    # Load images from the person's folder
    for file in os.listdir(person_folder):
        image_path = os.path.join(person_folder, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            Training_Data.append(np.asarray(image, dtype=np.uint8))
            labels.append(label_count - 1)  # Assign the corresponding label

labels = np.asarray(labels, dtype=np.int64)

# Train the model if dataset is available
if len(Training_Data) > 0:
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(labels))
    model.save(model_path)
    print('Dataset Model Training Completed')
else:
    print("No valid training data found!")
    exit()

# Load trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read(model_path)

# Face detection function
def face_detector(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return img, None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
        return img, roi

    return img, None

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    image, face = face_detector(frame)

    if face is not None:
        try:
            result = model.predict(face)
            confidence = int(100 * (1 - (result[1]) / 300))

            if confidence > 82:
                person_name = label_dict[result[0]]
                color = (255, 255, 255)  # Green for recognized
            else:
                person_name = "Unknown"
                color = (0, 0, 255)  # Red for unknown

            cv2.putText(image, person_name, (250, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 2)

        except Exception as e:
            print(f"Prediction Error: {e}")
            cv2.putText(image, "Error in Prediction", (250, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Face Detector', image)

    if cv2.waitKey(1) == 13:  # Press 'Enter' to exit
        break

cap.release()
cv2.destroyAllWindows()
