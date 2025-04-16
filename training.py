import cv2
import numpy as np
import os

data_path = 'C:/Users/DELL/OneDrive/Desktop/dataset'
Training_Data, labels = [], []
label_dict = {}  # Dictionary to store label mappings
label_count = 0  # Unique label counter

# Iterate over each person's folder
for person_name in os.listdir(data_path):
    person_folder = os.path.join(data_path, person_name)

    if not os.path.isdir(person_folder):
        continue  # Skip if it's not a directory
    
    # Assign a unique label to each person
    label_dict[person_name] = label_count
    label_count += 1

    # Iterate over images in the person's folder
    for file in os.listdir(person_folder):
        image_path = os.path.join(person_folder, file)
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if images is not None:  # Ensure image is read correctly
            Training_Data.append(np.asarray(images, dtype=np.uint8))
            labels.append(label_dict[person_name])

labels = np.asarray(labels, dtype=np.int64)

# Create and train the model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(labels))

# Save the trained model
model.save("C:/Users/DELL/OneDrive/Desktop/trained_model.xml")

print('Dataset Model Training Completed')
