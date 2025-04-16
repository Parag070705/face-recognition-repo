import cv2  #open CV for image processing 
import numpy as np  #handel nummerical operation
import os  #fileand directory management 

face_classifier = cv2.CascadeClassifier('C:/Users/DELL/OneDrive/Desktop/haarcascade_frontalface_default.xml')  #load pre-trained cascade classifier and path

def face_extractor(img):  #define face extraction function
    #haar cascades require grayscale input
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert BGR img into gray img (in python RGB writes as BGR)
    faces = face_classifier.detectMultiScale(gray, 1.1, 5) #detectface as per scalefactor and minNeighbors define 
    
    if len(faces) == 0:
        return None #no face detected
    
    x, y, w, h = faces[0]  # Select the first detected face and coordinates
    cropped_face = img[y:y+h, x:x+w] #height and width of image stored cropped_face
    return cropped_face


# Create dataset folder if not exists
dataset_path = 'C:/Users/DELL/OneDrive/Desktop/dataset'
os.makedirs(dataset_path, exist_ok=True)  #creates a folder for storing data and avoids error if the folder already exists

# Prompt for user ID or name
user_name = input("Enter the person's name or ID: ") #ask for user's name or id 
user_folder = os.path.join(dataset_path, user_name) #creates a separate folder for each person
os.makedirs(user_folder, exist_ok=True) #check if folder is already exist

cap = cv2.VideoCapture(0) #creates a video capture object  an 0 represents the default webcam
count =0 #initializs a counter

while True: #infinite loop
    ret, frame = cap.read() #captures a frame from webcam
    if not ret: #if false
        print("Failed to capture image")
        break #exits the loop

    extracted_face = face_extractor(frame) #detect and crops the face from the frame and stored in extracted_face 
    
    if extracted_face is not None: #returns None if no face is found
        count += 1 #increments the count variable by 1
        face = cv2.resize(extracted_face, (200,200)) #resizes the extracted face to 200×200 pixels using OpenCV’s cv2.resize() function
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) 

        file_name_path = os.path.join(user_folder, f"{count}.jpg") # creates a full file path for saving the extracted face image
        cv2.imwrite(file_name_path, face)

        cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2) #adds a count number to the face image
        cv2.imshow('Face Cropper', face) #display the extracted face image in a new window
    else:
        print("Face not found")

    if cv2.waitKey(1) == 13 or count == 50:  # Press Enter (13) to exit
        break

cap.release() #release the capture video function
cv2.destroyAllWindows() #for closing windows
print(f'Sample Collection Completed for {user_name}')



#TRAINING AND RECOGNITION CODE 

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

# iterate over subdirectories (each person's dataset)
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_folder):
        continue  # Skip non-directory files
    
    # issign a label to each person
    label_dict[label_count] = person_name
    label_count += 1 #new label for new person

    # load images from the person's folder
    for file in os.listdir(person_folder):
        image_path = os.path.join(person_folder, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is not None:
            Training_Data.append(np.asarray(image, dtype=np.uint8))
            labels.append(label_count - 1)  # Assign the corresponding label

labels = np.asarray(labels, dtype=np.int64)  #converts list into NumPy array


if len(Training_Data) > 0: #Ensures that there is at least one image in Training_Data before training
    model = cv2.face.LBPHFaceRecognizer_create() #LBPH (Local Binary Patterns Histograms) is effective for face recognition in different lighting conditions
    model.train(np.asarray(Training_Data), np.asarray(labels)) #converts into Numpy arrays
    model.save(model_path) #save as model_path
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
    faces = face_classifier.detectMultiScale(gray, 1.1, 5)

    return gray, faces

# Start video capture
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()  #capturing video through webcam
    if not ret: #if not
        print("Failed to capture image")
        break

    gray, faces = face_detector(frame)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w] #extracts the face region (roi=Region of Interest) from the grayscale image
            roi = cv2.resize(roi, (200, 200))
            
            try:
                result = model.predict(roi) #result predict by model
                confidence = int(100 * (1 - (result[1]) / 300)) #calculation of confidence witch define by model
                
                if confidence > 82: #when confidence greater than 82%
                    person_name = label_dict[result[0]] #user name or id
                    color = (0, 255, 0)  # Green for recognized
                else:
                    person_name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) #rectangle for face
                cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2) #displayed name of the user
            
            except Exception as e:
                print(f"Prediction Error: {e}")
                cv2.putText(frame, "Error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) #if face is not visible clearly

    cv2.imshow('Face Recognition', frame) #frame name

    if cv2.waitKey(1) == 13:  # Press 'Enter' to exit
        break

cap.release()
cv2.destroyAllWindows()

