import cv2
import numpy as np

def detect_age_and_gender():
    # Paths to the pre-trained models and prototxt files
    age_model = "age_net.caffemodel"
    age_proto = "age_deploy.prototxt"
    gender_model = "gender_net.caffemodel"
    gender_proto = "gender_deploy.prototxt"

    # Load pre-trained models
    age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
    gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

    # Define age and gender categories
    age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    gender_list = ['Male', 'Female']

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to quit.")

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        # Prepare the frame for prediction
        blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Detect faces using OpenCV's Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Crop the face
            face = frame[y:y+h, x:x+w]
            
            # Pass the cropped face to the models
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            
            # Predict gender
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            # Predict age
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = age_ranges[age_preds[0].argmax()]

            # Display the results
            label = f"{gender}, {age}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the frame with detections
        cv2.imshow("Age and Gender Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the function
detect_age_and_gender()
