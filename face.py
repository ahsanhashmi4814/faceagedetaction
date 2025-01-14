import cv2

def detect_faces_from_camera(cascade_path):
    # Load the Haar Cascade file
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("Error: Unable to load Haar Cascade XML file.")
        return

    # Open the laptop camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow("Face Detection", frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

# Path to the Haar Cascade XML file
cascade_path = 'C:\\laragon\\www\\python\\haarcascade_frontalface_default.xml'  # Replace with the actual path to the XML file

# Run face detection from the camera
detect_faces_from_camera(cascade_path)
