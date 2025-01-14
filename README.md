Face Detection from Laptop Camera using OpenCV

This Python script uses OpenCV to perform real-time face detection from your laptop's camera using the Haar Cascade classifier.
Prerequisites

    Python: Ensure you have Python 3.x installed on your system.
    OpenCV: Install OpenCV by running the following command:

    pip install opencv-python

    Haar Cascade XML File: Download the Haar Cascade file (haarcascade_frontalface_default.xml) from OpenCV GitHub repository and save it on your system. Update the cascade_path variable in the script to point to the downloaded XML file.

How to Use

    Clone or copy this script to your local machine.

    Update the cascade_path variable with the full path to the haarcascade_frontalface_default.xml file. For example:

cascade_path = 'C:\\laragon\\www\\python\\haarcascade_frontalface_default.xml'

Run the script:

    python script_name.py

    Replace script_name.py with the name of the Python script.

    Once the script runs:
        A camera window will open, and it will detect faces in real-time.
        Detected faces will be highlighted with blue rectangles.
        Press the 'q' key to exit the application.

Notes

    Ensure that your camera is properly connected and accessible.
    The haarcascade_frontalface_default.xml file is required for face detection. If the file is missing or the path is incorrect, the script will not run.
    The detection parameters, such as scaleFactor, minNeighbors, and minSize, can be adjusted to improve detection performance based on your requirements.

Example Output

When the script runs successfully, you'll see a window displaying a live video feed from your camera with rectangles drawn around detected faces.