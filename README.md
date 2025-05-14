# **Wildlife-conservation-and-monitoring**
LIBRARIES BEING USED: cv2: OpenCV, used for image and video processing.

numpy: Helps with handling numbers and arrays.

datetime: Used to get the current date and time.

os: Interacts with your file system (e.g., making folders).

time: For delays and time tracking. CLASS NAME SETUP: CLASSES: A list of 20 objects the AI model can recognize.

ANIMAL_CLASSES: A smaller set — just the animals we're interested in. PREPARING OUTPUT FOLDER AND LOG FILES: Creates a folder named captures to save images.

Makes sure there's a log file (animal_detections.txt) to write detections into. USING A WEBCAM: Opens your webcam (device 0 is the default one). TIMING SETUP: These help to avoid logging too many detections — only once every 2 seconds. START A LOOP: This loop runs continuously to keep detecting animals in real-time. READ A FRAME: Takes a picture (frame) from the webcam.

If it fails (ret is False), it quits. FILTER ANIMAL DETECTION: Only continue if confidence > 50%.

Check if the detected object is in our list of animals. SAVE THE LOG DETECTION: If it's been more than 2 seconds since the last detection:

Gets the current time.

Saves the frame as an image file (e.g., dog_2025-05-10_12-00-01.jpg).

Writes a message to animal_detections.txt.

Prints a message to the screen.

Updates the last_log_time. SHOW THE WEBCAM WINDOW: Displays the webcam feed with detection boxes.

Pressing q will stop the program.

