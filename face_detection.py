import cv2  # Import OpenCV library

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam (use 0 for the default camera)
video_capture = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not video_capture.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()

    # If the frame was not captured properly, exit the loop
    if not ret:
        print("Error: Failed to capture a frame.")
        break

    # Convert the frame to grayscale (required by the face detection algorithm)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces in the original frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame with the faces marked with rectangles
    cv2.imshow('Face Detection', frame)
     
    # Save the Detected Image
    cv2.imwrite('detected_faces.jpg', frame) 

    # Press 'q' to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
video_capture.release()
cv2.destroyAllWindows()