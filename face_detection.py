import cv2


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ============================================================
# OPEN WEBCAM
# ============================================================

print("Starting camera...")

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("Error: Could not open camera.")

    exit()


print("Camera started successfully!")
print("Press 'q' to quit.")


# ============================================================
# READ CAMERA FRAMES
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print("Error: Could not read camera frame.")

        break


    # ========================================================
    # CONVERT FRAME TO GRAYSCALE
    # ========================================================

    gray_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # DETECT FACES
    # ========================================================

    faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5
    )


    # ========================================================
    # DRAW RECTANGLE AROUND EACH FACE
    # ========================================================

    for (x, y, width, height) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )


    # ========================================================
    # DISPLAY CAMERA
    # ========================================================

    cv2.imshow(
        "Face Detection",
        frame
    )


    # ========================================================
    # CHECK FOR 'q'
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# RELEASE CAMERA
# ============================================================

camera.release()


# ============================================================
# CLOSE WINDOWS
# ============================================================

cv2.destroyAllWindows()


print("Camera closed.")