import cv2


# ============================================================
# OPEN WEBCAM
# ============================================================

print("Starting camera...")

camera = cv2.VideoCapture(0)


# ============================================================
# CHECK CAMERA
# ============================================================

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
    # DISPLAY FRAME
    # ========================================================

    cv2.imshow(
        "Camera Test",
        frame
    )


    # ========================================================
    # CHECK FOR 'q' KEY
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