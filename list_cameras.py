import cv2

print("Scanning for cameras...")
print()

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ Camera {i}: Available ({frame.shape[1]}x{frame.shape[0]})")
        else:
            print(f"? Camera {i}: Opened but can't read")
        cap.release()
    else:
        # Don't print unavailable cameras
        pass

print()
print("Run the workshop with a specific camera:")
print("  Change line 54 in workshop_session.py from:")
print("    cap = cv2.VideoCapture(0)")
print("  To:")
print("    cap = cv2.VideoCapture(X)  # where X is your camera number")
