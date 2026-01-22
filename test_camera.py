# test_camera.py
import cv2
import numpy as np

print("Testing camera access...\n")

# Try to open default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access camera 0")
    print("\nTroubleshooting:")
    print("1. Make sure no other app is using the camera")
    print("2. Try camera index 1: cv2.VideoCapture(1)")
    print("3. Check Windows camera privacy settings")
else:
    print("✓ Camera opened successfully!")
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"\nCamera info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    
    print("\nCapturing test frame...")
    ret, frame = cap.read()
    
    if ret:
        print("✓ Frame captured successfully!")
        print(f"  Frame shape: {frame.shape}")
        
        # Show the frame for 3 seconds
        print("\nDisplaying frame for 3 seconds...")
        print("(A window should pop up showing your camera)")
        cv2.imshow("Camera Test", frame)
        cv2.waitKey(3000)  # Wait 3 seconds
        cv2.destroyAllWindows()
        
        print("\n✓ Camera test complete!")
    else:
        print("❌ Failed to capture frame")
    
    cap.release()

print("\n" + "="*50)
print("Camera test finished")