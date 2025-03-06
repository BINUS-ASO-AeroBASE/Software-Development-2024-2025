import cv2
import time

global ptime

def process_frame(frame, last_position):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray,(5,7),0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 30, 90)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea, default=None)
    
    if largest_contour is not None and cv2.contourArea(largest_contour) > 1000:  # Filter small artifacts
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) == 4:  # Ensure the detected shape has four corners (like a frame)
            last_position = approx
        
        if (last_position is not None):
            for point in last_position:
                x, y = point.ravel()
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Get the center of the contour
        M = cv2.moments(approx)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Calculate the error (offset) from the center of the frame
            offsetx = 320 - cx
            offsety = 240 - cy

            # Draw the center of the contour and the line to the center of the frame
            cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 5, 3)
            cv2.line(frame, (320, 240), (cx, cy), color=(255, 255, 255))
        
        # Show camera center (320, 240)
        cv2.drawMarker(frame, (320, 240), (255, 255, 255), cv2.MARKER_CROSS, 5, 3)
    
    #show fps
    
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(frame, f"FPS : {int(fps)}", (0,20) , cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)  
    
    return frame, last_position , edges

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
last_position = None

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally (mirror)
    if not ret:
        break
    
    frame, last_position, before = process_frame(frame, last_position)
    
    # Show the output
    cv2.imshow('Frame Detection', frame)
    cv2.imshow('Grayscaled', before)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
