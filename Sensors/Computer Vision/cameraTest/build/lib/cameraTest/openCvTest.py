import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import time

# Camera setup
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not access the camera.")
else:
    print("Camera opened successfully!")

cam.set(3, 640)  # Set width
cam.set(4, 480)  # Set height

# Define blue color bounds in HSV
blue_lower = np.array([50, 80, 80])
blue_upper = np.array([120, 200, 150])

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'offset', 10)
        self.publishercamera_ = self.create_publisher(Image, 'image', 10)
        self.timer = self.create_timer(0.05, self.talker_callback)  # Timer to call the callback every 0.1 seconds
        self.i = [0, 0]
        self.bridge = CvBridge()

    def talker_callback(self):
        offsetx, offsety = 0, 0

        SUCCESS, frame = cam.read()
        if not SUCCESS:
            self.get_logger().error("Failed to capture frame from camera.")
            return
        
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally (mirror)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert to HSV color space

        # Create a mask for blue color
        mask = cv2.inRange(frame_hsv, blue_lower, blue_upper)
        
        # Get contours of the blue shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)  # Get largest contour
            approx_contour = cv2.approxPolyDP(largest_contour, 0.02 * cv2.arcLength(largest_contour, True), True)
            cv2.drawContours(frame, [approx_contour], -1, (255, 255, 255), 3)
        
            # Get the center of the contour
            M = cv2.moments(approx_contour)
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
        cv2.imshow("Result", frame)

        # Publish offset message
        msg = String()
        msg.data = f'Offset : ({offsetx:3d}, {offsety:3d})'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')

        # Convert the OpenCV image to a ROS image message and publish it
        ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.publishercamera_.publish(ros_image)  

def main():
    rclpy.init()

    minimal_publisher = MinimalPublisher()

    # Spin the ROS node
    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()

    # Release the camera and close OpenCV windows
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
