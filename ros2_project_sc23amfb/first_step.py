# Exercise 1 - Display an image of the camera feed to the screen
# and filter out all colours apart from green

#from __future__ import division
import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal


class colourIdentifier(Node):
    def __init__(self):
        super().__init__('cI')
        
        # Remember to initialise a CvBridge() and set up a subscriber to the image topic you wish to use
        # We covered which topic to subscribe to should you wish to receive image data
        self.bridge = CvBridge()
        self.sensitivity = 10

        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.callback,
            10
        )

        self.subscription  # prevent unused variable warning
        
    def callback(self, data):
        # Convert the received image into a opencv image
        # But remember that you should always wrap a call to this conversion method in an exception handler
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)
            return

        # Convert the rgb image into a hsv image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Green bounds
        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

        # Create green mask
        green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)

        # Apply mask to original image
        filtered_img = cv2.bitwise_and(image, image, mask=green_mask)

        # Show the resultant images
        cv2.namedWindow('camera_Feed', cv2.WINDOW_NORMAL)
        cv2.imshow('camera_Feed', filtered_img)
        cv2.resizeWindow('camera_Feed', 320, 240)
        cv2.waitKey(3)
        

# Create a node of your class in the main and ensure it stays up and running
# handling exceptions and such
def main():

    def signal_handler(sig, frame):
        if rclpy.ok():
            rclpy.shutdown()

    # Instantiate your class
    # And rclpy.init the entire node
    rclpy.init(args=None)
    cI = colourIdentifier()

    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(cI,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            continue
    except ROSInterruptException:
        pass

    # Remember to destroy all image windows before closing node
    cv2.destroyAllWindows()
    

# Check if the node is executing in the main path
if __name__ == '__main__':
    main()