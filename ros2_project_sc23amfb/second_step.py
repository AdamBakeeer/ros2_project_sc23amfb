# Exercise 2 - detecting two colours, and filtering out the third colour and background.

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

        # Initialise the value you wish to use for sensitivity in the colour detection (10 should be enough)
        self.sensitivity = 10
        self.bridge = CvBridge()

        # Subscribe to the camera topic
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def draw_largest_contour(self, mask, image, colour_name, colour_bgr):
        contours, _ = cv2.findContours(
            mask,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > 0:
                (x, y), radius = cv2.minEnclosingCircle(c)
                center = (int(x), int(y))
                radius = int(radius)

                rx, ry, rw, rh = cv2.boundingRect(c)

                cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), colour_bgr, 2)
                cv2.circle(image, center, radius, colour_bgr, 2)

                if area < 500:
                    distance_text = "far"
                elif area < 2000:
                    distance_text = "medium"
                else:
                    distance_text = "close"

                cv2.putText(
                    image,
                    f"{colour_name}: {int(area)} px ({distance_text})",
                    (rx, max(20, ry - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colour_bgr,
                    2
                )

    def callback(self, data):
        # Convert the received image into an opencv image
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)
            return

        # Convert the rgb image into a hsv image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Set the upper and lower bounds for the colours you wish to identify

        # Green
        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

        # Blue
        hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
        hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])

        # Red needs two ranges
        hsv_red_lower1 = np.array([0, 100, 100])
        hsv_red_upper1 = np.array([self.sensitivity, 255, 255])

        hsv_red_lower2 = np.array([180 - self.sensitivity, 100, 100])
        hsv_red_upper2 = np.array([180, 255, 255])

        # Filter out everything but particular colours using cv2.inRange()
        green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
        blue_mask = cv2.inRange(hsv_image, hsv_blue_lower, hsv_blue_upper)
        red_mask1 = cv2.inRange(hsv_image, hsv_red_lower1, hsv_red_upper1)
        red_mask2 = cv2.inRange(hsv_image, hsv_red_lower2, hsv_red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Combine the masks
        rg_mask = cv2.bitwise_or(red_mask, green_mask)
        rgb_mask = cv2.bitwise_or(rg_mask, blue_mask)

        # Apply the mask to the original image
        filtered_img = cv2.bitwise_and(image, image, mask=rgb_mask)

        # Draw contours / circles / rectangles on a copy of the original image
        output_image = image.copy()

        self.draw_largest_contour(red_mask, output_image, 'Red', (0, 0, 255))
        self.draw_largest_contour(green_mask, output_image, 'Green', (0, 255, 0))
        self.draw_largest_contour(blue_mask, output_image, 'Blue', (255, 0, 0))

        # Show the resultant images you have created
        cv2.namedWindow('camera_Feed', cv2.WINDOW_NORMAL)
        cv2.imshow('camera_Feed', output_image)
        cv2.resizeWindow('camera_Feed', 320, 240)

        cv2.namedWindow('filtered_Feed', cv2.WINDOW_NORMAL)
        cv2.imshow('filtered_Feed', filtered_img)
        cv2.resizeWindow('filtered_Feed', 320, 240)

        cv2.waitKey(3)


# Create a node of your class in the main and ensure it stays up and running
def main():
    def signal_handler(sig, frame):
        if rclpy.ok():
            rclpy.shutdown()

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

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()