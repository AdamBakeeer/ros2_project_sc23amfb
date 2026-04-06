# Exercise 4 - following a colour (green) and stopping upon sight of another (blue).

import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal


class Robot(Node):
    def __init__(self):
        super().__init__('robot')

        # Publisher to move robot
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.rate = self.create_rate(10)

        # Flags
        self.green_found = False
        self.blue_found = False
        self.stop_on_blue = False

        # Detection values
        self.green_area = 0
        self.green_cx = None
        self.image_width = 0

        # Sensitivity
        self.sensitivity = 10

        # Area thresholds
        self.min_area = 500
        self.close_area = 8000   # tune this if needed

        # Movement messages
        self.forward_msg = Twist()
        self.forward_msg.linear.x = 0.15

        self.left_msg = Twist()
        self.left_msg.angular.z = 0.3

        self.right_msg = Twist()
        self.right_msg.angular.z = -0.3

        self.stop_msg = Twist()

        # Camera
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.callback,
            10
        )
        self.subscription

    def callback(self, data):
        # Reset frame-based flags
        self.green_found = False
        self.blue_found = False
        self.green_area = 0
        self.green_cx = None

        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)
            return

        self.image_width = image.shape[1]

        # HSV conversion
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Green range
        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

        # Blue range
        hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
        hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])

        # Masks
        green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
        blue_mask = cv2.inRange(hsv_image, hsv_blue_lower, hsv_blue_upper)

        green_image = cv2.bitwise_and(image, image, mask=green_mask)
        blue_image = cv2.bitwise_and(image, image, mask=blue_mask)

        # Find contours
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # GREEN detection
        if len(green_contours) > 0:
            c = max(green_contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > self.min_area:
                self.green_found = True
                self.green_area = area

                M = cv2.moments(c)
                if M['m00'] != 0:
                    self.green_cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    self.green_cx = None
                    cy = 0

                (x, y), radius = cv2.minEnclosingCircle(c)
                center_x = int(x)
                center_y = int(y)
                radius = int(radius)

                cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"GREEN area: {int(area)}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # BLUE detection
        if len(blue_contours) > 0:
            c = max(blue_contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > self.min_area:
                self.blue_found = True
                self.stop_on_blue = True   # latch stop permanently once blue is seen

                (x, y), radius = cv2.minEnclosingCircle(c)
                center_x = int(x)
                center_y = int(y)
                radius = int(radius)

                cv2.circle(image, (center_x, center_y), radius, (255, 0, 0), 2)
                cv2.putText(
                    image,
                    f"BLUE DETECTED",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

        # Show windows
        cv2.namedWindow('camera_Feed', cv2.WINDOW_NORMAL)
        cv2.imshow('camera_Feed', image)
        cv2.resizeWindow('camera_Feed', 320, 240)

        cv2.namedWindow('green_Filtered', cv2.WINDOW_NORMAL)
        cv2.imshow('green_Filtered', green_image)
        cv2.resizeWindow('green_Filtered', 320, 240)

        cv2.namedWindow('blue_Filtered', cv2.WINDOW_NORMAL)
        cv2.imshow('blue_Filtered', blue_image)
        cv2.resizeWindow('blue_Filtered', 320, 240)

        cv2.waitKey(3)

    def walk_forward(self):
        self.publisher.publish(self.forward_msg)

    def turn_left(self):
        self.publisher.publish(self.left_msg)

    def turn_right(self):
        self.publisher.publish(self.right_msg)

    def stop(self):
        self.publisher.publish(self.stop_msg)


def main():
    def signal_handler(sig, frame):
        if rclpy.ok():
            robot.stop()
            rclpy.shutdown()

    rclpy.init(args=None)
    robot = Robot()

    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            # If blue was ever seen -> stop and stay stopped
            if robot.stop_on_blue:
                robot.stop()

            # Otherwise follow green
            elif robot.green_found:
                if robot.green_cx is not None:
                    image_center = robot.image_width // 2
                    tolerance = 40

                    if robot.green_cx < image_center - tolerance:
                        robot.turn_left()
                    elif robot.green_cx > image_center + tolerance:
                        robot.turn_right()
                    else:
                        # centered, now decide forward or stop
                        if robot.green_area < robot.close_area:
                            robot.walk_forward()
                        else:
                            robot.stop()
                else:
                    robot.stop()
            else:
                robot.stop()

    except ROSInterruptException:
        pass

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()