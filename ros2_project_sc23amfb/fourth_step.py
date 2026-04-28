import math
import time
from enum import Enum

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav2_msgs.action import NavigateToPose
from cv_bridge import CvBridge, CvBridgeError


class RobotState(Enum):
    GO_TO_POINT = 1
    SCAN = 2
    GO_BLUE = 3
    DONE = 4


class Robot(Node):
    def __init__(self):
        super().__init__('robot')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.bridge = CvBridge()

        self.state = RobotState.GO_TO_POINT

        self.goal_running = False
        self.goal_handle = None
        self.point_index = 0
        self.scan_finish_time = None

        self.seen_red = False
        self.seen_green = False
        self.seen_blue = False

        self.red_found = False
        self.green_found = False
        self.blue_found = False

        self.red_area = 0
        self.green_area = 0
        self.blue_area = 0

        self.red_x = None
        self.green_x = None
        self.blue_x = None

        self.img_width = 0

        self.sensitivity = 12
        self.min_area = 400

        self.scan_time = 6.0
        self.scan_speed = 0.35

        self.center_tol = 40
        self.forward_speed = 0.10
        self.turn_speed = 0.25

        self.blue_stop_area = 300000

        self.points = [
            (8.185800204767737, -12.389409526182579, 1.75),
            (7.335262837295241, -4.601748957176293, 3.04),
            (-11.165934345008875, 3.782590515528419, -0.58),
            (-9.425877960122047, -14.62815691153874, 0.66),
        ]

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('Robot node started')

    def yaw_to_quaternion(self, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return qz, qw

    def move_robot(self, lin=0.0, ang=0.0):
        msg = Twist()
        msg.linear.x = lin
        msg.angular.z = ang
        if rclpy.ok():
            self.cmd_pub.publish(msg)

    def stop_robot(self):
        self.move_robot(0.0, 0.0)

    def send_goal(self, x, y, yaw):
        if self.goal_running:
            return

        if not self.nav_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn('Nav2 action server not ready yet')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0

        qz, qw = self.yaw_to_quaternion(yaw)
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.get_logger().info(f'Going to point {self.point_index + 1}: x={x:.2f}, y={y:.2f}')

        self.goal_running = True
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response)

    def goal_response(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            self.goal_running = False
            return

        self.goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result)

    def goal_result(self, future):
        self.goal_running = False
        self.goal_handle = None

        self.get_logger().info('Goal reached, scanning')
        self.state = RobotState.SCAN
        self.scan_finish_time = time.monotonic() + self.scan_time

    def get_biggest_contour(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, 0, None, None

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area < self.min_area:
            return None, 0, None, None

        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx = None
            cy = None

        return c, area, cx, cy

    def image_callback(self, data):
        self.red_found = False
        self.green_found = False
        self.blue_found = False

        self.red_area = 0
        self.green_area = 0
        self.blue_area = 0

        self.red_x = None
        self.green_x = None
        self.blue_x = None

        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        self.img_width = image.shape[1]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(hsv, np.array([0, 120, 80]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 120, 80]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        green_mask = cv2.inRange(
            hsv,
            np.array([60 - self.sensitivity, 100, 80]),
            np.array([60 + self.sensitivity, 255, 255])
        )

        blue_mask = cv2.inRange(
            hsv,
            np.array([120 - self.sensitivity, 100, 80]),
            np.array([120 + self.sensitivity, 255, 255])
        )

        kernel = np.ones((5, 5), np.uint8)

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        red_contour, self.red_area, self.red_x, _ = self.get_biggest_contour(red_mask)
        if red_contour is not None:
            self.red_found = True
            self.seen_red = True
            (x, y), radius = cv2.minEnclosingCircle(red_contour)
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 0, 255), 2)

        green_contour, self.green_area, self.green_x, _ = self.get_biggest_contour(green_mask)
        if green_contour is not None:
            self.green_found = True
            self.seen_green = True
            (x, y), radius = cv2.minEnclosingCircle(green_contour)
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)

        blue_contour, self.blue_area, self.blue_x, _ = self.get_biggest_contour(blue_mask)
        if blue_contour is not None:
            self.blue_found = True
            self.seen_blue = True
            (x, y), radius = cv2.minEnclosingCircle(blue_contour)
            cv2.circle(image, (int(x), int(y)), int(radius), (255, 0, 0), 2)

        try:
            cv2.namedWindow('camera_feed', cv2.WINDOW_NORMAL)
            cv2.imshow('camera_feed', image)
            cv2.resizeWindow('camera_feed', 420, 320)
            cv2.waitKey(1)
        except Exception:
            pass

    def control_loop(self):
        if self.state == RobotState.GO_TO_POINT:
            self.stop_robot()

            if not self.goal_running:
                if self.point_index < len(self.points):
                    x, y, yaw = self.points[self.point_index]
                    self.send_goal(x, y, yaw)
                else:
                    self.state = RobotState.SCAN
                    self.scan_finish_time = time.monotonic() + self.scan_time

        elif self.state == RobotState.SCAN:
            if self.scan_finish_time is not None and time.monotonic() < self.scan_finish_time:
                self.move_robot(0.0, self.scan_speed)
                return

            self.stop_robot()
            self.point_index += 1

            if self.point_index < len(self.points):
                self.state = RobotState.GO_TO_POINT
                return

            if self.blue_found:
                self.state = RobotState.GO_BLUE
            else:
                self.scan_finish_time = time.monotonic() + self.scan_time
                self.state = RobotState.SCAN

        elif self.state == RobotState.GO_BLUE:
            if not self.blue_found or self.blue_x is None or self.img_width == 0:
                self.move_robot(0.0, 0.2)
                return

            middle = self.img_width // 2
            error = self.blue_x - middle

            if self.blue_area >= self.blue_stop_area:
                self.stop_robot()
                self.state = RobotState.DONE
                self.get_logger().info('Stopped near blue box')
                return

            if error < -self.center_tol:
                self.move_robot(0.0, self.turn_speed)
            elif error > self.center_tol:
                self.move_robot(0.0, -self.turn_speed)
            else:
                ang = -0.0025 * error
                ang = max(min(ang, 0.15), -0.15)
                self.move_robot(self.forward_speed, ang)

        elif self.state == RobotState.DONE:
            self.stop_robot()

    def destroy_safely(self):
        self.stop_robot()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.destroy_node()


def main(args=None):
    rclpy.init(args=args)
    robot = Robot()

    try:
        rclpy.spin(robot)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            robot.stop_robot()
        robot.destroy_safely()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()