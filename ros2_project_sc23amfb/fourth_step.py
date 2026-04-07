import math
import time
from enum import Enum

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from nav2_msgs.action import NavigateToPose
from cv_bridge import CvBridge, CvBridgeError


class RobotState(Enum):
    SEARCH = 1
    SCAN = 2
    APPROACH_BLUE = 3
    RECOVER = 4
    DONE = 5


class Robot(Node):
    def __init__(self):
        super().__init__('robot')

        # Publishers / subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Nav2 action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # CV bridge
        self.bridge = CvBridge()

        # State
        self.state = RobotState.SEARCH
        self.goal_in_progress = False
        self.current_goal_handle = None
        self.scan_end_time = None
        self.waypoint_index = 0
        self.blue_goal_cancelled = False

        # Recovery state
        self.recover_stage = None
        self.recover_end_time = None

        # Seen flags
        self.seen_red = False
        self.seen_green = False
        self.seen_blue = False

        # Current frame detections
        self.red_detected = False
        self.green_detected = False
        self.blue_detected = False

        self.red_area = 0
        self.green_area = 0
        self.blue_area = 0

        self.red_cx = None
        self.green_cx = None
        self.blue_cx = None

        self.image_width = 0

        # Laser / obstacle data
        self.front_min_range = float('inf')
        self.obstacle_threshold = 0.28   # if wall is closer than this, recover
        self.reverse_time = 0.8
        self.turn_time = 1.0

        # Detection tuning
        self.sensitivity = 12
        self.min_area = 400

        # Blue approach tuning
        self.blue_stop_area = 120500
        self.center_tolerance = 40
        self.forward_speed = 0.10
        self.turn_speed = 0.25

        # Scan tuning
        self.scan_duration = 6.0
        self.scan_turn_speed = 0.35

        # Your current waypoints
        self.waypoints = [
            (2.811505280235851, -6.463797541259088, 0.0),
            (-4.544747037905132, -1.8917900163693924, 0.0),
        ]

        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Coursework robot node started.')

    # ---------------------------
    # Utility helpers
    # ---------------------------
    def yaw_to_quaternion(self, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return qz, qw

    def publish_cmd(self, lin=0.0, ang=0.0):
        msg = Twist()
        msg.linear.x = lin
        msg.angular.z = ang
        if rclpy.ok():
            self.cmd_pub.publish(msg)

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)

    # ---------------------------
    # Laser scan / obstacle check
    # ---------------------------
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)

        # Remove invalid values
        valid = np.isfinite(ranges)
        ranges = ranges[valid]

        if ranges.size == 0:
            self.front_min_range = float('inf')
            return

        # Use front cone from both ends of scan array
        # Turtlebot front is around index 0 / wrap-around
        full_ranges = np.array(msg.ranges, dtype=np.float32)
        finite = np.isfinite(full_ranges)
        full_ranges = np.where(finite, full_ranges, np.inf)

        front_left = full_ranges[:20]
        front_right = full_ranges[-20:]
        front = np.concatenate((front_left, front_right))

        if front.size == 0:
            self.front_min_range = float('inf')
        else:
            self.front_min_range = float(np.min(front))

    def obstacle_ahead(self):
        return self.front_min_range < self.obstacle_threshold

    def start_recovery(self):
        if self.state != RobotState.RECOVER:
            self.get_logger().info('Obstacle detected during blue approach. Starting recovery.')
            self.state = RobotState.RECOVER
            self.recover_stage = 'reverse'
            self.recover_end_time = time.monotonic() + self.reverse_time

    # ---------------------------
    # Nav2 goal handling
    # ---------------------------
    def send_nav_goal(self, x, y, yaw):
        if self.goal_in_progress:
            return

        if not self.nav_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn('Nav2 action server not ready yet.')
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

        self.get_logger().info(f'Sending goal: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}')
        self.goal_in_progress = True
        self.blue_goal_cancelled = False

        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected.')
            self.goal_in_progress = False
            return

        self.get_logger().info('Goal accepted.')
        self.current_goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        self.goal_in_progress = False
        self.current_goal_handle = None

        if self.state == RobotState.APPROACH_BLUE:
            return

        self.get_logger().info('Goal finished. Starting scan.')
        self.state = RobotState.SCAN
        self.scan_end_time = time.monotonic() + self.scan_duration

    def cancel_current_goal(self):
        if self.current_goal_handle is not None and not self.blue_goal_cancelled:
            self.get_logger().info('Cancelling current Nav2 goal for blue approach.')
            cancel_future = self.current_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done_callback)
            self.blue_goal_cancelled = True

    def cancel_done_callback(self, future):
        self.goal_in_progress = False
        self.current_goal_handle = None
        self.get_logger().info('Goal cancel requested.')

    # ---------------------------
    # Vision
    # ---------------------------
    def extract_largest_contour(self, mask):
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
            cx, cy = None, None

        return c, area, cx, cy

    def image_callback(self, data):
        self.red_detected = False
        self.green_detected = False
        self.blue_detected = False

        self.red_area = 0
        self.green_area = 0
        self.blue_area = 0

        self.red_cx = None
        self.green_cx = None
        self.blue_cx = None

        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        self.image_width = image.shape[1]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red uses two HSV ranges
        red_lower1 = np.array([0, 120, 80])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 120, 80])
        red_upper2 = np.array([180, 255, 255])

        green_lower = np.array([60 - self.sensitivity, 100, 80])
        green_upper = np.array([60 + self.sensitivity, 255, 255])

        blue_lower = np.array([120 - self.sensitivity, 100, 80])
        blue_upper = np.array([120 + self.sensitivity, 255, 255])

        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        # Red
        red_contour, red_area, red_cx, red_cy = self.extract_largest_contour(red_mask)
        if red_contour is not None:
            self.red_detected = True
            self.red_area = red_area
            self.red_cx = red_cx
            self.seen_red = True

            (x, y), radius = cv2.minEnclosingCircle(red_contour)
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.putText(image, f'RED {int(red_area)}', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Green
        green_contour, green_area, green_cx, green_cy = self.extract_largest_contour(green_mask)
        if green_contour is not None:
            self.green_detected = True
            self.green_area = green_area
            self.green_cx = green_cx
            self.seen_green = True

            (x, y), radius = cv2.minEnclosingCircle(green_contour)
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.putText(image, f'GREEN {int(green_area)}', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Blue
        blue_contour, blue_area, blue_cx, blue_cy = self.extract_largest_contour(blue_mask)
        if blue_contour is not None:
            self.blue_detected = True
            self.blue_area = blue_area
            self.blue_cx = blue_cx
            self.seen_blue = True

            (x, y), radius = cv2.minEnclosingCircle(blue_contour)
            cv2.circle(image, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            cv2.putText(image, f'BLUE {int(blue_area)}', (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(
            image,
            f'Seen -> R:{self.seen_red} G:{self.seen_green} B:{self.seen_blue}',
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            f'State: {self.state.name}',
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        red_view = cv2.bitwise_and(image, image, mask=red_mask)
        green_view = cv2.bitwise_and(image, image, mask=green_mask)
        blue_view = cv2.bitwise_and(image, image, mask=blue_mask)

        try:
            cv2.namedWindow('camera_feed', cv2.WINDOW_NORMAL)
            cv2.imshow('camera_feed', image)
            cv2.resizeWindow('camera_feed', 420, 320)

            cv2.namedWindow('red_filtered', cv2.WINDOW_NORMAL)
            cv2.imshow('red_filtered', red_view)
            cv2.resizeWindow('red_filtered', 320, 240)

            cv2.namedWindow('green_filtered', cv2.WINDOW_NORMAL)
            cv2.imshow('green_filtered', green_view)
            cv2.resizeWindow('green_filtered', 320, 240)

            cv2.namedWindow('blue_filtered', cv2.WINDOW_NORMAL)
            cv2.imshow('blue_filtered', blue_view)
            cv2.resizeWindow('blue_filtered', 320, 240)

            cv2.waitKey(1)
        except Exception:
            pass

    # ---------------------------
    # Control logic
    # ---------------------------
    def next_waypoint(self):
        wp = self.waypoints[self.waypoint_index]
        self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
        return wp

    def control_loop(self):
        if (
            self.state != RobotState.DONE
            and self.seen_red
            and self.seen_green
            and self.blue_detected
        ):
            if self.state != RobotState.APPROACH_BLUE and self.state != RobotState.RECOVER:
                self.get_logger().info('Red and green seen. Blue visible. Starting final approach.')
                self.state = RobotState.APPROACH_BLUE

        if self.state == RobotState.SEARCH:
            self.stop_robot()

            if not self.goal_in_progress:
                x, y, yaw = self.next_waypoint()
                self.send_nav_goal(x, y, yaw)

        elif self.state == RobotState.SCAN:
            if self.seen_red and self.seen_green and self.blue_detected:
                self.state = RobotState.APPROACH_BLUE
                self.stop_robot()
                return

            if self.scan_end_time is not None and time.monotonic() < self.scan_end_time:
                self.publish_cmd(0.0, self.scan_turn_speed)
            else:
                self.stop_robot()
                self.state = RobotState.SEARCH

        elif self.state == RobotState.APPROACH_BLUE:
            self.cancel_current_goal()

            if not self.blue_detected or self.blue_cx is None or self.image_width == 0:
                self.publish_cmd(0.0, 0.2)
                return

            # Basic obstacle recovery if too close to wall in front
            if self.obstacle_ahead():
                self.start_recovery()
                return

            image_center = self.image_width // 2
            error = self.blue_cx - image_center

            if (
                self.blue_area >= self.blue_stop_area
                and abs(error) <= self.center_tolerance
            ):
                self.stop_robot()
                self.state = RobotState.DONE
                self.get_logger().info('Stopped near blue box. Task complete.')
                return

            if error < -self.center_tolerance:
                self.publish_cmd(0.0, self.turn_speed)
            elif error > self.center_tolerance:
                self.publish_cmd(0.0, -self.turn_speed)
            else:
                ang = -0.0025 * error
                ang = max(min(ang, 0.15), -0.15)
                self.publish_cmd(self.forward_speed, ang)

        elif self.state == RobotState.RECOVER:
            now = time.monotonic()

            if self.recover_stage == 'reverse':
                if now < self.recover_end_time:
                    self.publish_cmd(-0.08, 0.0)
                else:
                    self.recover_stage = 'turn'
                    self.recover_end_time = now + self.turn_time

            elif self.recover_stage == 'turn':
                if now < self.recover_end_time:
                    self.publish_cmd(0.0, 0.5)
                else:
                    self.recover_stage = None
                    self.recover_end_time = None
                    self.state = RobotState.APPROACH_BLUE

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