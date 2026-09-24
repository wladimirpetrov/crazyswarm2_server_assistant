#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64


class Cf231PoseToActualTopics(Node):
    """
    Subscribes to:
        /cf231/pose              geometry_msgs/PoseStamped

    Publishes:
        /x_act                   std_msgs/Float64
        /y_act                   std_msgs/Float64
        /z_act                   std_msgs/Float64
        /yaw_act                 std_msgs/Float64

    yaw_act is published in radians.
    """

    def __init__(self) -> None:
        super().__init__("cf231_pose_to_actual_topics")

        self.declare_parameter("pose_topic", "/cf231/pose")
        self.declare_parameter("x_topic", "/x_act")
        self.declare_parameter("y_topic", "/y_act")
        self.declare_parameter("z_topic", "/z_act")
        self.declare_parameter("yaw_topic", "/yaw_act")

        self.pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        self.x_topic = self.get_parameter("x_topic").get_parameter_value().string_value
        self.y_topic = self.get_parameter("y_topic").get_parameter_value().string_value
        self.z_topic = self.get_parameter("z_topic").get_parameter_value().string_value
        self.yaw_topic = self.get_parameter("yaw_topic").get_parameter_value().string_value

        self.x_pub = self.create_publisher(Float64, self.x_topic, 10)
        self.y_pub = self.create_publisher(Float64, self.y_topic, 10)
        self.z_pub = self.create_publisher(Float64, self.z_topic, 10)
        self.yaw_pub = self.create_publisher(Float64, self.yaw_topic, 10)

        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            10,
        )

        self.get_logger().info("---------------------------------------------")
        self.get_logger().info("cf231_pose_to_actual_topics started")
        self.get_logger().info(f"subscribing to : {self.pose_topic}")
        self.get_logger().info(f"publishing x   : {self.x_topic}")
        self.get_logger().info(f"publishing y   : {self.y_topic}")
        self.get_logger().info(f"publishing z   : {self.z_topic}")
        self.get_logger().info(f"publishing yaw : {self.yaw_topic}")
        self.get_logger().info("yaw_act is in radians")
        self.get_logger().info("---------------------------------------------")

    def pose_callback(self, msg: PoseStamped) -> None:
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        z = float(msg.pose.position.z)

        qx = float(msg.pose.orientation.x)
        qy = float(msg.pose.orientation.y)
        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)

        yaw = self.quaternion_to_yaw(qx, qy, qz, qw)

        self.publish_float(self.x_pub, x)
        self.publish_float(self.y_pub, y)
        self.publish_float(self.z_pub, z)
        self.publish_float(self.yaw_pub, yaw)

    def quaternion_to_yaw(self, qx: float, qy: float, qz: float, qw: float) -> float:
        """
        Convert quaternion to yaw angle around z-axis.

        Returns:
            yaw in radians, in range [-pi, pi].
        """
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw

    def publish_float(self, publisher, value: float) -> None:
        msg = Float64()
        msg.data = float(value)
        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = Cf231PoseToActualTopics()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
