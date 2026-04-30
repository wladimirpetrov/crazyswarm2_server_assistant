#!/usr/bin/env python3
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty


class ExtractIterKeyboardNode(Node):
    def __init__(self):
        super().__init__("extract_iter_keyboard_node")

        self.pub = self.create_publisher(Empty, "/increase_extract_iter", 10)

        self.old_terminal_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        self.get_logger().info("Keyboard control started.")
        self.get_logger().info("Press 'i' to increase extract_iter_minus_1_x/y by +1.")
        self.get_logger().info("Press 'q' to quit.")

    def restore_terminal(self):
        try:
            termios.tcsetattr(
                sys.stdin,
                termios.TCSADRAIN,
                self.old_terminal_settings,
            )
        except Exception:
            pass

    def get_key(self, timeout_sec=0.05):
        ready, _, _ = select.select([sys.stdin], [], [], timeout_sec)
        if ready:
            return sys.stdin.read(1)
        return None

    def publish_increase(self):
        msg = Empty()
        self.pub.publish(msg)
        self.get_logger().warn("Published /increase_extract_iter")


def main():
    rclpy.init()
    node = ExtractIterKeyboardNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            key = node.get_key(timeout_sec=0.01)

            if key == "i":
                node.publish_increase()

            elif key == "q":
                node.get_logger().info("Quitting keyboard node.")
                break

    except KeyboardInterrupt:
        pass

    finally:
        node.restore_terminal()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
