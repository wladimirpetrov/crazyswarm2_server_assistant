#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

from crazyflie_py import Crazyswarm


class L1TakeoffCmdPos(Node):
    """
    Behavior:
      1) Take off to Z=0.5
      2) Publish /l1_status = 1 (Float64) to enable your L1 node
      3) Subscribe to: /xend, /yend, /psiend, /z_augment, /psi_augment
      4) Command cf.goTo([xend, yend, z_augment], yaw=psiend, duration=cmd_dt) in a loop
    """

    def __init__(self, swarm: Crazyswarm, cf_name: str = "cf231"):
        super().__init__("l1_takeoff_cmdpos")

        self.swarm = swarm
        self.timeHelper = swarm.timeHelper
        self.allcfs = swarm.allcfs

        # Pick a single CF by name if possible; otherwise use the first enabled one
        self.cf = None
        for cf in self.allcfs.crazyflies:
            if getattr(cf, "name", "") == cf_name:
                self.cf = cf
                break
        if self.cf is None:
            if len(self.allcfs.crazyflies) == 0:
                raise RuntimeError("No crazyflies found in allcfs.crazyflies")
            self.cf = self.allcfs.crazyflies[0]

        self.get_logger().info(f"Using Crazyflie: name={getattr(self.cf,'name','<unknown>')}")

        # ---- pubs/subs ----
        self.l1_status_pub = self.create_publisher(Float64, "/l1_status", 10)

        self.create_subscription(Float64, "/xend", self._xend_cb, 10)
        self.create_subscription(Float64, "/yend", self._yend_cb, 10)
        self.create_subscription(Float64, "/psiend", self._psiend_cb, 10)
        self.create_subscription(Float64, "/z_augment", self._zaug_cb, 10)
        self.create_subscription(Float64, "/psi_augment", self._psiaug_cb, 10)  # subscribed but not used (for now)

        # ---- latest values ----
        self.xend = 0.0
        self.yend = 0.0
        self.psiend = 0.0
        self.z_aug = 0.0
        self.psi_aug = 0.0

        self.have_x = False
        self.have_y = False
        self.have_psi = False
        self.have_z = False

    # ---- callbacks ----
    def _xend_cb(self, msg: Float64):
        self.xend = float(msg.data)
        self.have_x = True

    def _yend_cb(self, msg: Float64):
        self.yend = float(msg.data)
        self.have_y = True

    def _psiend_cb(self, msg: Float64):
        self.psiend = float(msg.data)
        self.have_psi = True

    def _zaug_cb(self, msg: Float64):
        self.z_aug = float(msg.data)
        self.have_z = True

    def _psiaug_cb(self, msg: Float64):
        self.psi_aug = float(msg.data)

    def publish_l1_status(self, val: float):
        m = Float64()
        m.data = float(val)
        self.l1_status_pub.publish(m)

    def run(self):
        Z = 0.5

        # 1) TAKEOFF
        self.get_logger().info(f"Takeoff to Z={Z} ...")
        self.allcfs.takeoff(targetHeight=Z, duration=1.0 + Z)
        self.timeHelper.sleep(1.5 + Z)

        # go to initialPosition + [0,0,Z] like your example
        pos0 = np.array(self.cf.initialPosition) + np.array([0.0, 0.0, Z])
        self.cf.goTo(pos0, 0.0, 0.5)
        self.timeHelper.sleep(0.6)

        # 2) ENABLE L1
        self.get_logger().info("Publishing /l1_status = 1 to enable L1...")
        # Publish multiple times for robustness
        for _ in range(10):
            self.publish_l1_status(1.0)
            rclpy.spin_once(self, timeout_sec=0.0)
            self.timeHelper.sleep(0.05)

        # 3) RUN CONTROL LOOP
        cmd_dt = 0.05  # 20 Hz
        self.get_logger().info(
            "Running command loop at 20 Hz: x=/xend, y=/yend, z=/z_augment, yaw=/psiend"
        )

        while rclpy.ok():
            # process incoming msgs
            rclpy.spin_once(self, timeout_sec=0.0)

            # Keep L1 enabled
            self.publish_l1_status(1.0)

            # If inputs not ready, just hold current position at Z
            if not (self.have_x and self.have_y and self.have_psi and self.have_z):
                # hold at initial XY and Z
                self.cf.goTo(pos0, 0.0, cmd_dt)
                self.timeHelper.sleep(cmd_dt)
                continue

            # Use your mapping:
            x_cmd = self.xend
            y_cmd = self.yend
            z_cmd = self.z_aug
            yaw_cmd = self.psiend  # per your request (ignore /psi_augment for now)

            pos_cmd = np.array([x_cmd, y_cmd, z_cmd], dtype=float)
            self.cf.goTo(pos_cmd, yaw_cmd, cmd_dt)

            self.timeHelper.sleep(cmd_dt)


def main():
    if not rclpy.ok():
        rclpy.init()

    # Crazyswarm will connect to crazyflie_server (must be running already)
    swarm = Crazyswarm()

    node = L1TakeoffCmdPos(swarm=swarm, cf_name="cf231")
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

