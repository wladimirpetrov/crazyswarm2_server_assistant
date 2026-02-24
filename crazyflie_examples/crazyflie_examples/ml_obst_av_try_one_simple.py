#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from crazyflie_interfaces.msg import Position


class CmdPosTrajectory(Node):
    """
    One-CF cmd_position runner.

    If do_return == True:
      Mission 1: TAKEOFF -> TRACK(traj1) -> LAND
      Mission 2: TAKEOFF -> TRACK(traj2) -> LAND -> DONE

    If do_return == False:
      Mission 1 only, then DONE.

    Only publishes /<cf_name>/cmd_position (no takeoff/land services/actions).
    """

    def __init__(self):
        super().__init__("ml_obst_av_try_one")

        self.declare_parameter("cf_name", "cf231")

        self.declare_parameter("x0", 0.0)
        self.declare_parameter("y0", 1.2)
        self.declare_parameter("z0", 0.0)
        self.declare_parameter("yaw0", 0.0)

        self.declare_parameter("takeoff_height", 0.5)
        self.declare_parameter("takeoff_duration", 2.0)
        self.declare_parameter("land_duration", 2.0)

        self.declare_parameter("traj_step_sec", 0.1)
        self.declare_parameter("publish_rate_hz", 20.0)

        self.declare_parameter("hold_after_takeoff_sec", 0.3)
        self.declare_parameter("hold_after_traj_sec", 0.3)
        self.declare_parameter("hold_after_land_sec", 0.5)

        # flag: run return trajectory or not
        self.declare_parameter("do_return", False)

        self.cf_name = str(self.get_parameter("cf_name").value)

        self.x0 = float(self.get_parameter("x0").value)
        self.y0 = float(self.get_parameter("y0").value)
        self.z0 = float(self.get_parameter("z0").value)
        self.yaw0 = float(self.get_parameter("yaw0").value)

        self.takeoff_height = float(self.get_parameter("takeoff_height").value)
        self.takeoff_duration = float(self.get_parameter("takeoff_duration").value)
        self.land_duration = float(self.get_parameter("land_duration").value)

        self.traj_step_sec = float(self.get_parameter("traj_step_sec").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.hold_after_takeoff_sec = float(self.get_parameter("hold_after_takeoff_sec").value)
        self.hold_after_traj_sec = float(self.get_parameter("hold_after_traj_sec").value)
        self.hold_after_land_sec = float(self.get_parameter("hold_after_land_sec").value)

        self.do_return = bool(self.get_parameter("do_return").value)

        self.qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.cmd_topic = f"/{self.cf_name}/cmd_position"
        self.cmd_pub = self.create_publisher(Position, self.cmd_topic, self.qos_be)

        # traj1: "go"
        self.x_vec_1 = [0.0, 0.1, 0.2, 0.3, 0.4]
        self.y_vec_1 = [1.2, 1.2, 1.2, 1.2, 1.2]
        self.z_vec_1 = [0.5, 0.5, 0.5, 0.5, 0.5]
        self.psi_vec_1 = [0.0, 0.0, 0.0, 0.0, 0.0]

        # traj2: "return"
        self.x_vec_2 = [0.4, 0.3, 0.2, 0.1, 0.0]
        self.y_vec_2 = [1.2, 1.2, 1.2, 1.2, 1.2]
        self.z_vec_2 = [0.5, 0.5, 0.5, 0.5, 0.5]
        self.psi_vec_2 = [0.0, 0.0, 0.0, 0.0, 0.0]

        self._validate_traj(self.x_vec_1, self.y_vec_1, self.z_vec_1, self.psi_vec_1, tag="traj1")
        if self.do_return:
            self._validate_traj(self.x_vec_2, self.y_vec_2, self.z_vec_2, self.psi_vec_2, tag="traj2")

        # build list of missions depending on flag
        self.trajs = [
            {"x": self.x_vec_1, "y": self.y_vec_1, "z": self.z_vec_1, "psi": self.psi_vec_1, "name": "traj1"},
        ]
        if self.do_return:
            self.trajs.append(
                {"x": self.x_vec_2, "y": self.y_vec_2, "z": self.z_vec_2, "psi": self.psi_vec_2, "name": "traj2"}
            )

        self.traj_id = 0

        self.x_vec = self.trajs[self.traj_id]["x"]
        self.y_vec = self.trajs[self.traj_id]["y"]
        self.z_vec = self.trajs[self.traj_id]["z"]
        self.psi_vec = self.trajs[self.traj_id]["psi"]
        self.N = len(self.x_vec)
        self.idx = 0

        self.ground_x = self.x0
        self.ground_y = self.y0
        self.ground_yaw = self.yaw0

        self.state = "TAKEOFF"  # TAKEOFF -> HOLD1 -> TRACK -> HOLD2 -> LAND -> HOLDG -> (next TAKEOFF) -> DONE
        self.t_state0 = self.get_clock().now()

        self.cmd_x = self.ground_x
        self.cmd_y = self.ground_y
        self.cmd_z = self.z0
        self.cmd_yaw = self.ground_yaw

        self._last_step_time = self.get_clock().now()

        self._land_start_z = None
        self._land_x = None
        self._land_y = None
        self._land_yaw = None

        self.dt_pub = 1.0 / max(self.publish_rate_hz, 1e-9)
        self.timer = self.create_timer(self.dt_pub, self._tick)

        self.get_logger().info(f"Publishing to {self.cmd_topic}")
        self.get_logger().info(f"do_return = {self.do_return}")
        self.get_logger().info(
            f"traj1 N={len(self.x_vec_1)}"
            + (f" | traj2 N={len(self.x_vec_2)}" if self.do_return else " | traj2 skipped")
            + f" | step={self.traj_step_sec:.3f}s | pub={self.publish_rate_hz:.1f}Hz"
        )

    def _validate_traj(self, x_vec, y_vec, z_vec, psi_vec, tag="traj"):
        n = len(x_vec)
        if n == 0:
            raise RuntimeError(f"{tag}: empty vectors")
        if not (len(y_vec) == len(z_vec) == len(psi_vec) == n):
            raise RuntimeError(
                f"{tag}: length mismatch len(x)={len(x_vec)} len(y)={len(y_vec)} len(z)={len(z_vec)} len(psi)={len(psi_vec)}"
            )
        for name, vec in [("x", x_vec), ("y", y_vec), ("z", z_vec), ("psi", psi_vec)]:
            for i, v in enumerate(vec):
                if v is None or not math.isfinite(float(v)):
                    raise RuntimeError(f"{tag}: non-finite {name}[{i}] = {v}")

    def _publish_cmd(self, x: float, y: float, z: float, yaw: float):
        msg = Position()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        msg.yaw = float(yaw)
        self.cmd_pub.publish(msg)

    def _elapsed(self) -> float:
        now = self.get_clock().now()
        return (now - self.t_state0).nanoseconds * 1e-9

    def _set_state(self, new_state: str):
        self.state = new_state
        self.t_state0 = self.get_clock().now()

    def _load_current_traj(self):
        self.x_vec = self.trajs[self.traj_id]["x"]
        self.y_vec = self.trajs[self.traj_id]["y"]
        self.z_vec = self.trajs[self.traj_id]["z"]
        self.psi_vec = self.trajs[self.traj_id]["psi"]
        self.N = len(self.x_vec)
        self.idx = 0

    def _tick(self):
        now = self.get_clock().now()

        if self.state == "TAKEOFF":
            t = self._elapsed()
            alpha = 1.0 if self.takeoff_duration <= 1e-9 else max(0.0, min(1.0, t / self.takeoff_duration))

            self.cmd_x = self.ground_x
            self.cmd_y = self.ground_y
            self.cmd_yaw = self.ground_yaw
            self.cmd_z = self.z0 + alpha * (self.takeoff_height - self.z0)

            self._publish_cmd(self.cmd_x, self.cmd_y, self.cmd_z, self.cmd_yaw)

            if alpha >= 1.0:
                self.cmd_z = self.takeoff_height
                self._set_state("HOLD1")
            return

        if self.state == "HOLD1":
            self._publish_cmd(self.cmd_x, self.cmd_y, self.takeoff_height, self.cmd_yaw)
            if self._elapsed() >= self.hold_after_takeoff_sec:
                self.idx = 0
                self.cmd_x = self.x_vec[0]
                self.cmd_y = self.y_vec[0]
                self.cmd_z = self.z_vec[0]
                self.cmd_yaw = self.psi_vec[0]
                self._last_step_time = now
                self._set_state("TRACK")
                self.get_logger().info(f"Starting {self.trajs[self.traj_id]['name']} (N={self.N})")
            return

        if self.state == "TRACK":
            self._publish_cmd(self.cmd_x, self.cmd_y, self.cmd_z, self.cmd_yaw)

            age = (now - self._last_step_time).nanoseconds * 1e-9
            if age >= self.traj_step_sec:
                self.idx += 1
                self._last_step_time = now

                if self.idx >= self.N:
                    self._set_state("HOLD2")
                else:
                    self.cmd_x = self.x_vec[self.idx]
                    self.cmd_y = self.y_vec[self.idx]
                    self.cmd_z = self.z_vec[self.idx]
                    self.cmd_yaw = self.psi_vec[self.idx]
            return

        if self.state == "HOLD2":
            self._publish_cmd(self.cmd_x, self.cmd_y, self.cmd_z, self.cmd_yaw)
            if self._elapsed() >= self.hold_after_traj_sec:
                self._land_start_z = float(self.cmd_z)
                self._land_x = float(self.cmd_x)
                self._land_y = float(self.cmd_y)
                self._land_yaw = float(self.cmd_yaw)
                self._set_state("LAND")
            return

        if self.state == "LAND":
            t = self._elapsed()
            alpha = 1.0 if self.land_duration <= 1e-9 else max(0.0, min(1.0, t / self.land_duration))

            z = self._land_start_z + alpha * (self.z0 - self._land_start_z)
            self._publish_cmd(self._land_x, self._land_y, z, self._land_yaw)

            if alpha >= 1.0:
                self.ground_x = float(self._land_x)
                self.ground_y = float(self._land_y)
                self.ground_yaw = float(self._land_yaw)

                self.cmd_x = self.ground_x
                self.cmd_y = self.ground_y
                self.cmd_z = self.z0
                self.cmd_yaw = self.ground_yaw

                self.get_logger().info(f"Landed after {self.trajs[self.traj_id]['name']}.")
                self._set_state("HOLDG")
            return

        if self.state == "HOLDG":
            self._publish_cmd(self.cmd_x, self.cmd_y, self.z0, self.cmd_yaw)

            if self._elapsed() >= self.hold_after_land_sec:
                if (self.traj_id + 1) < len(self.trajs):
                    self.traj_id += 1
                    self._load_current_traj()
                    self.get_logger().info(
                        f"Next mission: {self.trajs[self.traj_id]['name']} (N={self.N}). Taking off again..."
                    )
                    self._set_state("TAKEOFF")
                else:
                    self._set_state("DONE")
                    self.get_logger().info("Done: mission complete. Stopping node.")
                    self.timer.cancel()
            return

        if self.state == "DONE":
            return


def main():
    rclpy.init()
    node = CmdPosTrajectory()

    try:
        while rclpy.ok() and node.state != "DONE":
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().warn("Ctrl-C: pushing z=0 for a moment and exiting (best-effort).")
        try:
            for _ in range(40):
                node._publish_cmd(node.cmd_x, node.cmd_y, 0.0, node.cmd_yaw)
                rclpy.spin_once(node, timeout_sec=0.0)
        except Exception:
            pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

