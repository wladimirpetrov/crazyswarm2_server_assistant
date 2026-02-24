#!/usr/bin/env python3
import math
import csv
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from crazyflie_interfaces.msg import Position


class CmdPosTrajectory(Node):
    """
    One-CF cmd_position runner.

    If do_return == True:
      Mission 1: TAKEOFF -> TRACK(traj1 forward-from-CSV) -> LAND
      Mission 2: TAKEOFF -> TRACK(traj2 reverse-from-CSV) -> LAND -> DONE

    If do_return == False:
      Mission 1 only, then DONE.

    Notes:
      - x,y,z are read from CSV columns L,M,N (0-based: 11,12,13)
      - heading/yaw is read from CSV column O (0-based: 14)
      - CSV heading is in radians, but we convert to DEGREES before publishing cmd_position.yaw
    """

    def __init__(self):
        super().__init__("ml_obst_av_try_one")

        self.declare_parameter("cf_name", "cf231")

        self.declare_parameter("x0", -1.75)
        self.declare_parameter("y0", -1.75)
        self.declare_parameter("z0", 0.0)
        self.declare_parameter("yaw0", 0.0)

        self.declare_parameter("takeoff_height", 1.0)
        self.declare_parameter("takeoff_duration", 3.0)
        self.declare_parameter("land_duration", 3.0)

        self.declare_parameter("traj_step_sec", 0.1)
        self.declare_parameter("publish_rate_hz", 20.0)

        self.declare_parameter("hold_after_takeoff_sec", 0.3)
        self.declare_parameter("hold_after_traj_sec", 0.3)
        self.declare_parameter("hold_after_land_sec", 0.5)

        self.declare_parameter("do_return", False)

        self.declare_parameter("traj_csv_name", "drone_trajectory.csv")

        # Downsample factor for CSV waypoints (e.g., 3 => keep every 3rd VALID row)
        self.declare_parameter("csv_stride", 5)

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

        self.csv_stride = int(self.get_parameter("csv_stride").value)
        if self.csv_stride < 1:
            self.get_logger().warn(f"csv_stride={self.csv_stride} is invalid; forcing to 1")
            self.csv_stride = 1

        self.qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.cmd_topic = f"/{self.cf_name}/cmd_position"
        self.cmd_pub = self.create_publisher(Position, self.cmd_topic, self.qos_be)

        # ------------------------------------------------------------
        # Load traj1 forward from CSV (L,M,N,O)
        # ------------------------------------------------------------
        csv_name = str(self.get_parameter("traj_csv_name").value).strip()
        self.x_vec_1, self.y_vec_1, self.z_vec_1, self.psi_vec_1 = self._load_traj_from_csv(
            csv_name, stride=self.csv_stride, reverse=False
        )

        # ------------------------------------------------------------
        # Load traj2 reverse from the SAME CSV (bottom -> top)
        # ------------------------------------------------------------
        self.x_vec_2, self.y_vec_2, self.z_vec_2, self.psi_vec_2 = self._load_traj_from_csv(
            csv_name, stride=self.csv_stride, reverse=True
        )

        self._validate_traj(self.x_vec_1, self.y_vec_1, self.z_vec_1, self.psi_vec_1, tag="traj1_forward_csv")
        if self.do_return:
            self._validate_traj(self.x_vec_2, self.y_vec_2, self.z_vec_2, self.psi_vec_2, tag="traj2_reverse_csv")

        self.trajs = [
            {"x": self.x_vec_1, "y": self.y_vec_1, "z": self.z_vec_1, "psi": self.psi_vec_1, "name": "traj1"},
        ]
        if self.do_return:
            self.trajs.append(
                {"x": self.x_vec_2, "y": self.y_vec_2, "z": self.z_vec_2, "psi": self.psi_vec_2, "name": "traj2"}
            )

        self.traj_id = 0
        self._load_current_traj()

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
        self.get_logger().info(f"csv_stride = {self.csv_stride}")
        self.get_logger().info(
            f"traj1 N={len(self.x_vec_1)}"
            + (f" | traj2 N={len(self.x_vec_2)}" if self.do_return else " | traj2 loaded but unused")
            + f" | step={self.traj_step_sec:.3f}s | pub={self.publish_rate_hz:.1f}Hz"
        )
        self.get_logger().info(
            f"traj1 first waypoint: x={self.x_vec_1[0]:.6f}, y={self.y_vec_1[0]:.6f}, z={self.z_vec_1[0]:.6f}, yaw_deg={self.psi_vec_1[0]:.6f}"
        )
        self.get_logger().info(
            f"traj2 first waypoint: x={self.x_vec_2[0]:.6f}, y={self.y_vec_2[0]:.6f}, z={self.z_vec_2[0]:.6f}, yaw_deg={self.psi_vec_2[0]:.6f}"
        )

    def _load_traj_from_csv(self, csv_filename: str, stride: int = 1, reverse: bool = False):
        """
        Reads x,y,z,yaw_rad from L,M,N,O then converts yaw_rad -> yaw_deg.
        Applies downsample stride on VALID rows, then optionally reverses order.
        """
        script_dir = Path(__file__).resolve().parent
        csv_path = script_dir / csv_filename
        if not csv_path.exists():
            raise RuntimeError(f"Trajectory CSV not found: {csv_path}")

        # Excel columns:
        #   L = 12th col (1-based) => index 11 (0-based)
        #   M => index 12
        #   N => index 13
        #   O => index 14 (heading/yaw in radians)
        col_x = 11
        col_y = 12
        col_z = 13
        col_yaw = 14

        if stride < 1:
            stride = 1

        raw_x, raw_y, raw_z, raw_yaw_deg = [], [], [], []

        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)

            # skip header row (row 1)
            try:
                next(reader)
            except StopIteration:
                raise RuntimeError(f"Trajectory CSV is empty: {csv_path}")

            for row in reader:
                if not row:
                    continue
                if len(row) <= col_yaw:
                    continue

                sx = row[col_x].strip()
                sy = row[col_y].strip()
                sz = row[col_z].strip()
                syaw = row[col_yaw].strip()

                if sx == "" or sy == "" or sz == "" or syaw == "":
                    continue

                try:
                    x = float(sx)
                    y = float(sy)
                    z = float(sz)
                    yaw_rad = float(syaw)
                except ValueError:
                    continue

                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and math.isfinite(yaw_rad)):
                    continue

                yaw_deg = math.degrees(yaw_rad)

                raw_x.append(x)
                raw_y.append(y)
                raw_z.append(z)
                raw_yaw_deg.append(yaw_deg)

        if len(raw_x) == 0:
            raise RuntimeError(f"No valid trajectory points loaded from: {csv_path}")

        # Downsample: keep every stride-th valid point: 0, stride, 2*stride, ...
        x_vec = raw_x[::stride]
        y_vec = raw_y[::stride]
        z_vec = raw_z[::stride]
        psi_vec = raw_yaw_deg[::stride]

        if len(x_vec) == 0:
            raise RuntimeError(f"No valid trajectory points after downsample (stride={stride}) from: {csv_path}")

        if reverse:
            x_vec = list(reversed(x_vec))
            y_vec = list(reversed(y_vec))
            z_vec = list(reversed(z_vec))
            psi_vec = list(reversed(psi_vec))

        return x_vec, y_vec, z_vec, psi_vec

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

    def _publish_cmd(self, x: float, y: float, z: float, yaw_deg: float):
        msg = Position()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        # yaw is published in DEGREES (converted from CSV radians)
        msg.yaw = float(yaw_deg)
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

            # debug print: current commanded waypoint (throttled so logs don't explode)
            if not hasattr(self, "_last_dbg_print"):
                self._last_dbg_print = now

            dbg_age = (now - self._last_dbg_print).nanoseconds * 1e-9
            if dbg_age >= 0.5:  # ~2 Hz
                self._last_dbg_print = now
                self.get_logger().info(
                    f"[TRACK] i={self.idx}/{self.N-1} "
                    f"cmd=(x={self.cmd_x:.3f}, y={self.cmd_y:.3f}, z={self.cmd_z:.3f}, yaw_deg={self.cmd_yaw:.3f})"
                )

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
