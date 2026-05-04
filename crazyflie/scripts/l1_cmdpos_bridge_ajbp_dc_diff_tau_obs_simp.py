#!/usr/bin/env python3

import csv
import os
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64
from crazyflie_interfaces.msg import Position


class L1CmdPosBridgeCsvPlayer(Node):
    """
    Simplified CSV-based cmd_position publisher.

    This node does NOT use:
      - /x_vector
      - /y_vector
      - /x_augment
      - /y_augment
      - Bernstein extraction
      - interpolation mode
      - segment mode
      - final capture
      - pose feedback

    Behavior:
      1. TAKEOFF:
         - publish x0, y0
         - ramp z from z0 to takeoff_height
         - yaw = yaw0

      2. RUN:
         - read x/y from CSV file
         - publish trajectory at publish_rate_hz
         - if trajectory_slowdown_factor = 1.0:
             one CSV row per timer tick
         - if trajectory_slowdown_factor > 1.0:
             interpolate between CSV rows and move slower
         - z = fixed_z_cmd
         - yaw = fixed_psi_cmd

      3. HOLD:
         - after CSV is finished, hold the last CSV x/y point
         - z = fixed_z_cmd
         - yaw = fixed_psi_cmd

    Expected CSV from MATLAB writetable:
        x,y
        -1.2,1.8
        -1.1999,1.8
        ...

    The node starts reading from the 2nd physical row by default,
    meaning it skips the CSV header row.
    """

    def __init__(self):
        super().__init__("l1_cmdpos_bridge_ajbp_dc_diff_tau_obs_simp")

        # ============================================================
        # Parameters
        # ============================================================

        self.declare_parameter("cf_name", "cf231")

        # CSV file path.
        # Can be absolute or relative to the directory where you run ros2 run.
        self.declare_parameter("csv_file", "xy_data_smoothened_xy.csv")

        # MATLAB writetable creates a header row: x,y
        self.declare_parameter("csv_has_header", True)

        # Physical CSV row number to start from.
        # Row 1 = header if csv_has_header=True.
        # Row 2 = first actual data row.
        self.declare_parameter("csv_start_physical_row", 2)

        # Publishing frequency.
        self.declare_parameter("publish_rate_hz", 100.0)

        # Trajectory slowdown factor.
        # 1.0 = normal speed, one CSV row per timer tick.
        # 2.0 = 2x slower.
        # 3.0 = 3x slower.
        # 5.0 = 5x slower.
        self.declare_parameter("trajectory_slowdown_factor", 3.0)

        # Takeoff settings.
        self.declare_parameter("takeoff_height", 0.6)
        self.declare_parameter("takeoff_duration", 2.0)

        # Initial takeoff position.
        self.declare_parameter("x0", -1.2)
        self.declare_parameter("y0", 1.8)
        self.declare_parameter("z0", 0.0)
        self.declare_parameter("yaw0", 0.0)

        # Constant z and psi during CSV trajectory.
        self.declare_parameter("fixed_z_cmd", 0.6)
        self.declare_parameter("fixed_psi_cmd", 0.0)

        # Hold last point after CSV finishes.
        self.declare_parameter("hold_last_point", True)

        # Optional /l1_status publisher.
        self.declare_parameter("publish_l1_status", True)

        # Debug logging period.
        self.declare_parameter("debug_period_sec", 0.5)

        # ============================================================
        # Read parameters
        # ============================================================

        self.cf_name = str(self.get_parameter("cf_name").value)
        self.csv_file = str(self.get_parameter("csv_file").value)

        self.csv_has_header = bool(self.get_parameter("csv_has_header").value)

        self.csv_start_physical_row = int(
            self.get_parameter("csv_start_physical_row").value
        )

        if self.csv_start_physical_row < 1:
            self.get_logger().warn(
                "csv_start_physical_row must be >= 1. Using 2."
            )
            self.csv_start_physical_row = 2

        if self.csv_has_header and self.csv_start_physical_row < 2:
            self.get_logger().warn(
                "csv_has_header=True, so csv_start_physical_row must be >= 2. Using 2."
            )
            self.csv_start_physical_row = 2

        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        if self.publish_rate_hz <= 0.0:
            self.get_logger().warn("publish_rate_hz must be > 0. Using 100.0 Hz.")
            self.publish_rate_hz = 100.0

        self.trajectory_slowdown_factor = float(
            self.get_parameter("trajectory_slowdown_factor").value
        )

        if self.trajectory_slowdown_factor < 1.0:
            self.get_logger().warn(
                "trajectory_slowdown_factor must be >= 1.0. Using 1.0."
            )
            self.trajectory_slowdown_factor = 1.0

        self.takeoff_height = float(self.get_parameter("takeoff_height").value)
        self.takeoff_duration = float(self.get_parameter("takeoff_duration").value)

        if self.takeoff_duration < 0.0:
            self.get_logger().warn("takeoff_duration must be >= 0. Using 2.0 s.")
            self.takeoff_duration = 2.0

        self.x0 = float(self.get_parameter("x0").value)
        self.y0 = float(self.get_parameter("y0").value)
        self.z0 = float(self.get_parameter("z0").value)
        self.yaw0 = float(self.get_parameter("yaw0").value)

        self.fixed_z_cmd = float(self.get_parameter("fixed_z_cmd").value)
        self.fixed_psi_cmd = float(self.get_parameter("fixed_psi_cmd").value)

        self.hold_last_point = bool(self.get_parameter("hold_last_point").value)
        self.publish_l1_status = bool(self.get_parameter("publish_l1_status").value)

        self.debug_period_sec = float(self.get_parameter("debug_period_sec").value)
        if self.debug_period_sec <= 0.0:
            self.debug_period_sec = 0.5

        # ============================================================
        # QoS
        # ============================================================

        self.qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ============================================================
        # Publishers
        # ============================================================

        self.cmd_topic = f"/{self.cf_name}/cmd_position"

        self.cmd_pub = self.create_publisher(
            Position,
            self.cmd_topic,
            self.qos_be,
        )

        self.l1_status_pub = self.create_publisher(
            Float64,
            "/l1_status",
            self.qos_be,
        )

        # ============================================================
        # Load CSV trajectory
        # ============================================================

        self.xy_points = self._load_xy_csv(
            csv_file=self.csv_file,
            has_header=self.csv_has_header,
            start_physical_row=self.csv_start_physical_row,
        )

        if len(self.xy_points) == 0:
            raise RuntimeError(
                "CSV trajectory is empty after applying csv_start_physical_row."
            )

        self.csv_index = 0

        # Floating CSV index used for trajectory slowdown.
        # If slowdown = 1.0, this advances by 1.0 every timer tick.
        # If slowdown = 3.0, this advances by 1/3 every timer tick.
        self.csv_float_index = 0.0

        self.last_cmd_x = self.x0
        self.last_cmd_y = self.y0
        self.last_cmd_z = self.z0
        self.last_cmd_psi = self.yaw0

        self.state = "TAKEOFF"
        self.t0 = self.get_clock().now()

        self._finished_logged = False
        self._last_debug_log_time_sec = -1e9

        self.dt = 1.0 / self.publish_rate_hz

        self.cmd_timer = self.create_timer(
            self.dt,
            self._cmd_timer_cb,
        )

        if self.publish_l1_status:
            self.status_timer = self.create_timer(
                1.0,
                self._status_timer_cb,
            )
        else:
            self.status_timer = None

        self.get_logger().info("Node started.")
        self.get_logger().info(f"Publishing to: {self.cmd_topic}")
        self.get_logger().info(f"CSV file: {self.csv_file}")
        self.get_logger().info(f"CSV has header: {self.csv_has_header}")
        self.get_logger().info(
            f"CSV start physical row: {self.csv_start_physical_row}"
        )
        self.get_logger().info(f"Loaded CSV trajectory points: {len(self.xy_points)}")
        self.get_logger().info(f"Publish rate: {self.publish_rate_hz:.3f} Hz")
        self.get_logger().info(
            f"Trajectory slowdown factor: {self.trajectory_slowdown_factor:.3f}"
        )

        self.get_logger().info(
            f"Takeoff config: "
            f"x0={self.x0:.6f}, y0={self.y0:.6f}, z0={self.z0:.6f}, "
            f"takeoff_height={self.takeoff_height:.6f}, "
            f"takeoff_duration={self.takeoff_duration:.3f}, yaw0={self.yaw0:.6f}"
        )

        self.get_logger().info(
            f"RUN config: fixed_z_cmd={self.fixed_z_cmd:.6f}, "
            f"fixed_psi_cmd={self.fixed_psi_cmd:.6f}"
        )

        self.get_logger().info(
            f"First CSV point: x={self.xy_points[0][0]:.6f}, "
            f"y={self.xy_points[0][1]:.6f}"
        )

        self.get_logger().info(
            f"Last CSV point: x={self.xy_points[-1][0]:.6f}, "
            f"y={self.xy_points[-1][1]:.6f}"
        )

    # ============================================================
    # CSV loader
    # ============================================================

    def _load_xy_csv(
        self,
        csv_file: str,
        has_header: bool,
        start_physical_row: int,
    ) -> List[Tuple[float, float]]:

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f'CSV file "{csv_file}" not found.')

        xy_points: List[Tuple[float, float]] = []

        with open(csv_file, "r", newline="") as f:
            if has_header:
                reader = csv.DictReader(f)

                if reader.fieldnames is None:
                    raise RuntimeError("CSV header not found.")

                fieldnames_original = reader.fieldnames
                fieldnames_lower = [
                    name.strip().lower() for name in fieldnames_original
                ]

                if "x" not in fieldnames_lower or "y" not in fieldnames_lower:
                    raise RuntimeError(
                        "CSV with header must contain columns named x and y. "
                        f"Found columns: {fieldnames_original}"
                    )

                x_key = fieldnames_original[fieldnames_lower.index("x")]
                y_key = fieldnames_original[fieldnames_lower.index("y")]

                # DictReader consumes row 1 as header.
                # First data row is physical row 2.
                for data_row_index, row in enumerate(reader, start=1):
                    physical_row = data_row_index + 1

                    if physical_row < start_physical_row:
                        continue

                    try:
                        x_val = float(row[x_key])
                        y_val = float(row[y_key])
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to parse x/y at physical CSV row "
                            f"{physical_row}: {row}"
                        ) from exc

                    xy_points.append((x_val, y_val))

            else:
                reader = csv.reader(f)

                for physical_row, row in enumerate(reader, start=1):
                    if physical_row < start_physical_row:
                        continue

                    if len(row) < 2:
                        raise RuntimeError(
                            f"CSV row {physical_row} has fewer than 2 columns: {row}"
                        )

                    try:
                        x_val = float(row[0])
                        y_val = float(row[1])
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to parse x/y at physical CSV row "
                            f"{physical_row}: {row}"
                        ) from exc

                    xy_points.append((x_val, y_val))

        if len(xy_points) == 0:
            raise RuntimeError(
                f'CSV file "{csv_file}" contains no usable x/y data '
                f"from physical row {start_physical_row}."
            )

        return xy_points

    # ============================================================
    # Timers
    # ============================================================

    def _status_timer_cb(self):
        if self.state in ("RUN", "HOLD"):
            msg = Float64()
            msg.data = 1.0
            self.l1_status_pub.publish(msg)

    def _cmd_timer_cb(self):
        now = self.get_clock().now()

        if self.state == "TAKEOFF":
            self._run_takeoff(now)
            return

        if self.state == "RUN":
            self._run_csv_trajectory(now)
            return

        if self.state == "HOLD":
            self._run_hold(now)
            return

        self.get_logger().warn(
            f"Unknown state '{self.state}'. Switching to HOLD."
        )
        self.state = "HOLD"
        self._run_hold(now)

    # ============================================================
    # State logic
    # ============================================================

    def _run_takeoff(self, now):
        elapsed = (now - self.t0).nanoseconds * 1e-9

        if self.takeoff_duration <= 1e-9:
            alpha = 1.0
        else:
            alpha = elapsed / self.takeoff_duration
            alpha = max(0.0, min(1.0, alpha))

        z_cmd = self.z0 + alpha * (self.takeoff_height - self.z0)

        self._publish_cmd(
            x=self.x0,
            y=self.y0,
            z=z_cmd,
            yaw=self.yaw0,
            source="TAKEOFF",
        )

        if alpha >= 1.0:
            self.state = "RUN"
            self.csv_index = 0
            self.csv_float_index = 0.0

            self.get_logger().info(
                "Takeoff complete. Starting CSV x/y trajectory."
            )

    def _run_csv_trajectory(self, now):
        last_index = len(self.xy_points) - 1

        if self.csv_float_index <= float(last_index):
            i0 = int(self.csv_float_index)
            i1 = min(i0 + 1, last_index)

            beta = self.csv_float_index - float(i0)

            x0, y0 = self.xy_points[i0]
            x1, y1 = self.xy_points[i1]

            # Linear interpolation between adjacent CSV rows.
            # This allows slower motion while still publishing at 100 Hz.
            x_cmd = x0 + beta * (x1 - x0)
            y_cmd = y0 + beta * (y1 - y0)

            self._publish_cmd(
                x=x_cmd,
                y=y_cmd,
                z=self.fixed_z_cmd,
                yaw=self.fixed_psi_cmd,
                source="CSV",
            )

            self.csv_index = i0 + 1

            # Normal speed:
            #   slowdown = 1.0 -> +1.0 row per timer tick.
            #
            # Slower:
            #   slowdown = 2.0 -> +0.5 row per timer tick.
            #   slowdown = 3.0 -> +0.333 row per timer tick.
            #   slowdown = 5.0 -> +0.2 row per timer tick.
            self.csv_float_index += 1.0 / self.trajectory_slowdown_factor

            now_sec = now.nanoseconds * 1e-9
            if (now_sec - self._last_debug_log_time_sec) >= self.debug_period_sec:
                physical_row = self.csv_start_physical_row + i0

                self.get_logger().info(
                    f"[CSV] physical_row~={physical_row}, "
                    f"float_index={self.csv_float_index:.3f}/{len(self.xy_points)}, "
                    f"slowdown={self.trajectory_slowdown_factor:.3f}, "
                    f"x={x_cmd:.6f}, y={y_cmd:.6f}, "
                    f"z={self.fixed_z_cmd:.6f}, "
                    f"psi={self.fixed_psi_cmd:.6f}"
                )

                self._last_debug_log_time_sec = now_sec

            return

        self.state = "HOLD"

        if not self._finished_logged:
            self.get_logger().warn(
                "CSV trajectory finished. Switching to HOLD at last point."
            )
            self._finished_logged = True

        self._run_hold(now)

    def _run_hold(self, now):
        if self.hold_last_point and len(self.xy_points) > 0:
            x_cmd, y_cmd = self.xy_points[-1]
        else:
            x_cmd = self.last_cmd_x
            y_cmd = self.last_cmd_y

        self._publish_cmd(
            x=x_cmd,
            y=y_cmd,
            z=self.fixed_z_cmd,
            yaw=self.fixed_psi_cmd,
            source="HOLD",
        )

        now_sec = now.nanoseconds * 1e-9
        if (now_sec - self._last_debug_log_time_sec) >= self.debug_period_sec:
            self.get_logger().info(
                f"[HOLD] x={x_cmd:.6f}, y={y_cmd:.6f}, "
                f"z={self.fixed_z_cmd:.6f}, psi={self.fixed_psi_cmd:.6f}"
            )

            self._last_debug_log_time_sec = now_sec

    # ============================================================
    # Publisher
    # ============================================================

    def _publish_cmd(self, x, y, z, yaw, source="CMD"):
        msg = Position()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        msg.yaw = float(yaw)

        self.cmd_pub.publish(msg)

        self.last_cmd_x = float(x)
        self.last_cmd_y = float(y)
        self.last_cmd_z = float(z)
        self.last_cmd_psi = float(yaw)


def main():
    rclpy.init()

    node = L1CmdPosBridgeCsvPlayer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
