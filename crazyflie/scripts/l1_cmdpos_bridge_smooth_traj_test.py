#!/usr/bin/env python3
import math
import os
import csv

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import Position


class L1CmdPosBridge(Node):
    """
    Publishes /<cf_name>/cmd_position (Position setpoints).

    Startup behavior:
      - ramp z from z0 -> takeoff_height over takeoff_duration seconds
      - after takeoff, publish /l1_status = 1.0 (keep at 1 Hz)
      - then:

        NORMAL MODE (use_smoothed_z_trajectory = False):
          publish cmd_position at publish_rate_hz using a ZOH behavior:
            - listen to /x_input, /y_input, /z_input, /psi_input
            - hold the last *committed* (x,y,z,psi) at high-rate
            - switch to a new command set ONLY when ALL four topics have been updated
              at least once since the last commit.

        SMOOTHED-Z MODE (use_smoothed_z_trajectory = True):
          - ignore incoming /x_input, /y_input, /z_input, /psi_input for command generation
          - command fixed x = x0
          - command fixed y = y0
          - command fixed yaw = yaw0
          - read z from CSV column z_smoothed
          - every smoothed_z_step_sec, advance to the next consecutive row
          - if smoothed_z_use_zoh = False:
                at publish_rate_hz, linearly interpolate between consecutive rows
          - if smoothed_z_use_zoh = True:
                hold each CSV value constant until the next row time, with no interpolation

      - additionally:
          - subscribe to actual drone pose
          - publish actual x, y, z, psi to:
              /x_act, /y_act, /z_act, /psi_act
    """

    def __init__(self):
        super().__init__("l1_cmdpos_bridge")

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter("cf_name", "cf231")
        self.declare_parameter("takeoff_height", 0.6)
        self.declare_parameter("takeoff_duration", 2.0)
        self.declare_parameter("publish_rate_hz", 100.0)

        self.declare_parameter("x0", 0.0)
        self.declare_parameter("y0", 1.2)
        self.declare_parameter("yaw0", 0.0)
        self.declare_parameter("z0", 0.0)

        # Used for warning-only staleness reporting (NOT gating).
        self.declare_parameter("input_timeout_sec", 1.0)

        # Debug prints
        self.declare_parameter("debug_inputs", True)
        self.declare_parameter("debug_period_sec", 1.0)

        # Smoothed trajectory functionality
        self.declare_parameter("use_smoothed_z_trajectory", True)
        self.declare_parameter("smoothed_z_csv", "z_input_data_smoothed.csv")
        self.declare_parameter("smoothed_z_step_sec", 0.5)
        self.declare_parameter("smoothed_z_use_zoh", True)   # NEW: if True, no interpolation
        self.declare_parameter("debug_smoothed_z", True)
        self.declare_parameter("debug_smoothed_z_period_sec", 1.0)

        # Actual pose topic
        self.declare_parameter("pose_topic", "")

        self.cf_name = str(self.get_parameter("cf_name").value)
        self.takeoff_height = float(self.get_parameter("takeoff_height").value)
        self.takeoff_duration = float(self.get_parameter("takeoff_duration").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.x0 = float(self.get_parameter("x0").value)
        self.y0 = float(self.get_parameter("y0").value)
        self.yaw0 = float(self.get_parameter("yaw0").value)
        self.z0 = float(self.get_parameter("z0").value)

        self.input_timeout_sec = float(self.get_parameter("input_timeout_sec").value)
        self.debug_inputs = bool(self.get_parameter("debug_inputs").value)
        self.debug_period_sec = float(self.get_parameter("debug_period_sec").value)

        self.use_smoothed_z_trajectory = bool(self.get_parameter("use_smoothed_z_trajectory").value)
        self.smoothed_z_csv = str(self.get_parameter("smoothed_z_csv").value)
        self.smoothed_z_step_sec = float(self.get_parameter("smoothed_z_step_sec").value)
        self.smoothed_z_use_zoh = bool(self.get_parameter("smoothed_z_use_zoh").value)
        self.debug_smoothed_z = bool(self.get_parameter("debug_smoothed_z").value)
        self.debug_smoothed_z_period_sec = float(self.get_parameter("debug_smoothed_z_period_sec").value)

        pose_topic_param = str(self.get_parameter("pose_topic").value).strip()
        if pose_topic_param:
            self.pose_topic = pose_topic_param
        else:
            self.pose_topic = f"/{self.cf_name}/pose"

        if self.smoothed_z_step_sec <= 1e-9:
            self.get_logger().warn("smoothed_z_step_sec must be > 0. Using 0.5 s.")
            self.smoothed_z_step_sec = 0.5

        # Resolve CSV path relative to this script directory if a relative path was given
        if os.path.isabs(self.smoothed_z_csv):
            self.smoothed_z_csv_path = self.smoothed_z_csv
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.smoothed_z_csv_path = os.path.join(script_dir, self.smoothed_z_csv)

        # ----------------------------
        # QoS
        # ----------------------------
        self.qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ----------------------------
        # Publishers/subscribers
        # ----------------------------
        self.cmd_topic = f"/{self.cf_name}/cmd_position"
        self.cmd_pub = self.create_publisher(Position, self.cmd_topic, self.qos_be)
        self.l1_status_pub = self.create_publisher(Float64, "/l1_status", self.qos_be)

        # Actual-state publishers
        self.x_act_pub = self.create_publisher(Float64, "/x_act", self.qos_be)
        self.y_act_pub = self.create_publisher(Float64, "/y_act", self.qos_be)
        self.z_act_pub = self.create_publisher(Float64, "/z_act", self.qos_be)
        self.psi_act_pub = self.create_publisher(Float64, "/psi_act", self.qos_be)

        # Inputs from MPC subscriber_node
        self.x_sub = self.create_subscription(Float64, "/x_input", self._x_cb, self.qos_rel)
        self.y_sub = self.create_subscription(Float64, "/y_input", self._y_cb, self.qos_rel)
        self.z_sub = self.create_subscription(Float64, "/z_input", self._z_cb, self.qos_rel)
        self.psi_sub = self.create_subscription(Float64, "/psi_input", self._psi_cb, self.qos_rel)

        # Actual pose subscriber
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self._pose_cb,
            self.qos_be
        )

        # ----------------------------
        # Latest raw input values + timestamps + sequence counters
        # ----------------------------
        self._x_val = None
        self._y_val = None
        self._z_val = None
        self._psi_val = None

        self._x_t = None
        self._y_t = None
        self._z_t = None
        self._psi_t = None

        self._x_seq = 0
        self._y_seq = 0
        self._z_seq = 0
        self._psi_seq = 0

        self._committed_seq = (0, 0, 0, 0)

        self._have_committed_set = False
        self._cmd_x = self.x0
        self._cmd_y = self.y0
        self._cmd_z = self.takeoff_height
        self._cmd_psi = self.yaw0

        # Latest actual state
        self._x_act = None
        self._y_act = None
        self._z_act = None
        self._psi_act = None

        # Debug / throttles
        self._last_missing_log_time_sec = -1e9
        self._last_debug_log_time_sec = -1e9
        self._last_stale_warn_time_sec = -1e9
        self._last_smoothed_log_time_sec = -1e9

        # State machine
        self.state = "TAKEOFF"  # TAKEOFF -> RUN
        self.t0 = self.get_clock().now()
        self.run_start_time = None

        # Smoothed trajectory storage
        self._traj_z = []

        if self.use_smoothed_z_trajectory:
            self._load_smoothed_z_csv(self.smoothed_z_csv_path)

        # Timing
        self.dt = 1.0 / max(self.publish_rate_hz, 1e-6)
        self.cmd_timer = self.create_timer(self.dt, self._cmd_timer_cb)
        self.status_timer = self.create_timer(1.0, self._status_timer_cb)

        self.get_logger().info(f"Node started. Publishing takeoff ramp to {self.cmd_topic}")
        self.get_logger().info("Subscribing to /x_input /y_input /z_input /psi_input")
        self.get_logger().info(f"Subscribing to actual pose topic: {self.pose_topic}")
        self.get_logger().info("Publishing actual state to /x_act /y_act /z_act /psi_act")

        if self.use_smoothed_z_trajectory:
            mode_str = "ZOH hold (no interpolation)" if self.smoothed_z_use_zoh else "linear interpolation"
            self.get_logger().info(
                "Mode: use_smoothed_z_trajectory = TRUE. "
                f"After takeoff: x={self.x0:.3f}, y={self.y0:.3f}, yaw={self.yaw0:.3f}, "
                f"z from CSV rows: {self.smoothed_z_csv_path}, "
                f"advance every {self.smoothed_z_step_sec:.3f}s, "
                f"publish at {self.publish_rate_hz:.1f} Hz, "
                f"playback_mode={mode_str}"
            )
        else:
            self.get_logger().info(
                f"ZOH behavior: publishing at {self.publish_rate_hz:.1f} Hz; "
                "switch to a new set only when all 4 topics updated since last switch."
            )

    # ----------------------------
    # CSV loading for smoothed-z mode
    # ----------------------------
    def _load_smoothed_z_csv(self, csv_path: str):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Smoothed Z CSV file not found: {csv_path}")

        traj_z = []

        with open(csv_path, "r", newline="") as f:
            sample = f.read(4096)
            f.seek(0)

            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            except csv.Error:
                dialect = csv.excel

            reader = csv.DictReader(f, dialect=dialect)

            if reader.fieldnames is None:
                raise RuntimeError(f"CSV file has no header: {csv_path}")

            fieldnames = [name.strip() for name in reader.fieldnames]

            if "z_smoothed" in fieldnames:
                z_col = "z_smoothed"
            elif "z_input" in fieldnames:
                z_col = "z_input"
            else:
                raise RuntimeError(
                    f'CSV must contain "z_smoothed" or "z_input". Found: {fieldnames}'
                )

            for row in reader:
                try:
                    z_val = float(str(row[z_col]).strip())
                except (ValueError, KeyError, TypeError):
                    continue

                if math.isfinite(z_val):
                    traj_z.append(z_val)

        if len(traj_z) < 2:
            raise RuntimeError(f"Not enough valid z points in CSV: {csv_path}")

        self._traj_z = traj_z

        total_duration = (len(self._traj_z) - 1) * self.smoothed_z_step_sec
        self.get_logger().info(
            f"Loaded smoothed Z trajectory from CSV: {csv_path} | "
            f"points={len(self._traj_z)} | "
            f"playback_duration={total_duration:.3f}s | "
            f"z_start={self._traj_z[0]:.4f} | z_end={self._traj_z[-1]:.4f}"
        )

    def _get_smoothed_z_at_time(self, t_sec: float) -> float:
        """
        Playback is based on CSV row order only:
          - every smoothed_z_step_sec, move to next consecutive row
          - if smoothed_z_use_zoh = True:
                hold that row value until the next step
          - if smoothed_z_use_zoh = False:
                interpolate linearly between consecutive rows
          - after last row, hold final value
        """
        if not self._traj_z:
            return self.takeoff_height

        if len(self._traj_z) == 1:
            return self._traj_z[0]

        if t_sec <= 0.0:
            return self._traj_z[0]

        seg_idx = int(t_sec / self.smoothed_z_step_sec)

        if self.smoothed_z_use_zoh:
            if seg_idx >= len(self._traj_z):
                return self._traj_z[-1]
            return self._traj_z[seg_idx]

        # Linear interpolation mode
        if seg_idx >= len(self._traj_z) - 1:
            return self._traj_z[-1]

        seg_t0 = seg_idx * self.smoothed_z_step_sec
        alpha = (t_sec - seg_t0) / self.smoothed_z_step_sec
        alpha = max(0.0, min(1.0, alpha))

        z0 = self._traj_z[seg_idx]
        z1 = self._traj_z[seg_idx + 1]

        return z0 + alpha * (z1 - z0)

    # ----------------------------
    # Input callbacks
    # ----------------------------
    def _x_cb(self, msg: Float64):
        self._x_val = float(msg.data)
        self._x_t = self.get_clock().now()
        self._x_seq += 1
        self._debug_log_inputs()

    def _y_cb(self, msg: Float64):
        self._y_val = float(msg.data)
        self._y_t = self.get_clock().now()
        self._y_seq += 1
        self._debug_log_inputs()

    def _z_cb(self, msg: Float64):
        self._z_val = float(msg.data)
        self._z_t = self.get_clock().now()
        self._z_seq += 1
        self._debug_log_inputs()

    def _psi_cb(self, msg: Float64):
        self._psi_val = float(msg.data)
        self._psi_t = self.get_clock().now()
        self._psi_seq += 1
        self._debug_log_inputs()

    def _pose_cb(self, msg: PoseStamped):
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        z = float(msg.pose.position.z)

        qx = float(msg.pose.orientation.x)
        qy = float(msg.pose.orientation.y)
        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)

        psi = self._yaw_from_quaternion(qx, qy, qz, qw)

        self._x_act = x
        self._y_act = y
        self._z_act = z
        self._psi_act = psi

        self._publish_actual_state(x, y, z, psi)

    # ----------------------------
    # /l1_status publisher
    # ----------------------------
    def _status_timer_cb(self):
        if self.state == "RUN":
            m = Float64()
            m.data = 1.0
            self.l1_status_pub.publish(m)

    # ----------------------------
    # Main loop
    # ----------------------------
    def _cmd_timer_cb(self):
        now = self.get_clock().now()

        if self.state == "TAKEOFF":
            t = (now - self.t0).nanoseconds * 1e-9
            if self.takeoff_duration <= 1e-6:
                alpha = 1.0
            else:
                alpha = max(0.0, min(1.0, t / self.takeoff_duration))

            z_cmd = self.z0 + alpha * (self.takeoff_height - self.z0)

            self._publish_cmd(self.x0, self.y0, z_cmd, self.yaw0)

            if alpha >= 1.0:
                self.state = "RUN"
                self.run_start_time = now

                if self.use_smoothed_z_trajectory:
                    self.get_logger().info(
                        "Takeoff complete. Publishing /l1_status = 1 and following smoothed Z trajectory."
                    )
                else:
                    self.get_logger().info(
                        "Takeoff complete. Publishing /l1_status = 1 and forwarding ZOH commands."
                    )
            return

        # RUN mode, smoothed-z trajectory mode
        if self.use_smoothed_z_trajectory:
            run_t = 0.0
            if self.run_start_time is not None:
                run_t = (now - self.run_start_time).nanoseconds * 1e-9

            z_cmd = self._get_smoothed_z_at_time(run_t)

            now_sec = now.nanoseconds * 1e-9
            if self.debug_smoothed_z and (now_sec - self._last_smoothed_log_time_sec) >= self.debug_smoothed_z_period_sec:
                seg_idx = int(run_t / self.smoothed_z_step_sec)
                if seg_idx >= len(self._traj_z):
                    seg_idx = len(self._traj_z) - 1

                playback_mode = "ZOH" if self.smoothed_z_use_zoh else "interp"

                self.get_logger().info(
                    f"[SMOOTHED_Z] mode={playback_mode} t={run_t:.3f}s idx={seg_idx} "
                    f"x={self.x0:.3f} y={self.y0:.3f} z={z_cmd:.4f} yaw={self.yaw0:.3f}"
                )
                self._last_smoothed_log_time_sec = now_sec

            self._publish_cmd(self.x0, self.y0, z_cmd, self.yaw0)
            return

        # RUN mode, original behavior
        self._try_commit_new_set()

        if not self._have_committed_set:
            self._log_missing_inputs_throttled(2.0)
            self._publish_cmd(self.x0, self.y0, self.takeoff_height, self.yaw0)
            return

        self._warn_if_stale_throttled(now, period_sec=2.0)
        self._publish_cmd(self._cmd_x, self._cmd_y, self._cmd_z, self._cmd_psi)

    # ----------------------------
    # Commit logic
    # ----------------------------
    def _try_commit_new_set(self):
        if self._x_val is None or self._y_val is None or self._z_val is None or self._psi_val is None:
            return

        cur_seq = (self._x_seq, self._y_seq, self._z_seq, self._psi_seq)

        if (
            cur_seq[0] > self._committed_seq[0]
            and cur_seq[1] > self._committed_seq[1]
            and cur_seq[2] > self._committed_seq[2]
            and cur_seq[3] > self._committed_seq[3]
        ):
            self._cmd_x = self._finite_or(self._x_val, self._cmd_x)
            self._cmd_y = self._finite_or(self._y_val, self._cmd_y)
            self._cmd_z = self._finite_or(self._z_val, self._cmd_z)
            self._cmd_psi = self._finite_or(self._psi_val, self._cmd_psi)

            self._committed_seq = cur_seq
            if not self._have_committed_set:
                self._have_committed_set = True

            self.get_logger().info(
                f"[COMMIT] New set committed: x={self._cmd_x:.4f} y={self._cmd_y:.4f} "
                f"z={self._cmd_z:.4f} psi={self._cmd_psi:.4f} | seq={self._committed_seq}"
            )

    # ----------------------------
    # Logging helpers
    # ----------------------------
    def _log_missing_inputs_throttled(self, period_sec: float):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if (now_sec - self._last_missing_log_time_sec) < period_sec:
            return

        missing = []
        if self._x_val is None:
            missing.append("/x_input")
        if self._y_val is None:
            missing.append("/y_input")
        if self._z_val is None:
            missing.append("/z_input")
        if self._psi_val is None:
            missing.append("/psi_input")

        self.get_logger().warn(
            f"RUN mode but no committed set yet (waiting for first full set). "
            f"Missing: {', '.join(missing) if missing else 'none'}"
        )
        self._last_missing_log_time_sec = now_sec

    def _debug_log_inputs(self):
        if not self.debug_inputs:
            return
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if (now_sec - self._last_debug_log_time_sec) < self.debug_period_sec:
            return

        def fmt(v):
            return "None" if v is None else f"{v:.4f}"

        self.get_logger().info(
            f"[rx] x={fmt(self._x_val)} y={fmt(self._y_val)} z={fmt(self._z_val)} psi={fmt(self._psi_val)} | "
            f"seq=({self._x_seq},{self._y_seq},{self._z_seq},{self._psi_seq})"
        )
        self._last_debug_log_time_sec = now_sec

    def _warn_if_stale_throttled(self, now, period_sec: float):
        if self.input_timeout_sec <= 0.0:
            return

        now_sec = now.nanoseconds * 1e-9
        if (now_sec - self._last_stale_warn_time_sec) < period_sec:
            return

        def age(tstamp):
            if tstamp is None:
                return None
            return (now - tstamp).nanoseconds * 1e-9

        ages = {
            "/x_input": age(self._x_t),
            "/y_input": age(self._y_t),
            "/z_input": age(self._z_t),
            "/psi_input": age(self._psi_t),
        }

        stale = {k: v for k, v in ages.items() if (v is None) or (v > self.input_timeout_sec)}
        if stale:
            parts = []
            for k, v in stale.items():
                if v is None:
                    parts.append(f"{k}: never")
                else:
                    parts.append(f"{k}: {v:.3f}s")
            self.get_logger().warn(
                f"Inputs are older than input_timeout_sec={self.input_timeout_sec:.3f}s "
                f"(but continuing ZOH): {', '.join(parts)}"
            )
            self._last_stale_warn_time_sec = now_sec

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def _finite_or(val: float, fallback: float) -> float:
        if val is None:
            return float(fallback)
        if not math.isfinite(val):
            return float(fallback)
        return float(val)

    @staticmethod
    def _yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def _publish_actual_state(self, x: float, y: float, z: float, psi: float):
        mx = Float64()
        my = Float64()
        mz = Float64()
        mpsi = Float64()

        mx.data = float(x)
        my.data = float(y)
        mz.data = float(z)
        mpsi.data = float(psi)

        self.x_act_pub.publish(mx)
        self.y_act_pub.publish(my)
        self.z_act_pub.publish(mz)
        self.psi_act_pub.publish(mpsi)

    def _publish_cmd(self, x: float, y: float, z: float, yaw: float):
        msg = Position()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        msg.yaw = float(yaw)
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = L1CmdPosBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
