#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64
from crazyflie_interfaces.msg import Position


class L1CmdPosBridge(Node):
    """
    Publishes /<cf_name>/cmd_position (Position setpoints).

    Startup behavior:
      - ramp z from z0 -> takeoff_height over takeoff_duration seconds
      - after takeoff, publish /l1_status = 1.0 (keep at 1 Hz)
      - then continuously publish cmd_position at publish_rate_hz using a ZOH behavior:
          - listen to /x_input, /y_input, /z_input, /psi_input
          - hold the last *committed* (x,y,z,psi) at high-rate
          - switch to a new command set ONLY when ALL four topics have been updated
            at least once since the last commit.
        If we have never committed a full set yet, hover at (x0, y0, takeoff_height, yaw0).

    Notes:
      - This is designed for slow/bursty MPC updates (e.g., ~0.6s).
      - input_timeout_sec is used only for WARNING logs (not gating).
    """

    def __init__(self):
        super().__init__("l1_cmdpos_bridge")

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter("cf_name", "cf231")
        self.declare_parameter("takeoff_height", 0.5)
        self.declare_parameter("takeoff_duration", 2.0)
        self.declare_parameter("publish_rate_hz", 20.0)

        self.declare_parameter("x0", 0.0)
        self.declare_parameter("y0", 1.2)
        self.declare_parameter("yaw0", 0.0)
        self.declare_parameter("z0", 0.0)

        # Used for warning-only staleness reporting (NOT gating).
        self.declare_parameter("input_timeout_sec", 1.0)

        # Debug prints
        self.declare_parameter("debug_inputs", True)
        self.declare_parameter("debug_period_sec", 1.0)

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

        # ----------------------------
        # QoS
        # ----------------------------
        # crazyflie_server endpoints are typically BEST_EFFORT
        self.qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # For inputs from your C++ subscriber_node:
        # Use RELIABLE + VOLATILE (matches default rclcpp QoS and avoids DURABILITY mismatch).
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

        # Publish status as BEST_EFFORT (matches your L1 node subscription)
        self.l1_status_pub = self.create_publisher(Float64, "/l1_status", self.qos_be)

        # Inputs from MPC subscriber_node
        self.x_sub = self.create_subscription(Float64, "/x_input", self._x_cb, self.qos_rel)
        self.y_sub = self.create_subscription(Float64, "/y_input", self._y_cb, self.qos_rel)
        self.z_sub = self.create_subscription(Float64, "/z_input", self._z_cb, self.qos_rel)
        self.psi_sub = self.create_subscription(Float64, "/psi_input", self._psi_cb, self.qos_rel)

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

        # Incremented each time that topic updates
        self._x_seq = 0
        self._y_seq = 0
        self._z_seq = 0
        self._psi_seq = 0

        # The seq tuple that was last *committed* into the active command set
        self._committed_seq = (0, 0, 0, 0)

        # The actual command set we publish at high-rate (ZOH)
        self._have_committed_set = False
        self._cmd_x = self.x0
        self._cmd_y = self.y0
        self._cmd_z = self.takeoff_height
        self._cmd_psi = self.yaw0

        # Debug / throttles
        self._last_missing_log_time_sec = -1e9
        self._last_debug_log_time_sec = -1e9
        self._last_stale_warn_time_sec = -1e9

        # State machine
        self.state = "TAKEOFF"  # TAKEOFF -> RUN
        self.t0 = self.get_clock().now()

        # Timing
        self.dt = 1.0 / max(self.publish_rate_hz, 1e-6)
        self.cmd_timer = self.create_timer(self.dt, self._cmd_timer_cb)
        self.status_timer = self.create_timer(1.0, self._status_timer_cb)

        self.get_logger().info(f"Node started. Publishing takeoff ramp to {self.cmd_topic}")
        self.get_logger().info("Subscribing to /x_input /y_input /z_input /psi_input")
        self.get_logger().info(
            f"ZOH behavior: publishing at {self.publish_rate_hz:.1f} Hz; "
            "switch to a new set only when all 4 topics updated since last switch."
        )

    # ----------------------------
    # Input callbacks (store value + timestamp + bump seq)
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

            # Hold x/y/yaw fixed during takeoff
            self._publish_cmd(self.x0, self.y0, z_cmd, self.yaw0)

            if alpha >= 1.0:
                self.state = "RUN"
                self.get_logger().info("Takeoff complete. Publishing /l1_status = 1 and forwarding ZOH commands.")
            return

        # RUN mode:
        # 1) Try to commit a new (x,y,z,psi) set if ALL four updated since last commit
        self._try_commit_new_set()

        # 2) If we don't have any committed set yet, hover safely
        if not self._have_committed_set:
            self._log_missing_inputs_throttled(2.0)
            self._publish_cmd(self.x0, self.y0, self.takeoff_height, self.yaw0)
            return

        # 3) Optionally warn if inputs are very old (but do NOT stop)
        self._warn_if_stale_throttled(now, period_sec=2.0)

        # 4) Publish the current committed set (ZOH) at high rate
        self._publish_cmd(self._cmd_x, self._cmd_y, self._cmd_z, self._cmd_psi)

    # ----------------------------
    # Commit logic: switch only when all 4 topics updated
    # ----------------------------
    def _try_commit_new_set(self):
        # Need at least one message from each topic
        if self._x_val is None or self._y_val is None or self._z_val is None or self._psi_val is None:
            return

        cur_seq = (self._x_seq, self._y_seq, self._z_seq, self._psi_seq)

        # Commit new set only if ALL four sequences advanced since last commit
        if (cur_seq[0] > self._committed_seq[0] and
            cur_seq[1] > self._committed_seq[1] and
            cur_seq[2] > self._committed_seq[2] and
            cur_seq[3] > self._committed_seq[3]):

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
        if self._x_val is None: missing.append("/x_input")
        if self._y_val is None: missing.append("/y_input")
        if self._z_val is None: missing.append("/z_input")
        if self._psi_val is None: missing.append("/psi_input")

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
