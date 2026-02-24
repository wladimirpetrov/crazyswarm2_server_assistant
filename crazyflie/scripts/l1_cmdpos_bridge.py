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
      - then continuously forward:
          x   <- /xend
          y   <- /yend
          z   <- /z_augment
          yaw <- /psiend
        If required inputs missing/stale, keep hovering at (x0, y0, takeoff_height, yaw0).

    Notes:
      - /z_augment and /psi_augment are subscribed with TRANSIENT_LOCAL so we get the last value
        immediately even if we start late (requires publisher to be transient_local too).
      - We enforce "freshness" (staleness timeout) on ALL inputs.
      - We log when /z_augment is actually received (throttled) to verify data flow.
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

        # Safe hover defaults (use your YAML initial position)
        # self.declare_parameter("x0", 1.760813811)
        # self.declare_parameter("y0", -0.177819071)
        # self.declare_parameter("yaw0", 0.0)
        # self.declare_parameter("z0", 0.0)
        
        self.declare_parameter("x0", 0.0)
        self.declare_parameter("y0", 1.2)
        self.declare_parameter("yaw0", 0.0)
        self.declare_parameter("z0", 0.0)

        # Freshness timeout (seconds). Inputs older than this are considered invalid.
        # Use 0.2s if you publish ~20Hz; 1.0s if your chain is slower.
        self.declare_parameter("input_timeout_sec", 0.2)

        # Debug prints
        self.declare_parameter("debug_zaugment", True)
        self.declare_parameter("debug_zaugment_period_sec", 1.0)

        self.cf_name = str(self.get_parameter("cf_name").value)
        self.takeoff_height = float(self.get_parameter("takeoff_height").value)
        self.takeoff_duration = float(self.get_parameter("takeoff_duration").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.x0 = float(self.get_parameter("x0").value)
        self.y0 = float(self.get_parameter("y0").value)
        self.yaw0 = float(self.get_parameter("yaw0").value)
        self.z0 = float(self.get_parameter("z0").value)

        self.input_timeout_sec = float(self.get_parameter("input_timeout_sec").value)
        self.debug_zaugment = bool(self.get_parameter("debug_zaugment").value)
        self.debug_zaugment_period_sec = float(self.get_parameter("debug_zaugment_period_sec").value)

        # ----------------------------
        # QoS
        #   - crazyflie_server endpoints are typically BEST_EFFORT
        # ----------------------------
        self.qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Default RELIABLE volatile (works for typical C++/Simulink pubs)
        self.qos_rel = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # "Latched" RELIABLE transient_local (requires publisher transient_local)
        self.qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ----------------------------
        # Publishers/subscribers
        # ----------------------------
        self.cmd_topic = f"/{self.cf_name}/cmd_position"
        self.cmd_pub = self.create_publisher(Position, self.cmd_topic, self.qos_be)

        # Publish status as BEST_EFFORT (matches your l1 node subscription in your code)
        self.l1_status_pub = self.create_publisher(Float64, "/l1_status", self.qos_be)

        # Inputs:
        # - xend/yend/psiend typically come from your optimizer/simulink nodes (RELIABLE volatile is fine)
        # - z_augment and psi_augment we want "latched" behavior to avoid startup order sensitivity
        self.xend_sub = self.create_subscription(Float64, "/xend", self._xend_cb, self.qos_rel)
        self.yend_sub = self.create_subscription(Float64, "/yend", self._yend_cb, self.qos_rel)

        # IMPORTANT: subscribe to BOTH /z_augment and /psi_augment with TRANSIENT_LOCAL
        self.zaug_sub = self.create_subscription(Float64, "/z_augment", self._zaug_cb, self.qos_latched)
        self.paug_sub = self.create_subscription(Float64, "/psi_augment", self._paug_cb, self.qos_latched)

        # Your yaw command comes from /psiend (desired yaw), not from augmentation
        self.psiend_sub = self.create_subscription(Float64, "/psiend", self._psiend_cb, self.qos_rel)

        # Latest values
        self.xend = None
        self.yend = None
        self.zaug = None
        self.psiend = None

        # Last-received times (ROS time)
        self.xend_t = None
        self.yend_t = None
        self.zaug_t = None
        self.psiend_t = None

        # Debug / throttles
        self._last_missing_log_time_sec = -1e9
        self._last_zaug_debug_log_time_sec = -1e9

        # State machine
        self.state = "TAKEOFF"  # TAKEOFF -> RUN
        self.t0 = self.get_clock().now()

        # Timing
        self.dt = 1.0 / max(self.publish_rate_hz, 1e-6)
        self.cmd_timer = self.create_timer(self.dt, self._cmd_timer_cb)
        self.status_timer = self.create_timer(1.0, self._status_timer_cb)

        self.get_logger().info(f"Node started. Publishing takeoff ramp to {self.cmd_topic}")
        self.get_logger().info("Subscribing to /xend /yend /z_augment /psi_augment /psiend")
        self.get_logger().info(
            f"Freshness timeout: {self.input_timeout_sec:.3f}s. "
            "Inputs older than this are treated as missing/stale."
        )

    # ----------------------------
    # Input callbacks (store value + timestamp)
    # ----------------------------
    def _xend_cb(self, msg: Float64):
        self.xend = float(msg.data)
        self.xend_t = self.get_clock().now()

    def _yend_cb(self, msg: Float64):
        self.yend = float(msg.data)
        self.yend_t = self.get_clock().now()

    def _zaug_cb(self, msg: Float64):
        self.zaug = float(msg.data)
        self.zaug_t = self.get_clock().now()
        self._debug_log_zaugment(self.zaug)

    def _paug_cb(self, msg: Float64):
        # Not used for yaw command in this bridge (we use /psiend),
        # but we subscribe to confirm augmentation topic is alive and latched.
        # If you ever want yaw augmentation, you can switch yaw_cmd to this.
        _ = float(msg.data)

    def _psiend_cb(self, msg: Float64):
        self.psiend = float(msg.data)
        self.psiend_t = self.get_clock().now()

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
                self.get_logger().info("Takeoff complete. Publishing /l1_status = 1 and forwarding commands.")
            return

        # RUN mode
        ok, missing = self._inputs_ok(now)
        if not ok:
            self._log_missing_inputs_throttled(2.0, missing)
            # Keep hovering safely
            self._publish_cmd(self.x0, self.y0, self.takeoff_height, self.yaw0)
            return

        # Forward commands (values are guaranteed present + fresh here)
        x_cmd = self._finite_or(self.xend, self.x0)
        y_cmd = self._finite_or(self.yend, self.y0)
        z_cmd = self._finite_or(self.zaug, self.takeoff_height)
        yaw_cmd = self._finite_or(self.psiend, self.yaw0)

        self._publish_cmd(x_cmd, y_cmd, z_cmd, yaw_cmd)

    # ----------------------------
    # Freshness / validity checks
    # ----------------------------
    def _inputs_ok(self, now) -> tuple[bool, list[str]]:
        missing = []

        # We treat "missing" as either:
        #   - never received (timestamp None), OR
        #   - stale (> input_timeout_sec old)
        def is_fresh(tstamp):
            if tstamp is None:
                return False
            age = (now - tstamp).nanoseconds * 1e-9
            return age <= self.input_timeout_sec

        if not is_fresh(self.xend_t):
            missing.append("/xend")
        if not is_fresh(self.yend_t):
            missing.append("/yend")
        if not is_fresh(self.zaug_t):
            missing.append("/z_augment")
        if not is_fresh(self.psiend_t):
            missing.append("/psiend")

        return (len(missing) == 0), missing

    # ----------------------------
    # Logging helpers
    # ----------------------------
    def _log_missing_inputs_throttled(self, period_sec: float, missing: list[str]):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if (now_sec - self._last_missing_log_time_sec) < period_sec:
            return

        # Also report ages for quick diagnosis
        ages = []
        now = self.get_clock().now()

        def age_str(name, tstamp):
            if tstamp is None:
                return f"{name}: never"
            age = (now - tstamp).nanoseconds * 1e-9
            return f"{name}: {age:.3f}s"

        ages.append(age_str("/xend", self.xend_t))
        ages.append(age_str("/yend", self.yend_t))
        ages.append(age_str("/z_augment", self.zaug_t))
        ages.append(age_str("/psiend", self.psiend_t))

        self.get_logger().warn(
            f"RUN mode but missing/stale inputs: {', '.join(missing)}. "
            f"Holding hover. Ages: [{'; '.join(ages)}]"
        )
        self._last_missing_log_time_sec = now_sec

    def _debug_log_zaugment(self, z_val: float):
        if not self.debug_zaugment:
            return
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if (now_sec - self._last_zaug_debug_log_time_sec) < self.debug_zaugment_period_sec:
            return
        self.get_logger().info(f"[rx] /z_augment = {z_val:.6f}")
        self._last_zaug_debug_log_time_sec = now_sec

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

