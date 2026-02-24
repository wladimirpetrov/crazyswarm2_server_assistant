#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import Position


class L1CmdPosBridgeCompTwoDrones(Node):
    """
    Extends your L1CmdPosBridge to command TWO drones:

    Drone 1 (cf231):
      - identical behavior to your current bridge:
          TAKEOFF: ramp z0 -> takeoff_height over takeoff_duration
          RUN: publish /l1_status=1.0 at 1Hz and forward:
              x   <- /xend
              y   <- /yend
              z   <- /z_augment
              yaw <- /psiend
          Freshness timeout applies to /xend /yend /z_augment /psiend.

    Drone 2 (cf232):
      - independent TAKEOFF: ramp z0_2 -> takeoff_height_2 over takeoff_duration_2
      - RUN: command fixed x/y/yaw (constants) and z from /zend2:
              x   = x2_const
              y   = y2_const
              yaw = yaw2_const
              z   <- /zend2
        Freshness timeout applies to /zend2.

      - Also subscribes to /<cf232_name>/pose, extracts z, publishes /z_act2 (Float64, transient_local).

    Notes:
      - /z_augment and /psi_augment are subscribed as TRANSIENT_LOCAL so we get last value immediately.
      - We keep hovering with safe defaults if inputs are missing/stale.
    """

    def __init__(self):
        super().__init__("l1_cmdpos_bridge_comp_two_drones")

        # ----------------------------
        # Parameters (Drone 1: cf231)
        # ----------------------------
        self.declare_parameter("cf1_name", "cf231")
        self.declare_parameter("takeoff_height", 0.5)
        self.declare_parameter("takeoff_duration", 2.0)
        self.declare_parameter("publish_rate_hz", 20.0)

        # Safe hover defaults for cf231
        self.declare_parameter("x0", 0.0)
        self.declare_parameter("y0", 1.2)
        self.declare_parameter("yaw0", 0.0)
        self.declare_parameter("z0", 0.0)

        # ----------------------------
        # Parameters (Drone 2: cf232)
        # ----------------------------
        self.declare_parameter("cf2_name", "cf232")

        # Constants (never change in RUN)
        self.declare_parameter("x2_const", 1.18)
        self.declare_parameter("y2_const", 1.22)
        self.declare_parameter("yaw2_const", 0.0)

        # Takeoff behavior for cf232
        self.declare_parameter("takeoff_height_2", 0.5)
        self.declare_parameter("takeoff_duration_2", 2.0)
        self.declare_parameter("z0_2", 0.0)

        # ----------------------------
        # General parameters
        # ----------------------------
        self.declare_parameter("input_timeout_sec", 0.2)

        # Debug prints
        self.declare_parameter("debug_zaugment", True)
        self.declare_parameter("debug_zaugment_period_sec", 1.0)

        # ----------------------------
        # Read params
        # ----------------------------
        self.cf1_name = str(self.get_parameter("cf1_name").value)
        self.takeoff_height = float(self.get_parameter("takeoff_height").value)
        self.takeoff_duration = float(self.get_parameter("takeoff_duration").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.x0 = float(self.get_parameter("x0").value)
        self.y0 = float(self.get_parameter("y0").value)
        self.yaw0 = float(self.get_parameter("yaw0").value)
        self.z0 = float(self.get_parameter("z0").value)

        self.cf2_name = str(self.get_parameter("cf2_name").value)
        self.x2_const = float(self.get_parameter("x2_const").value)
        self.y2_const = float(self.get_parameter("y2_const").value)
        self.yaw2_const = float(self.get_parameter("yaw2_const").value)

        self.takeoff_height_2 = float(self.get_parameter("takeoff_height_2").value)
        self.takeoff_duration_2 = float(self.get_parameter("takeoff_duration_2").value)
        self.z0_2 = float(self.get_parameter("z0_2").value)

        self.input_timeout_sec = float(self.get_parameter("input_timeout_sec").value)

        self.debug_zaugment = bool(self.get_parameter("debug_zaugment").value)
        self.debug_zaugment_period_sec = float(self.get_parameter("debug_zaugment_period_sec").value)

        # ----------------------------
        # QoS profiles
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

        self.qos_latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Latched publisher for /z_act2
        self.qos_latched_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ----------------------------
        # Publishers
        # ----------------------------
        self.cmd1_topic = f"/{self.cf1_name}/cmd_position"
        self.cmd2_topic = f"/{self.cf2_name}/cmd_position"

        self.cmd1_pub = self.create_publisher(Position, self.cmd1_topic, self.qos_be)
        self.cmd2_pub = self.create_publisher(Position, self.cmd2_topic, self.qos_be)

        # Status for L1 node (same as your current design)
        self.l1_status_pub = self.create_publisher(Float64, "/l1_status", self.qos_be)

        # cf232 extracted z actual
        self.z_act2_pub = self.create_publisher(Float64, "/z_act2", self.qos_latched_pub)

        # ----------------------------
        # Subscriptions (cf231 command chain)
        # ----------------------------
        self.xend_sub = self.create_subscription(Float64, "/xend", self._xend_cb, self.qos_rel)
        self.yend_sub = self.create_subscription(Float64, "/yend", self._yend_cb, self.qos_rel)

        self.zaug_sub = self.create_subscription(Float64, "/z_augment", self._zaug_cb, self.qos_latched)
        self.paug_sub = self.create_subscription(Float64, "/psi_augment", self._paug_cb, self.qos_latched)

        self.psiend_sub = self.create_subscription(Float64, "/psiend", self._psiend_cb, self.qos_rel)

        # ----------------------------
        # Subscriptions (cf232 desired + pose)
        # ----------------------------
        self.zend2_sub = self.create_subscription(Float64, "/zend2", self._zend2_cb, self.qos_rel)

        # Pose for cf232 (to extract z and publish /z_act2)
        self.cf2_pose_sub = self.create_subscription(
            PoseStamped, f"/{self.cf2_name}/pose", self._cf2_pose_cb, self.qos_be
        )

        # ----------------------------
        # Latest values + timestamps
        # ----------------------------
        # cf231 inputs
        self.xend = None
        self.yend = None
        self.zaug = None
        self.psiend = None

        self.xend_t = None
        self.yend_t = None
        self.zaug_t = None
        self.psiend_t = None

        # cf232 input
        self.zend2 = None
        self.zend2_t = None

        # cf232 actual z
        self.z2_act = None
        self.z2_act_t = None

        # Debug / throttles
        self._last_missing_log_time_sec = -1e9
        self._last_zaug_debug_log_time_sec = -1e9

        # ----------------------------
        # State machines per drone
        # ----------------------------
        self.state1 = "TAKEOFF"  # TAKEOFF -> RUN
        self.state2 = "TAKEOFF"  # TAKEOFF -> RUN

        self.t0_1 = self.get_clock().now()
        self.t0_2 = self.get_clock().now()

        # ----------------------------
        # Timers
        # ----------------------------
        self.dt = 1.0 / max(self.publish_rate_hz, 1e-6)
        self.cmd_timer = self.create_timer(self.dt, self._cmd_timer_cb)
        self.status_timer = self.create_timer(1.0, self._status_timer_cb)

        self.get_logger().info("Two-drone bridge started.")
        self.get_logger().info(f"cf231 cmd: {self.cmd1_topic} | cf232 cmd: {self.cmd2_topic}")
        self.get_logger().info("cf231 subscribes: /xend /yend /z_augment /psi_augment /psiend")
        self.get_logger().info("cf232 subscribes: /zend2 and /cf232/pose; publishes /z_act2")
        self.get_logger().info(
            f"Freshness timeout: {self.input_timeout_sec:.3f}s. "
            "Inputs older than this are treated as stale."
        )
        self.get_logger().info(
            f"cf232 fixed XY/yaw: x={self.x2_const:.3f}, y={self.y2_const:.3f}, yaw={self.yaw2_const:.3f}"
        )

    # ----------------------------
    # Input callbacks: cf231
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
        # but subscribed to confirm augmentation topic is alive and latched.
        _ = float(msg.data)

    def _psiend_cb(self, msg: Float64):
        self.psiend = float(msg.data)
        self.psiend_t = self.get_clock().now()

    # ----------------------------
    # Input callbacks: cf232
    # ----------------------------
    def _zend2_cb(self, msg: Float64):
        self.zend2 = float(msg.data)
        self.zend2_t = self.get_clock().now()

    def _cf2_pose_cb(self, msg: PoseStamped):
        # extract z for cf232 and publish /z_act2 (latched)
        z = float(msg.pose.position.z)
        self.z2_act = z
        self.z2_act_t = self.get_clock().now()

        m = Float64()
        m.data = z
        self.z_act2_pub.publish(m)

    # ----------------------------
    # /l1_status publisher
    # ----------------------------
    def _status_timer_cb(self):
        # Keep same semantics as your original node:
        # only publish status once cf231 is in RUN mode.
        if self.state1 == "RUN":
            m = Float64()
            m.data = 1.0
            self.l1_status_pub.publish(m)

    # ----------------------------
    # Main loop (drives both drones at same rate)
    # ----------------------------
    def _cmd_timer_cb(self):
        now = self.get_clock().now()

        # ---- Drone 1: cf231 ----
        if self.state1 == "TAKEOFF":
            self._do_takeoff_1(now)
        else:
            self._do_run_1(now)

        # ---- Drone 2: cf232 ----
        if self.state2 == "TAKEOFF":
            self._do_takeoff_2(now)
        else:
            self._do_run_2(now)

    # ----------------------------
    # Drone 1 helpers
    # ----------------------------
    def _do_takeoff_1(self, now):
        t = (now - self.t0_1).nanoseconds * 1e-9
        if self.takeoff_duration <= 1e-6:
            alpha = 1.0
        else:
            alpha = max(0.0, min(1.0, t / self.takeoff_duration))

        z_cmd = self.z0 + alpha * (self.takeoff_height - self.z0)

        # Hold x/y/yaw fixed during takeoff
        self._publish_cmd(self.cmd1_pub, self.cf1_name, self.x0, self.y0, z_cmd, self.yaw0)

        if alpha >= 1.0:
            self.state1 = "RUN"
            self.get_logger().info("cf231 takeoff complete. Publishing /l1_status=1 and forwarding commands.")

    def _do_run_1(self, now):
        ok, missing = self._inputs_ok_cf231(now)
        if not ok:
            self._log_missing_inputs_throttled(2.0, missing)
            self._publish_cmd(self.cmd1_pub, self.cf1_name, self.x0, self.y0, self.takeoff_height, self.yaw0)
            return

        x_cmd = self._finite_or(self.xend, self.x0)
        y_cmd = self._finite_or(self.yend, self.y0)
        z_cmd = self._finite_or(self.zaug, self.takeoff_height)
        yaw_cmd = self._finite_or(self.psiend, self.yaw0)

        self._publish_cmd(self.cmd1_pub, self.cf1_name, x_cmd, y_cmd, z_cmd, yaw_cmd)

    # ----------------------------
    # Drone 2 helpers
    # ----------------------------
    def _do_takeoff_2(self, now):
        t = (now - self.t0_2).nanoseconds * 1e-9
        if self.takeoff_duration_2 <= 1e-6:
            alpha = 1.0
        else:
            alpha = max(0.0, min(1.0, t / self.takeoff_duration_2))

        z_cmd = self.z0_2 + alpha * (self.takeoff_height_2 - self.z0_2)

        # Hold x/y/yaw fixed during takeoff (constants)
        self._publish_cmd(self.cmd2_pub, self.cf2_name, self.x2_const, self.y2_const, z_cmd, self.yaw2_const)

        if alpha >= 1.0:
            self.state2 = "RUN"
            self.get_logger().info("cf232 takeoff complete. Now using /zend2 for z command (XY/yaw fixed).")

    def _do_run_2(self, now):
        ok, missing = self._inputs_ok_cf232(now)
        if not ok:
            # Log missing separately (still uses same throttled logger)
            self._log_missing_inputs_throttled(2.0, missing)
            self._publish_cmd(
                self.cmd2_pub, self.cf2_name, self.x2_const, self.y2_const, self.takeoff_height_2, self.yaw2_const
            )
            return

        z_cmd = self._finite_or(self.zend2, self.takeoff_height_2)

        self._publish_cmd(self.cmd2_pub, self.cf2_name, self.x2_const, self.y2_const, z_cmd, self.yaw2_const)

    # ----------------------------
    # Freshness / validity checks
    # ----------------------------
    def _inputs_ok_cf231(self, now):
        missing = []

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

    def _inputs_ok_cf232(self, now):
        missing = []

        def is_fresh(tstamp):
            if tstamp is None:
                return False
            age = (now - tstamp).nanoseconds * 1e-9
            return age <= self.input_timeout_sec

        if not is_fresh(self.zend2_t):
            missing.append("/zend2")

        return (len(missing) == 0), missing

    # ----------------------------
    # Logging helpers
    # ----------------------------
    def _log_missing_inputs_throttled(self, period_sec: float, missing: list):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if (now_sec - self._last_missing_log_time_sec) < period_sec:
            return

        now = self.get_clock().now()

        def age_str(name, tstamp):
            if tstamp is None:
                return f"{name}: never"
            age = (now - tstamp).nanoseconds * 1e-9
            return f"{name}: {age:.3f}s"

        ages = [
            age_str("/xend", self.xend_t),
            age_str("/yend", self.yend_t),
            age_str("/z_augment", self.zaug_t),
            age_str("/psiend", self.psiend_t),
            age_str("/zend2", self.zend2_t),
        ]

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

    def _publish_cmd(self, pub, cf_name: str, x: float, y: float, z: float, yaw: float):
        msg = Position()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        msg.yaw = float(yaw)
        pub.publish(msg)


def main():
    rclpy.init()
    node = L1CmdPosBridgeCompTwoDrones()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

