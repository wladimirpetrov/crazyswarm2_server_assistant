#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64, Float64MultiArray
from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import Position


class L1CmdPosBridge(Node):
    """
    Publishes /<cf_name>/cmd_position (Position setpoints).

    Startup behavior:
      - ramp z from z0 -> takeoff_height over takeoff_duration seconds
      - after takeoff, publish /l1_status = 1.0 (keep at 1 Hz)
      - then continuously publish cmd_position at publish_rate_hz.

    Two RUN modes:

    1) use_mpc_interpolation = False
       Original ZOH behavior:
         - listen to /x_input, /y_input, /z_input, /psi_input
         - hold the last committed (x,y,z,psi) at high-rate
         - switch to a new command set ONLY when ALL four topics have been updated
           at least once since the last commit

    2) use_mpc_interpolation = True
       Interpolated behavior:
         - when a new full MPC set is committed,
         - read the drone's actual pose from /<cf_name>/pose
         - interpolate from ACTUAL pose -> NEW target over interp_duration_sec
         - publish interpolated commands at publish_rate_hz
         - if actual pose is not yet available, fall back to current commanded pose
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

        self.declare_parameter("x0", -1.2)
        self.declare_parameter("y0", 1.75)
        self.declare_parameter("yaw0", 0.0)
        self.declare_parameter("z0", 0.0)

        # Used for warning-only staleness reporting (NOT gating).
        self.declare_parameter("input_timeout_sec", 1.0)

        # Debug prints
        self.declare_parameter("debug_inputs", True)
        self.declare_parameter("debug_period_sec", 1.0)

        # New interpolation parameters
        self.declare_parameter("use_mpc_interpolation", True)
        self.declare_parameter("interp_duration_sec", 0.1)

        # Actual pose topic
        self.declare_parameter("pose_topic", "")

        # New: enable/disable Bernstein correction
        self.declare_parameter("enable_bernstein_correction", True)

        # Extraction settings for Bernstein vectors
        self.declare_parameter("extract_ts", 0.05)
        self.declare_parameter("extract_iter_minus_1_x", 10.0) #200 for libbebotv1
        self.declare_parameter("extract_iter_minus_1_y", 10.0) #200 for libbebotv1
        self.declare_parameter("extract_iter_minus_1_z", 200.0) #200 for libbebotv1
        self.declare_parameter("extract_iter_minus_1_psi", 10.0) #200 for libbebotv1

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

        self.use_mpc_interpolation = bool(self.get_parameter("use_mpc_interpolation").value)
        self.interp_duration_sec = float(self.get_parameter("interp_duration_sec").value)

        self.enable_bernstein_correction = bool(self.get_parameter("enable_bernstein_correction").value)

        self.extract_ts = float(self.get_parameter("extract_ts").value)
        self.extract_iter_minus_1_x = float(self.get_parameter("extract_iter_minus_1_x").value)
        self.extract_iter_minus_1_y = float(self.get_parameter("extract_iter_minus_1_y").value)
        self.extract_iter_minus_1_z = float(self.get_parameter("extract_iter_minus_1_z").value)
        self.extract_iter_minus_1_psi = float(self.get_parameter("extract_iter_minus_1_psi").value)

        pose_topic_param = str(self.get_parameter("pose_topic").value).strip()
        if pose_topic_param:
            self.pose_topic = pose_topic_param
        else:
            self.pose_topic = f"/{self.cf_name}/pose"

        if self.interp_duration_sec <= 1e-9:
            self.get_logger().warn("interp_duration_sec must be > 0. Using 0.05 s.")
            self.interp_duration_sec = 0.05

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

        # Inputs from your C++ subscriber_node
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

        # Publish status as BEST_EFFORT
        self.l1_status_pub = self.create_publisher(Float64, "/l1_status", self.qos_be)

        # Inputs from MPC subscriber_node
        self.x_sub = self.create_subscription(Float64, "/x_input", self._x_cb, self.qos_rel)
        self.y_sub = self.create_subscription(Float64, "/y_input", self._y_cb, self.qos_rel)
        self.z_sub = self.create_subscription(Float64, "/z_input", self._z_cb, self.qos_rel)
        self.psi_sub = self.create_subscription(Float64, "/psi_input", self._psi_cb, self.qos_rel)

        # New: Bernstein control-point vectors
        self.x_vector_sub = self.create_subscription(
            Float64MultiArray, "/x_vector", self._x_vector_cb, self.qos_rel
        )
        self.y_vector_sub = self.create_subscription(
            Float64MultiArray, "/y_vector", self._y_vector_cb, self.qos_rel
        )
        self.z_vector_sub = self.create_subscription(
            Float64MultiArray, "/z_vector", self._z_vector_cb, self.qos_rel
        )
        self.psi_vector_sub = self.create_subscription(
            Float64MultiArray, "/psi_vector", self._psi_vector_cb, self.qos_rel
        )

        # Endpoint / augment topics for Bernstein correction
        self.xend_sub = self.create_subscription(Float64, "/x_augment", self._xend_cb, self.qos_rel)
        self.yend_sub = self.create_subscription(Float64, "/y_augment", self._yend_cb, self.qos_rel)
        self.zaug_sub = self.create_subscription(Float64, "/z_augment", self._z_augment_cb, self.qos_rel)
        self.psiaug_sub = self.create_subscription(Float64, "/psi_augment", self._psi_augment_cb, self.qos_rel)

        # Actual pose subscriber
        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self._pose_cb,
            self.qos_be,
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

        # Bernstein vector storage
        self._x_vec = None
        self._y_vec = None
        self._z_vec = None
        self._psi_vec = None

        self._x_vec_t = None
        self._y_vec_t = None
        self._z_vec_t = None
        self._psi_vec_t = None

        self._x_vec_seq = 0
        self._y_vec_seq = 0
        self._z_vec_seq = 0
        self._psi_vec_seq = 0

        self._committed_vec_seq = (0, 0, 0, 0)

        # Corrected vectors
        self._x_vec_corr = None
        self._y_vec_corr = None
        self._z_vec_corr = None
        self._psi_vec_corr = None

        # Endpoint / augment storage
        self._xend = None
        self._yend = None
        self._zend = None
        self._psiend = None

        self._xend_t = None
        self._yend_t = None
        self._zend_t = None
        self._psiend_t = None

        # Current target command set
        self._have_committed_set = False
        self._cmd_x = self.x0
        self._cmd_y = self.y0
        self._cmd_z = self.takeoff_height
        self._cmd_psi = self.yaw0

        # Latest actual pose
        self._act_x = None
        self._act_y = None
        self._act_z = None
        self._act_psi = None
        self._act_t = None

        # Interpolation state
        self._interp_active = False
        self._interp_t0 = None

        self._interp_start_x = self.x0
        self._interp_start_y = self.y0
        self._interp_start_z = self.z0
        self._interp_start_psi = self.yaw0

        self._interp_goal_x = self.x0
        self._interp_goal_y = self.y0
        self._interp_goal_z = self.takeoff_height
        self._interp_goal_psi = self.yaw0

        # Last actually published command (for fallback)
        self._last_pub_x = self.x0
        self._last_pub_y = self.y0
        self._last_pub_z = self.z0
        self._last_pub_psi = self.yaw0

        # Debug / throttles
        self._last_missing_log_time_sec = -1e9
        self._last_debug_log_time_sec = -1e9
        self._last_stale_warn_time_sec = -1e9
        self._last_interp_log_time_sec = -1e9

        # State machine
        self.state = "TAKEOFF"  # TAKEOFF -> RUN
        self.t0 = self.get_clock().now()

        # Timing
        self.dt = 1.0 / max(self.publish_rate_hz, 1e-6)
        self.cmd_timer = self.create_timer(self.dt, self._cmd_timer_cb)
        self.status_timer = self.create_timer(1.0, self._status_timer_cb)

        self.get_logger().info(f"Node started. Publishing takeoff ramp to {self.cmd_topic}")
        self.get_logger().info("Subscribing to /x_input /y_input /z_input /psi_input")
        self.get_logger().info(f"Subscribing to actual pose on {self.pose_topic}")
        self.get_logger().info(
            f"Bernstein correction enabled={self.enable_bernstein_correction}"
        )
        if self.enable_bernstein_correction:
            self.get_logger().info("Subscribing to /x_vector /y_vector /z_vector /psi_vector")
            self.get_logger().info("Subscribing to /xend /yend /z_augment /psi_augment")

        if self.use_mpc_interpolation:
            self.get_logger().info(
                f"Interpolation mode ON. Interpolating actual pose -> MPC target over "
                f"{self.interp_duration_sec:.3f} s at {self.publish_rate_hz:.1f} Hz."
            )
        else:
            self.get_logger().info(
                f"ZOH mode ON. Publishing at {self.publish_rate_hz:.1f} Hz; "
                "switch to a new set only when all 4 topics updated since last switch."
            )

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

    def _x_vector_cb(self, msg: Float64MultiArray):
        self._x_vec = [float(v) for v in msg.data]
        self._x_vec_t = self.get_clock().now()
        self._x_vec_seq += 1

    def _y_vector_cb(self, msg: Float64MultiArray):
        self._y_vec = [float(v) for v in msg.data]
        self._y_vec_t = self.get_clock().now()
        self._y_vec_seq += 1

    def _z_vector_cb(self, msg: Float64MultiArray):
        self._z_vec = [float(v) for v in msg.data]
        self._z_vec_t = self.get_clock().now()
        self._z_vec_seq += 1

    def _psi_vector_cb(self, msg: Float64MultiArray):
        self._psi_vec = [float(v) for v in msg.data]
        self._psi_vec_t = self.get_clock().now()
        self._psi_vec_seq += 1

    def _xend_cb(self, msg: Float64):
        self._xend = float(msg.data)
        self._xend_t = self.get_clock().now()

    def _yend_cb(self, msg: Float64):
        self._yend = float(msg.data)
        self._yend_t = self.get_clock().now()

    def _z_augment_cb(self, msg: Float64):
        self._zend = float(msg.data)
        self._zend_t = self.get_clock().now()

    def _psi_augment_cb(self, msg: Float64):
        self._psiend = float(msg.data)
        self._psiend_t = self.get_clock().now()

    def _pose_cb(self, msg: PoseStamped):
        self._act_x = float(msg.pose.position.x)
        self._act_y = float(msg.pose.position.y)
        self._act_z = float(msg.pose.position.z)

        qx = float(msg.pose.orientation.x)
        qy = float(msg.pose.orientation.y)
        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)

        self._act_psi = self._yaw_from_quaternion(qx, qy, qz, qw)
        self._act_t = self.get_clock().now()

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
                if self.use_mpc_interpolation:
                    self.get_logger().info(
                        "Takeoff complete. Publishing /l1_status = 1 and forwarding interpolated commands."
                    )
                else:
                    self.get_logger().info(
                        "Takeoff complete. Publishing /l1_status = 1 and forwarding ZOH commands."
                    )
            return

        # RUN mode:
        # 1) Try to commit a new (x,y,z,psi) set if ALL four updated since last commit
        self._try_commit_new_set(now)

        # 2) If we don't have any committed set yet, hover safely
        if not self._have_committed_set:
            self._log_missing_inputs_throttled(2.0)
            self._publish_cmd(self.x0, self.y0, self.takeoff_height, self.yaw0)
            return

        # 3) Optionally warn if inputs are very old (but do NOT stop)
        self._warn_if_stale_throttled(now, period_sec=2.0)

        # 4) Publish according to selected mode
        if self.use_mpc_interpolation:
            x_cmd, y_cmd, z_cmd, psi_cmd = self._get_interpolated_cmd(now)
            self._publish_cmd(x_cmd, y_cmd, z_cmd, psi_cmd)
        else:
            self._publish_cmd(self._cmd_x, self._cmd_y, self._cmd_z, self._cmd_psi)

    # ----------------------------
    # Commit logic
    # ----------------------------
    def _try_commit_new_set(self, now):
        if self.enable_bernstein_correction:
            if self._x_vec is None or self._y_vec is None or self._z_vec is None or self._psi_vec is None:
                return

            if self._act_x is None or self._act_y is None or self._act_z is None or self._act_psi is None:
                return

            if self._xend is None or self._yend is None or self._zend is None or self._psiend is None:
                return

            cur_vec_seq = (self._x_vec_seq, self._y_vec_seq, self._z_vec_seq, self._psi_vec_seq)

            if (
                cur_vec_seq[0] <= self._committed_vec_seq[0]
                or cur_vec_seq[1] <= self._committed_vec_seq[1]
                or cur_vec_seq[2] <= self._committed_vec_seq[2]
                or cur_vec_seq[3] <= self._committed_vec_seq[3]
            ):
                return

            npx = len(self._x_vec)
            npy = len(self._y_vec)
            npz = len(self._z_vec)
            npp = len(self._psi_vec)

            if npx == 0 or npy == 0 or npz == 0 or npp == 0:
                self.get_logger().warn("[WARP] One or more Bernstein vectors are empty; skipping commit.")
                return
            if not (npx == npy == npz == npp):
                self.get_logger().warn(
                    f"[WARP] Vector lengths mismatch: len(x)={npx} len(y)={npy} len(z)={npz} len(psi)={npp}; skipping commit."
                )
                return

            x_vec = list(self._x_vec)
            y_vec = list(self._y_vec)
            z_vec = list(self._z_vec)
            psi_vec = list(self._psi_vec)

            self._x_vec_corr = self._warp_control_points(x_vec, self._act_x, self._xend, wrap_angles=False)
            self._y_vec_corr = self._warp_control_points(y_vec, self._act_y, self._yend, wrap_angles=False)
            self._z_vec_corr = self._warp_control_points(z_vec, self._act_z, self._zend, wrap_angles=False)
            self._psi_vec_corr = self._warp_control_points(psi_vec, self._act_psi, self._psiend, wrap_angles=True)

            ts = self.extract_ts

            t_sample_x = self.extract_iter_minus_1_x * ts
            denom_tau_x = t_sample_x + 1.0
            if abs(denom_tau_x) > 1e-12:
                tau_sample_x = (t_sample_x - 1.0) / denom_tau_x
            else:
                tau_sample_x = -1.0

            t_sample_y = self.extract_iter_minus_1_y * ts
            denom_tau_y = t_sample_y + 1.0
            if abs(denom_tau_y) > 1e-12:
                tau_sample_y = (t_sample_y - 1.0) / denom_tau_y
            else:
                tau_sample_y = -1.0

            t_sample_z = self.extract_iter_minus_1_z * ts
            denom_tau_z = t_sample_z + 1.0
            if abs(denom_tau_z) > 1e-12:
                tau_sample_z = (t_sample_z - 1.0) / denom_tau_z
            else:
                tau_sample_z = -1.0

            t_sample_psi = self.extract_iter_minus_1_psi * ts
            denom_tau_psi = t_sample_psi + 1.0
            if abs(denom_tau_psi) > 1e-12:
                tau_sample_psi = (t_sample_psi - 1.0) / denom_tau_psi
            else:
                tau_sample_psi = -1.0

            new_x = self._bernstein_eval_tau(self._x_vec_corr, tau_sample_x, -1.0, 1.0)
            new_y = self._bernstein_eval_tau(self._y_vec_corr, tau_sample_y, -1.0, 1.0)
            new_z = self._bernstein_eval_tau(self._z_vec_corr, tau_sample_z, -1.0, 1.0)
            new_psi = self._bernstein_eval_tau(self._psi_vec_corr, tau_sample_psi, -1.0, 1.0)

            self._committed_vec_seq = cur_vec_seq

            if self.debug_inputs:
                self.get_logger().info(
                    f"[COMMIT+BP] corrected target: "
                    f"x={new_x:.4f} y={new_y:.4f} z={new_z:.4f} psi={new_psi:.4f} | "
                    f"tau_x={tau_sample_x:.4f} tau_y={tau_sample_y:.4f} "
                    f"tau_z={tau_sample_z:.4f} tau_psi={tau_sample_psi:.4f} | "
                    f"vec_seq={self._committed_vec_seq}"
                )
        else:
            # Need at least one message from each topic
            if self._x_val is None or self._y_val is None or self._z_val is None or self._psi_val is None:
                return

            cur_seq = (self._x_seq, self._y_seq, self._z_seq, self._psi_seq)

            # Commit new set only if ALL four sequences advanced since last commit
            if not (
                cur_seq[0] > self._committed_seq[0]
                and cur_seq[1] > self._committed_seq[1]
                and cur_seq[2] > self._committed_seq[2]
                and cur_seq[3] > self._committed_seq[3]
            ):
                return

            new_x = self._finite_or(self._x_val, self._cmd_x)
            new_y = self._finite_or(self._y_val, self._cmd_y)
            new_z = self._finite_or(self._z_val, self._cmd_z)
            new_psi = self._finite_or(self._psi_val, self._cmd_psi)

        # Update target set
        self._cmd_x = new_x
        self._cmd_y = new_y
        self._cmd_z = new_z
        self._cmd_psi = new_psi

        self._committed_seq = (self._x_seq, self._y_seq, self._z_seq, self._psi_seq)
        if not self._have_committed_set:
            self._have_committed_set = True

        # If interpolation is enabled, snapshot start pose now
        if self.use_mpc_interpolation:
            if (
                self._act_x is not None
                and self._act_y is not None
                and self._act_z is not None
                and self._act_psi is not None
            ):
                self._interp_start_x = self._act_x
                self._interp_start_y = self._act_y
                self._interp_start_z = self._act_z
                self._interp_start_psi = self._act_psi
            else:
                # Fallback if actual pose not available yet
                self._interp_start_x = self._last_pub_x
                self._interp_start_y = self._last_pub_y
                self._interp_start_z = self._last_pub_z
                self._interp_start_psi = self._last_pub_psi
                self.get_logger().warn(
                    "Interpolation requested but actual pose not available yet. "
                    "Using last published command as interpolation start."
                )

            self._interp_goal_x = new_x
            self._interp_goal_y = new_y
            self._interp_goal_z = new_z
            self._interp_goal_psi = new_psi

            self._interp_t0 = now
            self._interp_active = True

            self.get_logger().info(
                f"[COMMIT+INTERP] start=({self._interp_start_x:.4f}, {self._interp_start_y:.4f}, "
                f"{self._interp_start_z:.4f}, {self._interp_start_psi:.4f}) -> "
                f"goal=({self._interp_goal_x:.4f}, {self._interp_goal_y:.4f}, "
                f"{self._interp_goal_z:.4f}, {self._interp_goal_psi:.4f}) "
                f"over {self.interp_duration_sec:.3f}s"
            )
        else:
            self.get_logger().info(
                f"[COMMIT] New set committed: x={self._cmd_x:.4f} y={self._cmd_y:.4f} "
                f"z={self._cmd_z:.4f} psi={self._cmd_psi:.4f}"
            )

    # ----------------------------
    # Interpolation
    # ----------------------------
    def _get_interpolated_cmd(self, now):
        if not self._interp_active or self._interp_t0 is None:
            return self._cmd_x, self._cmd_y, self._cmd_z, self._cmd_psi

        dt = (now - self._interp_t0).nanoseconds * 1e-9

        if dt <= 0.0:
            alpha = 0.0
        else:
            alpha = dt / self.interp_duration_sec

        alpha = max(0.0, min(1.0, alpha))

        x = self._lerp(self._interp_start_x, self._interp_goal_x, alpha)
        y = self._lerp(self._interp_start_y, self._interp_goal_y, alpha)
        z = self._lerp(self._interp_start_z, self._interp_goal_z, alpha)
        psi = self._interp_angle(self._interp_start_psi, self._interp_goal_psi, alpha)

        now_sec = now.nanoseconds * 1e-9
        if (now_sec - self._last_interp_log_time_sec) >= 0.5:
            self.get_logger().info(
                f"[INTERP] alpha={alpha:.3f} "
                f"x={x:.4f} y={y:.4f} z={z:.4f} psi={psi:.4f}"
            )
            self._last_interp_log_time_sec = now_sec

        if alpha >= 1.0:
            self._interp_active = False
            return self._interp_goal_x, self._interp_goal_y, self._interp_goal_z, self._interp_goal_psi

        return x, y, z, psi

    # ----------------------------
    # Logging helpers
    # ----------------------------
    def _log_missing_inputs_throttled(self, period_sec: float):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if (now_sec - self._last_missing_log_time_sec) < period_sec:
            return

        missing = []
        if self.enable_bernstein_correction:
            if self._x_vec is None:
                missing.append("/x_vector")
            if self._y_vec is None:
                missing.append("/y_vector")
            if self._z_vec is None:
                missing.append("/z_vector")
            if self._psi_vec is None:
                missing.append("/psi_vector")
            if self._xend is None:
                missing.append("/xend")
            if self._yend is None:
                missing.append("/yend")
            if self._zend is None:
                missing.append("/z_augment")
            if self._psiend is None:
                missing.append("/psi_augment")
            if self._act_x is None or self._act_y is None or self._act_z is None or self._act_psi is None:
                missing.append(self.pose_topic)
        else:
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

        if self.enable_bernstein_correction:
            ages = {
                "/x_vector": age(self._x_vec_t),
                "/y_vector": age(self._y_vec_t),
                "/z_vector": age(self._z_vec_t),
                "/psi_vector": age(self._psi_vec_t),
                "/xend": age(self._xend_t),
                "/yend": age(self._yend_t),
                "/z_augment": age(self._zend_t),
                "/psi_augment": age(self._psiend_t),
                self.pose_topic: age(self._act_t),
            }
        else:
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
                f"(but continuing): {', '.join(parts)}"
            )
            self._last_stale_warn_time_sec = now_sec

    # ----------------------------
    # Bernstein helpers
    # ----------------------------
    @staticmethod
    def _binom_coeffs(n: int):
        coeffs = [0.0] * (n + 1)
        coeffs[0] = 1.0
        for k in range(1, n + 1):
            coeffs[k] = coeffs[k - 1] * float(n - k + 1) / float(k)
        return coeffs

    @classmethod
    def _bernstein_eval_tau(cls, coeffs, t, t0=0.0, tf=1.0):
        n = len(coeffs) - 1
        if n < 0:
            return 0.0

        denom = tf - t0
        if abs(denom) < 1e-12:
            return float(coeffs[0])

        s = (t - t0) / denom
        s = max(0.0, min(1.0, s))
        one_minus = 1.0 - s

        C = cls._binom_coeffs(n)

        val = 0.0
        for i in range(n + 1):
            term = float(coeffs[i]) * C[i]
            term *= (s ** i)
            term *= (one_minus ** (n - i))
            val += term
        return float(val)

    def _warp_control_points(self, coeffs, new_start, new_end, wrap_angles=False):
        n = len(coeffs) - 1
        if n < 0:
            return []
        if n == 0:
            val = float(new_start)
            return [self._wrap_angle(val)] if wrap_angles else [val]

        old_start = float(coeffs[0])
        old_end = float(coeffs[-1])

        if wrap_angles:
            e0 = self._wrap_angle(float(new_start) - old_start)
            eN = self._wrap_angle(float(new_end) - old_end)
        else:
            e0 = float(new_start) - old_start
            eN = float(new_end) - old_end

        corrected = []
        for i, p in enumerate(coeffs):
            alpha0 = 1.0 - (float(i) / float(n))
            alphaN = float(i) / float(n)
            p_new = float(p) + alpha0 * e0 + alphaN * eN
            if wrap_angles:
                p_new = self._wrap_angle(p_new)
            corrected.append(p_new)

        if wrap_angles:
            corrected[0] = self._wrap_angle(float(new_start))
            corrected[-1] = self._wrap_angle(float(new_end))
        else:
            corrected[0] = float(new_start)
            corrected[-1] = float(new_end)

        return corrected

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
    def _lerp(a: float, b: float, alpha: float) -> float:
        return a + alpha * (b - a)

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    def _interp_angle(self, a0: float, a1: float, alpha: float) -> float:
        d = self._wrap_angle(a1 - a0)
        return self._wrap_angle(a0 + alpha * d)

    @staticmethod
    def _yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def _publish_cmd(self, x: float, y: float, z: float, yaw: float):
        msg = Position()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        msg.yaw = float(yaw)
        self.cmd_pub.publish(msg)

        self._last_pub_x = float(x)
        self._last_pub_y = float(y)
        self._last_pub_z = float(z)
        self._last_pub_psi = float(yaw)


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
