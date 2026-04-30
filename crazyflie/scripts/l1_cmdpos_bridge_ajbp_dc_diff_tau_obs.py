#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float64, Float64MultiArray, Empty
from geometry_msgs.msg import PoseStamped
from crazyflie_interfaces.msg import Position


class L1CmdPosBridge(Node):
    """
    Publishes /<cf_name>/cmd_position.

    RUN behavior:

    Mode 1: execution_mode = "segment"
      - receive /x_vector and /y_vector Bernstein control points
      - receive actual pose from /<cf_name>/pose
      - receive endpoint corrections from /x_augment and /y_augment
      - warp x/y control points so:
            first control point = actual position
            last control point  = augmented endpoint
      - cut the first segment_lambda_percent part of the reshaped Bernstein trajectory
      - sample segment_num_points points from that segment using de Casteljau
      - store those points in a buffer
      - publish them sequentially at publish_rate_hz

    Mode 2: execution_mode = "interpolation"
      - receive /x_vector and /y_vector Bernstein control points
      - receive actual pose from /<cf_name>/pose
      - receive endpoint corrections from /x_augment and /y_augment
      - warp x/y control points
      - extract ONE x/y point from the Bernstein trajectory
      - interpolate from current actual drone position to that extracted point

    z and psi are fixed:
      z   = fixed_z_cmd
      psi = fixed_psi_cmd

    If block_xy_updates = True:
      - ignore trajectory x/y commands
      - always command x0, y0
      - still compute and print extracted Bernstein x/y point for debugging
      - still use fixed_z_cmd and fixed_psi_cmd
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

        self.declare_parameter("fixed_z_cmd", 0.6)
        self.declare_parameter("fixed_psi_cmd", 0.0)

        self.declare_parameter("input_timeout_sec", 1.0)

        self.declare_parameter("debug_inputs", True)
        self.declare_parameter("debug_period_sec", 1.0)

        self.declare_parameter("pose_topic", "")

        # Main behavior switches
        # "segment"       = de Casteljau segment buffer
        # "interpolation" = extract one Bernstein point and interpolate to it
        self.declare_parameter("execution_mode", "interpolation")

        # Runtime switch:
        # ros2 param set /l1_cmdpos_bridge block_xy_updates true
        # ros2 param set /l1_cmdpos_bridge block_xy_updates false
        self.declare_parameter("block_xy_updates", False)

        # Enable/disable using Bernstein vectors
        self.declare_parameter("enable_bernstein_correction", True)

        # Enable/disable control-point trajectory projection / warping
        self.declare_parameter("adjust_control_points", False)

        # Segment mode parameters
        self.declare_parameter("segment_num_points", 20)

        # Direct percentage of trajectory to extract.
        # 0.10 means first 10% of the reshaped Bernstein trajectory.
        self.declare_parameter("segment_lambda_percent", 0.10)

        # Interpolation mode parameters
        self.declare_parameter("interp_duration_sec", 0.03) #0.1

        # Extraction settings for one-point Bernstein evaluation
        self.declare_parameter("extract_ts", 0.05)
        self.declare_parameter("extract_iter_minus_1_x", 27.0)
        self.declare_parameter("extract_iter_minus_1_y", 27.0)

        self.cf_name = str(self.get_parameter("cf_name").value)
        self.takeoff_height = float(self.get_parameter("takeoff_height").value)
        self.takeoff_duration = float(self.get_parameter("takeoff_duration").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.x0 = float(self.get_parameter("x0").value)
        self.y0 = float(self.get_parameter("y0").value)
        self.yaw0 = float(self.get_parameter("yaw0").value)
        self.z0 = float(self.get_parameter("z0").value)

        self.fixed_z_cmd = float(self.get_parameter("fixed_z_cmd").value)
        self.fixed_psi_cmd = float(self.get_parameter("fixed_psi_cmd").value)

        self.input_timeout_sec = float(self.get_parameter("input_timeout_sec").value)
        self.debug_inputs = bool(self.get_parameter("debug_inputs").value)
        self.debug_period_sec = float(self.get_parameter("debug_period_sec").value)

        self.execution_mode = str(self.get_parameter("execution_mode").value).strip().lower()
        if self.execution_mode not in ("segment", "interpolation"):
            self.get_logger().warn(
                f"Unknown execution_mode='{self.execution_mode}'. Using 'interpolation'."
            )
            self.execution_mode = "interpolation"

        self.block_xy_updates = bool(self.get_parameter("block_xy_updates").value)

        self.enable_bernstein_correction = bool(
            self.get_parameter("enable_bernstein_correction").value
        )

        self.adjust_control_points = bool(
            self.get_parameter("adjust_control_points").value
        )

        self.segment_num_points = int(self.get_parameter("segment_num_points").value)
        if self.segment_num_points < 2:
            self.get_logger().warn("segment_num_points must be >= 2. Using 10.")
            self.segment_num_points = 10

        self.segment_lambda_percent = float(
            self.get_parameter("segment_lambda_percent").value
        )
        self.segment_lambda_percent = max(0.0, min(1.0, self.segment_lambda_percent))

        self.interp_duration_sec = float(self.get_parameter("interp_duration_sec").value)
        if self.interp_duration_sec <= 1e-9:
            self.get_logger().warn("interp_duration_sec must be > 0. Using 0.05 s.")
            self.interp_duration_sec = 0.05

        self.extract_ts = float(self.get_parameter("extract_ts").value)
        self.extract_iter_minus_1_x = float(
            self.get_parameter("extract_iter_minus_1_x").value
        )
        self.extract_iter_minus_1_y = float(
            self.get_parameter("extract_iter_minus_1_y").value
        )

        pose_topic_param = str(self.get_parameter("pose_topic").value).strip()
        if pose_topic_param:
            self.pose_topic = pose_topic_param
        else:
            self.pose_topic = f"/{self.cf_name}/pose"

        # Runtime parameter callback.
        self.add_on_set_parameters_callback(self._on_parameter_update)

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

        self.x_vector_sub = self.create_subscription(
            Float64MultiArray, "/x_vector", self._x_vector_cb, self.qos_rel
        )
        self.y_vector_sub = self.create_subscription(
            Float64MultiArray, "/y_vector", self._y_vector_cb, self.qos_rel
        )

        self.xend_sub = self.create_subscription(
            Float64, "/x_augment", self._xend_cb, self.qos_rel
        )
        self.yend_sub = self.create_subscription(
            Float64, "/y_augment", self._yend_cb, self.qos_rel
        )

        # Button topic:
        # Every Empty message on /increase_extract_iter increases both
        # extract_iter_minus_1_x and extract_iter_minus_1_y by +1.
        self.increase_extract_iter_sub = self.create_subscription(
            Empty,
            "/increase_extract_iter",
            self._increase_extract_iter_cb,
            self.qos_rel,
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self._pose_cb,
            self.qos_be,
        )

        # ----------------------------
        # Storage
        # ----------------------------
        self._x_vec = None
        self._y_vec = None

        self._x_vec_t = None
        self._y_vec_t = None

        self._x_vec_seq = 0
        self._y_vec_seq = 0
        self._committed_vec_seq = (0, 0)

        self._xend = None
        self._yend = None
        self._xend_t = None
        self._yend_t = None

        self._act_x = None
        self._act_y = None
        self._act_z = None
        self._act_psi = None
        self._act_t = None

        self._have_committed_set = False

        self._cmd_x = self.x0
        self._cmd_y = self.y0
        self._cmd_z = self.fixed_z_cmd
        self._cmd_psi = self.fixed_psi_cmd

        # Buffered sampled command segment
        self._cmd_buffer = []
        self._cmd_buffer_index = 0

        # Interpolation state
        self._interp_active = False
        self._interp_t0 = None

        self._interp_start_x = self.x0
        self._interp_start_y = self.y0
        self._interp_start_z = self.z0
        self._interp_start_psi = self.yaw0

        self._interp_goal_x = self.x0
        self._interp_goal_y = self.y0
        self._interp_goal_z = self.fixed_z_cmd
        self._interp_goal_psi = self.fixed_psi_cmd

        # Last actually published command
        self._last_pub_x = self.x0
        self._last_pub_y = self.y0
        self._last_pub_z = self.z0
        self._last_pub_psi = self.yaw0

        # Debug / throttles
        self._last_missing_log_time_sec = -1e9
        self._last_debug_log_time_sec = -1e9
        self._last_stale_warn_time_sec = -1e9
        self._last_interp_log_time_sec = -1e9
        self._last_publish_log_time_sec = -1e9

        # Debug print throttling for extracted point in block_xy mode
        self._last_block_extract_log_time_sec = -1e9
        self._last_block_extract_seq = (0, 0)

        self.state = "TAKEOFF"
        self.t0 = self.get_clock().now()

        self.dt = 1.0 / max(self.publish_rate_hz, 1e-6)
        self.cmd_timer = self.create_timer(self.dt, self._cmd_timer_cb)
        self.status_timer = self.create_timer(1.0, self._status_timer_cb)

        self.get_logger().info(f"Node started. Publishing to {self.cmd_topic}")
        self.get_logger().info(f"Subscribing to actual pose on {self.pose_topic}")
        self.get_logger().info("Subscribing to /x_vector /y_vector")
        self.get_logger().info("Subscribing to /x_augment /y_augment")
        self.get_logger().info("Subscribing to /increase_extract_iter")
        self.get_logger().info(f"execution_mode={self.execution_mode}")
        self.get_logger().info(f"block_xy_updates={self.block_xy_updates}")
        self.get_logger().info(f"Bernstein correction enabled={self.enable_bernstein_correction}")
        self.get_logger().info(f"Control-point adjustment enabled={self.adjust_control_points}")
        self.get_logger().info(
            f"Segment mode config: {self.segment_num_points} points, "
            f"first {100.0 * self.segment_lambda_percent:.2f}% of trajectory, "
            f"publish_rate={self.publish_rate_hz:.1f} Hz."
        )
        self.get_logger().info(
            f"Interpolation mode config: interp_duration_sec={self.interp_duration_sec:.3f}, "
            f"extract_iter_minus_1_x={self.extract_iter_minus_1_x:.3f}, "
            f"extract_iter_minus_1_y={self.extract_iter_minus_1_y:.3f}, "
            f"extract_ts={self.extract_ts:.3f}"
        )
        self.get_logger().info(
            f"Fixed z={self.fixed_z_cmd:.4f}, fixed psi={self.fixed_psi_cmd:.4f}"
        )

    # ----------------------------
    # Runtime parameter update callback
    # ----------------------------
    def _on_parameter_update(self, params):
        for param in params:
            if param.name == "block_xy_updates":
                if param.type_ != Parameter.Type.BOOL:
                    return SetParametersResult(
                        successful=False,
                        reason="block_xy_updates must be a boolean: true or false",
                    )

                old_value = self.block_xy_updates
                new_value = bool(param.value)
                self.block_xy_updates = new_value

                if new_value:
                    # When blocking is turned ON, immediately stop any active interpolation
                    # or buffered trajectory so the timer publishes fixed x0/y0.
                    self._interp_active = False
                    self._interp_t0 = None
                    self._cmd_buffer = []
                    self._cmd_buffer_index = 0

                    self._cmd_x = self.x0
                    self._cmd_y = self.y0
                    self._cmd_z = self.fixed_z_cmd
                    self._cmd_psi = self.fixed_psi_cmd
                else:
                    # When blocking is turned OFF, force the node to recommit the latest
                    # available x/y vectors instead of continuing an old stale command.
                    self._interp_active = False
                    self._interp_t0 = None
                    self._cmd_buffer = []
                    self._cmd_buffer_index = 0
                    self._have_committed_set = False
                    self._committed_vec_seq = (0, 0)

                self.get_logger().warn(
                    f"[PARAM UPDATE] block_xy_updates changed: "
                    f"{old_value} -> {new_value}"
                )

            elif param.name == "extract_iter_minus_1_x":
                if param.type_ not in (Parameter.Type.DOUBLE, Parameter.Type.INTEGER):
                    return SetParametersResult(
                        successful=False,
                        reason="extract_iter_minus_1_x must be numeric",
                    )

                old_value = self.extract_iter_minus_1_x
                self.extract_iter_minus_1_x = float(param.value)

                # Force recommit using latest available vector and new extraction index.
                self._interp_active = False
                self._interp_t0 = None
                self._cmd_buffer = []
                self._cmd_buffer_index = 0
                self._have_committed_set = False
                self._committed_vec_seq = (0, 0)

                self.get_logger().warn(
                    f"[PARAM UPDATE] extract_iter_minus_1_x changed: "
                    f"{old_value:.3f} -> {self.extract_iter_minus_1_x:.3f}"
                )

            elif param.name == "extract_iter_minus_1_y":
                if param.type_ not in (Parameter.Type.DOUBLE, Parameter.Type.INTEGER):
                    return SetParametersResult(
                        successful=False,
                        reason="extract_iter_minus_1_y must be numeric",
                    )

                old_value = self.extract_iter_minus_1_y
                self.extract_iter_minus_1_y = float(param.value)

                # Force recommit using latest available vector and new extraction index.
                self._interp_active = False
                self._interp_t0 = None
                self._cmd_buffer = []
                self._cmd_buffer_index = 0
                self._have_committed_set = False
                self._committed_vec_seq = (0, 0)

                self.get_logger().warn(
                    f"[PARAM UPDATE] extract_iter_minus_1_y changed: "
                    f"{old_value:.3f} -> {self.extract_iter_minus_1_y:.3f}"
                )

        return SetParametersResult(successful=True)

    # ----------------------------
    # Button callback
    # ----------------------------
    def _increase_extract_iter_cb(self, msg: Empty):
        new_x = self.extract_iter_minus_1_x + 1.0
        new_y = self.extract_iter_minus_1_y + 1.0

        results = self.set_parameters(
            [
                Parameter("extract_iter_minus_1_x", Parameter.Type.DOUBLE, new_x),
                Parameter("extract_iter_minus_1_y", Parameter.Type.DOUBLE, new_y),
            ]
        )

        all_successful = all(result.successful for result in results)

        if not all_successful:
            reasons = [result.reason for result in results if not result.successful]
            self.get_logger().warn(
                f"[BUTTON] Failed to increase extract_iter values: {reasons}"
            )
            return

        tau_x = self._tau_from_extract_iter(self.extract_iter_minus_1_x)
        tau_y = self._tau_from_extract_iter(self.extract_iter_minus_1_y)

        self.get_logger().warn(
            f"[BUTTON] Increased extract_iter values by +1 | "
            f"extract_iter_minus_1_x={self.extract_iter_minus_1_x:.3f}, "
            f"extract_iter_minus_1_y={self.extract_iter_minus_1_y:.3f}, "
            f"tau_x={tau_x:.6f}, tau_y={tau_y:.6f}"
        )

    # ----------------------------
    # Callbacks
    # ----------------------------
    def _x_vector_cb(self, msg: Float64MultiArray):
        self._x_vec = [float(v) for v in msg.data]
        self._x_vec_t = self.get_clock().now()
        self._x_vec_seq += 1

    def _y_vector_cb(self, msg: Float64MultiArray):
        self._y_vec = [float(v) for v in msg.data]
        self._y_vec_t = self.get_clock().now()
        self._y_vec_seq += 1

    def _xend_cb(self, msg: Float64):
        self._xend = float(msg.data)
        self._xend_t = self.get_clock().now()

    def _yend_cb(self, msg: Float64):
        self._yend = float(msg.data)
        self._yend_t = self.get_clock().now()

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
    # Status publisher
    # ----------------------------
    def _status_timer_cb(self):
        if self.state == "RUN":
            msg = Float64()
            msg.data = 1.0
            self.l1_status_pub.publish(msg)

    # ----------------------------
    # Main timer
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
            self._publish_cmd(self.x0, self.y0, z_cmd, self.yaw0, source="TAKEOFF")

            if alpha >= 1.0:
                self.state = "RUN"
                self.get_logger().info(
                    "Takeoff complete. RUN mode started."
                )
            return

        # If blocking xy updates, ignore all trajectory data and keep x0/y0,
        # but still compute and print the extracted Bernstein point for debugging.
        if self.block_xy_updates:
            self._print_extracted_point_debug(now)

            self._have_committed_set = True
            self._cmd_x = self.x0
            self._cmd_y = self.y0
            self._cmd_z = self.fixed_z_cmd
            self._cmd_psi = self.fixed_psi_cmd
            self._publish_cmd(
                self.x0,
                self.y0,
                self.fixed_z_cmd,
                self.fixed_psi_cmd,
                source="BLOCK_XY",
            )
            return

        self._try_commit_new_command(now)

        if not self._have_committed_set:
            self._log_missing_inputs_throttled(2.0)
            self._publish_cmd(
                self.x0,
                self.y0,
                self.fixed_z_cmd,
                self.fixed_psi_cmd,
                source="WAITING",
            )
            return

        self._warn_if_stale_throttled(now, period_sec=2.0)

        if self.execution_mode == "segment":
            x_cmd, y_cmd, z_cmd, psi_cmd = self._get_next_buffered_cmd()
            self._publish_cmd(x_cmd, y_cmd, z_cmd, psi_cmd, source="SEGMENT")
        elif self.execution_mode == "interpolation":
            x_cmd, y_cmd, z_cmd, psi_cmd = self._get_interpolated_cmd(now)
            self._publish_cmd(x_cmd, y_cmd, z_cmd, psi_cmd, source="INTERP")
        else:
            self._publish_cmd(
                self._cmd_x,
                self._cmd_y,
                self.fixed_z_cmd,
                self.fixed_psi_cmd,
                source="FALLBACK",
            )

    # ----------------------------
    # Commit new command logic
    # ----------------------------
    def _try_commit_new_command(self, now):
        if not self.enable_bernstein_correction:
            return

        if self._x_vec is None or self._y_vec is None:
            return

        if self.adjust_control_points:
            if self._act_x is None or self._act_y is None:
                return

            if self._xend is None or self._yend is None:
                return

        cur_vec_seq = (self._x_vec_seq, self._y_vec_seq)

        if (
            cur_vec_seq[0] <= self._committed_vec_seq[0]
            or cur_vec_seq[1] <= self._committed_vec_seq[1]
        ):
            return

        npx = len(self._x_vec)
        npy = len(self._y_vec)

        if npx == 0 or npy == 0:
            self.get_logger().warn("[COMMIT] x/y Bernstein vectors are empty. Skipping commit.")
            return

        if npx != npy:
            self.get_logger().warn(
                f"[COMMIT] Vector length mismatch: len(x)={npx}, len(y)={npy}. Skipping commit."
            )
            return

        x_vec = list(self._x_vec)
        y_vec = list(self._y_vec)

        if self.adjust_control_points:
            x_work = self._warp_control_points(
                x_vec,
                new_start=self._act_x,
                new_end=self._xend,
                wrap_angles=False,
            )
            y_work = self._warp_control_points(
                y_vec,
                new_start=self._act_y,
                new_end=self._yend,
                wrap_angles=False,
            )
            projection_mode = "PROJECTED"
        else:
            x_work = x_vec
            y_work = y_vec
            projection_mode = "RAW"

        if self.execution_mode == "segment":
            self._commit_segment_mode(
                x_work=x_work,
                y_work=y_work,
                cur_vec_seq=cur_vec_seq,
                projection_mode=projection_mode,
            )
        elif self.execution_mode == "interpolation":
            self._commit_interpolation_mode(
                x_work=x_work,
                y_work=y_work,
                cur_vec_seq=cur_vec_seq,
                projection_mode=projection_mode,
                now=now,
            )

    def _commit_segment_mode(self, x_work, y_work, cur_vec_seq, projection_mode):
        lambda_end_x = self.segment_lambda_percent
        lambda_end_y = self.segment_lambda_percent

        x_points = self._sample_first_bernstein_segment(
            x_work,
            lambda_end_x,
            self.segment_num_points,
        )
        y_points = self._sample_first_bernstein_segment(
            y_work,
            lambda_end_y,
            self.segment_num_points,
        )

        new_buffer = []
        for i in range(self.segment_num_points):
            new_buffer.append(
                (
                    x_points[i],
                    y_points[i],
                    self.fixed_z_cmd,
                    self.fixed_psi_cmd,
                )
            )

        self._cmd_buffer = new_buffer
        self._cmd_buffer_index = 0

        self._cmd_x = self._cmd_buffer[0][0]
        self._cmd_y = self._cmd_buffer[0][1]
        self._cmd_z = self.fixed_z_cmd
        self._cmd_psi = self.fixed_psi_cmd

        self._committed_vec_seq = cur_vec_seq
        self._have_committed_set = True

        self.get_logger().info(
            f"[COMMIT+SEGMENT] mode={projection_mode} | "
            f"Stored {self.segment_num_points} points. "
            f"lambda_x={lambda_end_x:.4f}, lambda_y={lambda_end_y:.4f} | "
            f"first=({self._cmd_buffer[0][0]:.6f}, {self._cmd_buffer[0][1]:.6f}) | "
            f"last=({self._cmd_buffer[-1][0]:.6f}, {self._cmd_buffer[-1][1]:.6f}) | "
            f"seq={self._committed_vec_seq}"
        )

    def _commit_interpolation_mode(self, x_work, y_work, cur_vec_seq, projection_mode, now):
        tau_sample_x = self._tau_from_extract_iter(self.extract_iter_minus_1_x)
        tau_sample_y = self._tau_from_extract_iter(self.extract_iter_minus_1_y)

        new_x = self._bernstein_eval_tau(x_work, tau_sample_x, 0.0, 1.0)
        new_y = self._bernstein_eval_tau(y_work, tau_sample_y, 0.0, 1.0)

        self.get_logger().info(
            f"[EXTRACTED POINT FOR INTERPOLATION] "
            f"x_extract={new_x:.6f}, y_extract={new_y:.6f}, "
            f"tau_x={tau_sample_x:.6f}, tau_y={tau_sample_y:.6f}"
        )

        new_z = self.fixed_z_cmd
        new_psi = self.fixed_psi_cmd

        self._cmd_x = new_x
        self._cmd_y = new_y
        self._cmd_z = new_z
        self._cmd_psi = new_psi

        self._committed_vec_seq = cur_vec_seq
        self._have_committed_set = True

        if self._act_x is not None and self._act_y is not None:
            self._interp_start_x = self._act_x
            self._interp_start_y = self._act_y
        else:
            self._interp_start_x = self._last_pub_x
            self._interp_start_y = self._last_pub_y
            self.get_logger().warn(
                "Interpolation requested but actual x/y pose not available. "
                "Using last published x/y as interpolation start."
            )

        self._interp_start_z = self.fixed_z_cmd
        self._interp_start_psi = self.fixed_psi_cmd

        self._interp_goal_x = new_x
        self._interp_goal_y = new_y
        self._interp_goal_z = self.fixed_z_cmd
        self._interp_goal_psi = self.fixed_psi_cmd

        self._interp_t0 = now
        self._interp_active = True

        self.get_logger().info(
            f"[COMMIT+INTERP] mode={projection_mode} | "
            f"tau_x={tau_sample_x:.6f}, tau_y={tau_sample_y:.6f} | "
            f"start=({self._interp_start_x:.6f}, {self._interp_start_y:.6f}) -> "
            f"goal=({self._interp_goal_x:.6f}, {self._interp_goal_y:.6f}) | "
            f"z={self.fixed_z_cmd:.6f}, psi={self.fixed_psi_cmd:.6f} | "
            f"duration={self.interp_duration_sec:.3f}s | seq={self._committed_vec_seq}"
        )

    def _print_extracted_point_debug(self, now):
        if not self.enable_bernstein_correction:
            return

        if self._x_vec is None or self._y_vec is None:
            return

        cur_vec_seq = (self._x_vec_seq, self._y_vec_seq)

        now_sec = now.nanoseconds * 1e-9
        should_print_for_new_seq = cur_vec_seq != self._last_block_extract_seq
        should_print_periodic = (now_sec - self._last_block_extract_log_time_sec) >= 1.0

        if not should_print_for_new_seq and not should_print_periodic:
            return

        npx = len(self._x_vec)
        npy = len(self._y_vec)

        if npx == 0 or npy == 0:
            return

        if npx != npy:
            self.get_logger().warn(
                f"[BLOCK_XY DEBUG] Vector length mismatch: len(x)={npx}, len(y)={npy}. Cannot extract debug point."
            )
            return

        x_vec = list(self._x_vec)
        y_vec = list(self._y_vec)

        if self.adjust_control_points:
            if self._act_x is None or self._act_y is None:
                return

            if self._xend is None or self._yend is None:
                return

            x_work = self._warp_control_points(
                x_vec,
                new_start=self._act_x,
                new_end=self._xend,
                wrap_angles=False,
            )
            y_work = self._warp_control_points(
                y_vec,
                new_start=self._act_y,
                new_end=self._yend,
                wrap_angles=False,
            )
            projection_mode = "PROJECTED"
        else:
            x_work = x_vec
            y_work = y_vec
            projection_mode = "RAW"

        tau_sample_x = self._tau_from_extract_iter(self.extract_iter_minus_1_x)
        tau_sample_y = self._tau_from_extract_iter(self.extract_iter_minus_1_y)

        x_extract = self._bernstein_eval_tau(x_work, tau_sample_x, -1.0, 1.0)
        y_extract = self._bernstein_eval_tau(y_work, tau_sample_y, -1.0, 1.0)

        self._last_block_extract_seq = cur_vec_seq
        self._last_block_extract_log_time_sec = now_sec

        self.get_logger().info(
            f"[BLOCK_XY DEBUG EXTRACTED POINT] mode={projection_mode} | "
            f"x_extract={x_extract:.6f}, y_extract={y_extract:.6f} | "
            f"tau_x={tau_sample_x:.6f}, tau_y={tau_sample_y:.6f} | "
            f"extract_iter_x={self.extract_iter_minus_1_x:.3f}, "
            f"extract_iter_y={self.extract_iter_minus_1_y:.3f}, "
            f"extract_ts={self.extract_ts:.3f} | "
            f"commanded_to_drone=({self.x0:.6f}, {self.y0:.6f}, "
            f"{self.fixed_z_cmd:.6f}, {self.fixed_psi_cmd:.6f}) | "
            f"seq={cur_vec_seq}"
        )

    def _get_next_buffered_cmd(self):
        if not self._cmd_buffer:
            return self._cmd_x, self._cmd_y, self.fixed_z_cmd, self.fixed_psi_cmd

        if self._cmd_buffer_index < len(self._cmd_buffer):
            cmd = self._cmd_buffer[self._cmd_buffer_index]
            self._cmd_buffer_index += 1

            self._cmd_x = cmd[0]
            self._cmd_y = cmd[1]
            self._cmd_z = self.fixed_z_cmd
            self._cmd_psi = self.fixed_psi_cmd

            return cmd

        last = self._cmd_buffer[-1]
        self._cmd_x = last[0]
        self._cmd_y = last[1]
        self._cmd_z = self.fixed_z_cmd
        self._cmd_psi = self.fixed_psi_cmd

        return last[0], last[1], self.fixed_z_cmd, self.fixed_psi_cmd

    # ----------------------------
    # Interpolation
    # ----------------------------
    def _get_interpolated_cmd(self, now):
        if not self._interp_active or self._interp_t0 is None:
            return self._cmd_x, self._cmd_y, self.fixed_z_cmd, self.fixed_psi_cmd

        dt = (now - self._interp_t0).nanoseconds * 1e-9

        if dt <= 0.0:
            alpha = 0.0
        else:
            alpha = dt / self.interp_duration_sec

        alpha = max(0.0, min(1.0, alpha))

        x = self._lerp(self._interp_start_x, self._interp_goal_x, alpha)
        y = self._lerp(self._interp_start_y, self._interp_goal_y, alpha)
        z = self.fixed_z_cmd
        psi = self.fixed_psi_cmd

        now_sec = now.nanoseconds * 1e-9
        if (now_sec - self._last_interp_log_time_sec) >= 0.5:
            self.get_logger().info(
                f"[INTERP] alpha={alpha:.3f} "
                f"x={x:.6f} y={y:.6f} z={z:.6f} psi={psi:.6f}"
            )
            self._last_interp_log_time_sec = now_sec

        if alpha >= 1.0:
            self._interp_active = False
            return self._interp_goal_x, self._interp_goal_y, self.fixed_z_cmd, self.fixed_psi_cmd

        return x, y, z, psi

    # ----------------------------
    # Bernstein / de Casteljau helpers
    # ----------------------------
    def _tau_from_extract_iter(self, extract_iter_minus_1):
        t_sample = float(extract_iter_minus_1) * self.extract_ts
        denom_tau = t_sample + 1.0

        if abs(denom_tau) > 1e-12:
            tau = (t_sample - 1.0) / denom_tau
        else:
            tau = -1.0

        return max(-1.0, min(1.0, tau))

    def _lambda_from_extract_iter(self, extract_iter_minus_1):
        tau = self._tau_from_extract_iter(extract_iter_minus_1)
        lam = 0.5 * (tau + 1.0)
        return max(0.0, min(1.0, lam))

    @staticmethod
    def _decasteljau_split_1d(coeffs, lam):
        if not coeffs:
            return [], [], 0.0

        lam = max(0.0, min(1.0, float(lam)))
        points = [float(v) for v in coeffs]

        left = [points[0]]
        right = [points[-1]]

        while len(points) > 1:
            next_points = []
            for i in range(len(points) - 1):
                val = (1.0 - lam) * points[i] + lam * points[i + 1]
                next_points.append(val)

            left.append(next_points[0])
            right.append(next_points[-1])
            points = next_points

        right.reverse()
        pos = points[0]

        return left, right, pos

    def _sample_first_bernstein_segment(self, coeffs, lambda_end, num_points):
        if not coeffs:
            return []

        if num_points < 2:
            raise ValueError("num_points must be at least 2.")

        lambda_end = max(0.0, min(1.0, float(lambda_end)))

        segment_coeffs, _, _ = self._decasteljau_split_1d(coeffs, lambda_end)

        samples = []

        for i in range(num_points):
            mu = float(i) / float(num_points - 1)
            _, _, point = self._decasteljau_split_1d(segment_coeffs, mu)
            samples.append(float(point))

        return samples

    @staticmethod
    def _binom_coeffs(n):
        coeffs = [0.0] * (n + 1)
        coeffs[0] = 1.0
        for k in range(1, n + 1):
            coeffs[k] = coeffs[k - 1] * float(n - k + 1) / float(k)
        return coeffs

    @classmethod
    def _bernstein_eval_tau(cls, coeffs, t, t0=-1.0, tf=1.0):
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
            term *= s ** i
            term *= one_minus ** (n - i)
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
    # Logging helpers
    # ----------------------------
    def _log_missing_inputs_throttled(self, period_sec):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if (now_sec - self._last_missing_log_time_sec) < period_sec:
            return

        missing = []

        if self.block_xy_updates:
            missing_text = "none; block_xy_updates=True"
        else:
            if self._x_vec is None:
                missing.append("/x_vector")
            if self._y_vec is None:
                missing.append("/y_vector")

            if self.adjust_control_points:
                if self._xend is None:
                    missing.append("/x_augment")
                if self._yend is None:
                    missing.append("/y_augment")
                if self._act_x is None or self._act_y is None:
                    missing.append(self.pose_topic)

            missing_text = ", ".join(missing) if missing else "none"

        self.get_logger().warn(
            f"RUN mode but no committed Bernstein command yet. Missing: {missing_text}"
        )
        self._last_missing_log_time_sec = now_sec

    def _warn_if_stale_throttled(self, now, period_sec):
        if self.input_timeout_sec <= 0.0:
            return

        now_sec = now.nanoseconds * 1e-9
        if (now_sec - self._last_stale_warn_time_sec) < period_sec:
            return

        if self.block_xy_updates:
            return

        def age(tstamp):
            if tstamp is None:
                return None
            return (now - tstamp).nanoseconds * 1e-9

        ages = {
            "/x_vector": age(self._x_vec_t),
            "/y_vector": age(self._y_vec_t),
        }

        if self.adjust_control_points:
            ages["/x_augment"] = age(self._xend_t)
            ages["/y_augment"] = age(self._yend_t)
            ages[self.pose_topic] = age(self._act_t)

        stale = {
            k: v
            for k, v in ages.items()
            if (v is None) or (v > self.input_timeout_sec)
        }

        if stale:
            parts = []
            for k, v in stale.items():
                if v is None:
                    parts.append(f"{k}: never")
                else:
                    parts.append(f"{k}: {v:.3f}")

            self.get_logger().warn(
                f"Inputs are older than input_timeout_sec={self.input_timeout_sec:.3f}s "
                f"(but continuing): {', '.join(parts)}"
            )
            self._last_stale_warn_time_sec = now_sec

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def _lerp(a, b, alpha):
        return float(a) + float(alpha) * (float(b) - float(a))

    @staticmethod
    def _wrap_angle(a):
        return math.atan2(math.sin(a), math.cos(a))

    @staticmethod
    def _yaw_from_quaternion(qx, qy, qz, qw):
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def _publish_cmd(self, x, y, z, yaw, source="CMD"):
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

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if (now_sec - self._last_publish_log_time_sec) >= 0.1:
            self.get_logger().info(
                f"[PUBLISH {source}] x={float(x):.6f}, y={float(y):.6f}, "
                f"z={float(z):.6f}, psi={float(yaw):.6f}"
            )
            self._last_publish_log_time_sec = now_sec


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
