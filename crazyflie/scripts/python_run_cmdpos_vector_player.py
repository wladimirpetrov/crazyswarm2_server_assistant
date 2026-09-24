#!/usr/bin/env python3
"""
python_run_cmdpos_vector_player_interpolated.py

ROS 2 node that reads python_run.csv and plays it as:

1. TAKEOFF
   - ramp z from takeoff_z0 to the first CSV point z
   - hold x/y at the first CSV point
   - yaw = first CSV psi or 0

2. RUN
   - for each CSV row, publish /cf_name/cmd_position using:
       init_px, init_py, init_pz, init_psi
   - optionally interpolate from the previously commanded position to the
     current row target over cmd_interpolation_steps timer ticks
   - choose a new Bernstein-vector row every vector_publish_stride rows
   - once a vector row is selected, keep re-publishing that SAME active
     vector bundle regularly while cmd_position advances row by row:
       /x_vector   = [px_c0 ... px_c4]
       /y_vector   = [py_c0 ... py_c4]
       /z_vector   = [pz_c0 ... pz_c4]
       /psi_vector = [psi_c0 ... psi_c4]

   In other words, vector_publish_stride controls when the ACTIVE vector
   changes, not whether vectors are published only once.

3. PAUSE
   - when any point_id in pause_point_ids is reached, hold that row's cmd_position for pause_duration_sec
   - during the pause, keep publishing the previously active vector bundle, not the pause row's vectors
   - then continue at the next row
   - by default, force-publish vectors again when resuming after the pause

4. FINAL_HOVER
   - when final_point_id is reached, hold that final cmd_position for final_hover_sec

5. LAND
   - ramp z down to land_z while holding final x/y/yaw
   - then keep publishing the landed command in LANDED state
"""

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64, Float64MultiArray
from crazyflie_interfaces.msg import Position


@dataclass
class PythonRunRow:
    point_id: int
    x: float
    y: float
    z: float
    psi: float
    px: List[float]
    py: List[float]
    pz: List[float]
    ppsi: List[float]


class PythonRunCmdposVectorPlayer(Node):
    def __init__(self) -> None:
        super().__init__('python_run_cmdpos_vector_player')

        # ============================================================
        # Parameters
        # ============================================================
        self.declare_parameter('cf_name', 'cf231')
        self.declare_parameter('csv_file', 'python_run.csv')
        self.declare_parameter('csv_has_header', True)

        # Topics expected by bernstein_trajectory_player.
        self.declare_parameter('x_vector_topic', '/x_vector')
        self.declare_parameter('y_vector_topic', '/y_vector')
        self.declare_parameter('z_vector_topic', '/z_vector')
        self.declare_parameter('psi_vector_topic', '/psi_vector')

        # Playback timing.
        self.declare_parameter('publish_rate_hz', 100.0)

        # Number of cmd_position commands used to move from the previously
        # commanded point to the next CSV row target.
        # 1 = no interpolation: publish the row directly.
        # 5 = publish 5 interpolated commands, ending exactly at the row target.
        self.declare_parameter('cmd_interpolation_steps', 20)

        # Backward-compatible old parameter. Not used for interpolation.
        self.declare_parameter('row_repeat_count', 1)

        # Vector publication cadence.
        # If publish_vector_at_group_end=True and vector_publish_stride=5:
        #   publish vectors at row sequence indices 0, 4, 9, 14, ...
        # This matches the interpretation: first row, then after every 5 rows.
        # If publish_vector_at_group_end=False and vector_publish_stride=5:
        #   publish vectors at row sequence indices 0, 5, 10, 15, ...
        self.declare_parameter('vector_publish_stride', 3)
        self.declare_parameter('publish_vector_at_group_end', True)
        self.declare_parameter('force_vector_publish_at_point_ids', [0, 100])

        # If True, selected/active vectors are re-published every timer tick.
        # vector_publish_stride then means: choose a NEW vector every N rows.
        self.declare_parameter('republish_active_vectors', True)

        # Takeoff settings.
        self.declare_parameter('takeoff_duration', 2.0)
        self.declare_parameter('takeoff_z0', 0.0)
        self.declare_parameter('use_first_csv_point_for_takeoff', True)
        self.declare_parameter('takeoff_x', -1.8)
        self.declare_parameter('takeoff_y', 3.0)
        self.declare_parameter('takeoff_z', 0.6)
        self.declare_parameter('takeoff_psi', 0.0)

        # Pause at point_id 99, then resume at point_id 100.
        self.declare_parameter('pause_enabled', True)
        # Backward-compatible single pause parameter.
        self.declare_parameter('pause_point_id', 99)
        # Main parameter: one or more pause point_ids.
        # Default: pause at point_id 99 and point_id 134.
        self.declare_parameter('pause_point_ids', [99, 134])
        self.declare_parameter('pause_duration_sec', 2.0)
        self.declare_parameter('force_vector_after_pause', True)

        # Final hover and landing at point_id 199.
        self.declare_parameter('final_point_id', 199)
        self.declare_parameter('final_hover_sec', 2.0)
        self.declare_parameter('landing_duration_sec', 2.0)
        self.declare_parameter('land_z', 0.0)

        self.declare_parameter('publish_l1_status', True)
        self.declare_parameter('debug_period_sec', 0.5)

        # ============================================================
        # Read parameters
        # ============================================================
        self.cf_name = str(self.get_parameter('cf_name').value)
        self.csv_file = str(self.get_parameter('csv_file').value)
        self.csv_has_header = bool(self.get_parameter('csv_has_header').value)

        self.x_vector_topic = str(self.get_parameter('x_vector_topic').value)
        self.y_vector_topic = str(self.get_parameter('y_vector_topic').value)
        self.z_vector_topic = str(self.get_parameter('z_vector_topic').value)
        self.psi_vector_topic = str(self.get_parameter('psi_vector_topic').value)

        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        if self.publish_rate_hz <= 0.0:
            self.get_logger().warn('publish_rate_hz must be > 0. Using 100.0 Hz.')
            self.publish_rate_hz = 100.0

        self.cmd_interpolation_steps = int(self.get_parameter('cmd_interpolation_steps').value)
        if self.cmd_interpolation_steps < 1:
            self.get_logger().warn('cmd_interpolation_steps must be >= 1. Using 1.')
            self.cmd_interpolation_steps = 1

        # Old parameter is still read only so old launch files do not fail.
        self.row_repeat_count = int(self.get_parameter('row_repeat_count').value)

        self.vector_publish_stride = int(self.get_parameter('vector_publish_stride').value)
        if self.vector_publish_stride < 1:
            self.get_logger().warn('vector_publish_stride must be >= 1. Using 1.')
            self.vector_publish_stride = 1

        self.publish_vector_at_group_end = bool(
            self.get_parameter('publish_vector_at_group_end').value
        )
        self.force_vector_publish_at_point_ids = self._read_int_list_parameter(
            'force_vector_publish_at_point_ids'
        )
        self.republish_active_vectors = bool(
            self.get_parameter('republish_active_vectors').value
        )

        self.takeoff_duration = float(self.get_parameter('takeoff_duration').value)
        if self.takeoff_duration < 0.0:
            self.get_logger().warn('takeoff_duration must be >= 0. Using 2.0 s.')
            self.takeoff_duration = 2.0

        self.takeoff_z0 = float(self.get_parameter('takeoff_z0').value)
        self.use_first_csv_point_for_takeoff = bool(
            self.get_parameter('use_first_csv_point_for_takeoff').value
        )
        self.takeoff_x = float(self.get_parameter('takeoff_x').value)
        self.takeoff_y = float(self.get_parameter('takeoff_y').value)
        self.takeoff_z = float(self.get_parameter('takeoff_z').value)
        self.takeoff_psi = float(self.get_parameter('takeoff_psi').value)

        self.pause_enabled = bool(self.get_parameter('pause_enabled').value)
        self.pause_point_id = int(self.get_parameter('pause_point_id').value)
        self.pause_point_ids = self._read_int_list_parameter('pause_point_ids')
        if len(self.pause_point_ids) == 0:
            self.pause_point_ids = {self.pause_point_id}
        self.pause_duration_sec = float(self.get_parameter('pause_duration_sec').value)
        if self.pause_duration_sec < 0.0:
            self.pause_duration_sec = 0.0
        self.force_vector_after_pause = bool(self.get_parameter('force_vector_after_pause').value)

        self.final_point_id = int(self.get_parameter('final_point_id').value)
        self.final_hover_sec = float(self.get_parameter('final_hover_sec').value)
        if self.final_hover_sec < 0.0:
            self.final_hover_sec = 0.0

        self.landing_duration_sec = float(self.get_parameter('landing_duration_sec').value)
        if self.landing_duration_sec < 0.0:
            self.landing_duration_sec = 0.0
        self.land_z = float(self.get_parameter('land_z').value)

        self.publish_l1_status = bool(self.get_parameter('publish_l1_status').value)
        self.debug_period_sec = float(self.get_parameter('debug_period_sec').value)
        if self.debug_period_sec <= 0.0:
            self.debug_period_sec = 0.5

        # ============================================================
        # QoS
        # ============================================================
        self.qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # The BernsteinTrajectoryPlayer subscribers use RELIABLE QoS.
        self.qos_vectors = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )

        # ============================================================
        # Publishers
        # ============================================================
        self.cmd_topic = f'/{self.cf_name}/cmd_position'
        self.cmd_pub = self.create_publisher(Position, self.cmd_topic, self.qos_cmd)

        self.x_vector_pub = self.create_publisher(Float64MultiArray, self.x_vector_topic, self.qos_vectors)
        self.y_vector_pub = self.create_publisher(Float64MultiArray, self.y_vector_topic, self.qos_vectors)
        self.z_vector_pub = self.create_publisher(Float64MultiArray, self.z_vector_topic, self.qos_vectors)
        self.psi_vector_pub = self.create_publisher(Float64MultiArray, self.psi_vector_topic, self.qos_vectors)

        self.l1_status_pub = self.create_publisher(Float64, '/l1_status', self.qos_cmd)

        # ============================================================
        # Load CSV
        # ============================================================
        self.rows = self._load_python_run_csv(self.csv_file)
        if len(self.rows) == 0:
            raise RuntimeError('python_run.csv contains no usable trajectory rows.')

        self.point_id_to_index: Dict[int, int] = {
            row.point_id: idx for idx, row in enumerate(self.rows)
        }

        if self.use_first_csv_point_for_takeoff:
            first = self.rows[0]
            self.takeoff_x = first.x
            self.takeoff_y = first.y
            self.takeoff_z = first.z
            self.takeoff_psi = first.psi

        if self.final_point_id not in self.point_id_to_index:
            self.get_logger().warn(
                f'final_point_id={self.final_point_id} not found in CSV. '
                f'Using last CSV point_id={self.rows[-1].point_id} as final.'
            )
            self.final_point_id = self.rows[-1].point_id

        # ============================================================
        # State variables
        # ============================================================
        self.state = 'TAKEOFF'
        self.state_start_time = self.get_clock().now()
        self.timer_period = 1.0 / self.publish_rate_hz
        self.row_sequence_index = 0
        self.cmd_interp_step = 0
        self.cmd_segment_start: Optional[PythonRunRow] = None
        self.cmd_segment_start_values: Optional[List[float]] = None
        self.cmd_segment_target_point_id: Optional[int] = None
        self._paused_point_ids: Set[int] = set()
        self._force_vector_on_next_row = False
        self.active_vector_row: Optional[PythonRunRow] = None
        self.active_vector_point_id: Optional[int] = None

        self.last_cmd_x = self.takeoff_x
        self.last_cmd_y = self.takeoff_y
        self.last_cmd_z = self.takeoff_z0
        self.last_cmd_psi = self.takeoff_psi

        self._last_debug_log_time_sec = -1e9
        self._finished_logged = False

        self.cmd_timer = self.create_timer(self.timer_period, self._timer_cb)
        if self.publish_l1_status:
            self.status_timer = self.create_timer(1.0, self._status_timer_cb)
        else:
            self.status_timer = None

        # ============================================================
        # Startup logs
        # ============================================================
        self.get_logger().info('---------------------------------------------')
        self.get_logger().info('python_run_cmdpos_vector_player started')
        self.get_logger().info(f'cmd_topic                         : {self.cmd_topic}')
        self.get_logger().info(f'csv_file                          : {self.csv_file}')
        self.get_logger().info(f'loaded rows                        : {len(self.rows)}')
        self.get_logger().info(f'first point_id                     : {self.rows[0].point_id}')
        self.get_logger().info(f'last point_id                      : {self.rows[-1].point_id}')
        self.get_logger().info(f'x_vector_topic                    : {self.x_vector_topic}')
        self.get_logger().info(f'y_vector_topic                    : {self.y_vector_topic}')
        self.get_logger().info(f'z_vector_topic                    : {self.z_vector_topic}')
        self.get_logger().info(f'psi_vector_topic                  : {self.psi_vector_topic}')
        self.get_logger().info(f'publish_rate_hz                   : {self.publish_rate_hz:.3f}')
        self.get_logger().info(f'cmd_interpolation_steps           : {self.cmd_interpolation_steps}')
        self.get_logger().info(f'vector_publish_stride             : {self.vector_publish_stride}')
        self.get_logger().info(f'publish_vector_at_group_end       : {self.publish_vector_at_group_end}')
        self.get_logger().info(f'force_vector_publish_at_point_ids : {sorted(self.force_vector_publish_at_point_ids)}')
        self.get_logger().info(f'republish_active_vectors          : {self.republish_active_vectors}')
        self.get_logger().info(
            f'takeoff target                    : '
            f'[{self.takeoff_x:.6f}, {self.takeoff_y:.6f}, {self.takeoff_z:.6f}], '
            f'psi={self.takeoff_psi:.6f}'
        )
        self.get_logger().info(
            f'pause                             : enabled={self.pause_enabled}, '
            f'point_ids={sorted(self.pause_point_ids)}, duration={self.pause_duration_sec:.3f}s'
        )
        self.get_logger().info(
            f'final/landing                     : final_point_id={self.final_point_id}, '
            f'hover={self.final_hover_sec:.3f}s, landing={self.landing_duration_sec:.3f}s, land_z={self.land_z:.6f}'
        )
        self.get_logger().info('---------------------------------------------')

    # ============================================================
    # Parameter helpers
    # ============================================================
    def _read_int_list_parameter(self, name: str) -> Set[int]:
        value = self.get_parameter(name).value
        if value is None:
            return set()
        if isinstance(value, (list, tuple)):
            return {int(v) for v in value}
        if isinstance(value, str):
            if not value.strip():
                return set()
            return {int(part.strip()) for part in value.split(',') if part.strip()}
        try:
            return {int(value)}
        except Exception:
            self.get_logger().warn(f'Could not parse {name}={value}. Using empty set.')
            return set()

    # ============================================================
    # CSV loading
    # ============================================================
    def _load_python_run_csv(self, csv_file: str) -> List[PythonRunRow]:
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f'CSV file "{csv_file}" not found.')

        required_cols = [
            'point_id',
            'init_px', 'init_py', 'init_pz',
            'px_c0', 'px_c1', 'px_c2', 'px_c3', 'px_c4',
            'py_c0', 'py_c1', 'py_c2', 'py_c3', 'py_c4',
            'pz_c0', 'pz_c1', 'pz_c2', 'pz_c3', 'pz_c4',
            'psi_c0', 'psi_c1', 'psi_c2', 'psi_c3', 'psi_c4',
        ]

        rows: List[PythonRunRow] = []

        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise RuntimeError('CSV header not found.')

            normalized_to_original = {
                name.strip(): name for name in reader.fieldnames
            }

            missing = [col for col in required_cols if col not in normalized_to_original]
            if missing:
                raise RuntimeError(
                    f'python_run.csv is missing required columns: {missing}. '
                    f'Found columns: {reader.fieldnames}'
                )

            # init_psi is optional. If missing, yaw is zero.
            init_psi_key = normalized_to_original.get('init_psi')

            for physical_row, row in enumerate(reader, start=2):
                try:
                    point_id = int(float(row[normalized_to_original['point_id']]))

                    x = float(row[normalized_to_original['init_px']])
                    y = float(row[normalized_to_original['init_py']])
                    z = float(row[normalized_to_original['init_pz']])

                    if init_psi_key is not None and row[init_psi_key] != '':
                        psi = float(row[init_psi_key])
                    else:
                        psi = 0.0

                    px = [float(row[normalized_to_original[f'px_c{i}']]) for i in range(5)]
                    py = [float(row[normalized_to_original[f'py_c{i}']]) for i in range(5)]
                    pz = [float(row[normalized_to_original[f'pz_c{i}']]) for i in range(5)]
                    ppsi = [float(row[normalized_to_original[f'psi_c{i}']]) for i in range(5)]

                    rows.append(PythonRunRow(
                        point_id=point_id,
                        x=x,
                        y=y,
                        z=z,
                        psi=psi,
                        px=px,
                        py=py,
                        pz=pz,
                        ppsi=ppsi,
                    ))
                except Exception as exc:
                    raise RuntimeError(
                        f'Failed to parse python_run.csv at physical row {physical_row}: {row}'
                    ) from exc

        rows.sort(key=lambda r: r.point_id)
        return rows

    # ============================================================
    # Timer callbacks
    # ============================================================
    def _status_timer_cb(self) -> None:
        if self.state in ('RUN', 'PAUSE', 'FINAL_HOVER', 'LAND'):
            msg = Float64()
            msg.data = 1.0
            self.l1_status_pub.publish(msg)

    def _timer_cb(self) -> None:
        now = self.get_clock().now()

        if self.state == 'TAKEOFF':
            self._run_takeoff(now)
        elif self.state == 'RUN':
            self._run_rows(now)
        elif self.state == 'PAUSE':
            self._run_pause(now)
        elif self.state == 'FINAL_HOVER':
            self._run_final_hover(now)
        elif self.state == 'LAND':
            self._run_land(now)
        elif self.state == 'LANDED':
            self._run_landed(now)
        else:
            self.get_logger().warn(f'Unknown state {self.state}. Switching to LANDED.')
            self.state = 'LANDED'
            self._run_landed(now)

    # ============================================================
    # State logic
    # ============================================================
    def _run_takeoff(self, now) -> None:
        elapsed = (now - self.state_start_time).nanoseconds * 1e-9
        if self.takeoff_duration <= 1e-9:
            alpha = 1.0
        else:
            alpha = elapsed / self.takeoff_duration
        alpha = max(0.0, min(1.0, alpha))

        z_cmd = self.takeoff_z0 + alpha * (self.takeoff_z - self.takeoff_z0)

        self._publish_cmd(
            x=self.takeoff_x,
            y=self.takeoff_y,
            z=z_cmd,
            yaw=self.takeoff_psi,
            source='TAKEOFF',
        )

        if alpha >= 1.0:
            self.state = 'RUN'
            self.state_start_time = now
            self.row_sequence_index = 0
            self.cmd_interp_step = 0
            self.cmd_segment_start_values = None
            self.cmd_segment_target_point_id = None
            self.get_logger().info('Takeoff complete. Starting interpolated python_run.csv playback.')

    def _run_rows(self, now) -> None:
        if self.row_sequence_index >= len(self.rows):
            self._start_final_hover(now)
            return

        row = self.rows[self.row_sequence_index]

        # At the beginning of each new row segment, choose whether a new
        # Bernstein vector bundle becomes active. Then keep re-publishing the
        # active bundle while the cmd_position interpolation runs.
        vector_selected_now = False
        if self.cmd_segment_target_point_id != row.point_id:
            self._begin_cmd_segment(row)

            pause_trigger_for_this_row = (
                self.pause_enabled and
                row.point_id in self.pause_point_ids and
                row.point_id not in self._paused_point_ids
            )

            # Pause rows are special: do NOT select the pause row's vectors.
            # Keep the old active vector during the pause. After the pause, the
            # next row force-selects its own vectors.
            if not pause_trigger_for_this_row:
                vector_selected_now = self._should_select_vectors_for_current_row(row)
                if vector_selected_now:
                    self._set_active_vector_row(row, reason='cadence/forced')

        self._publish_active_vectors_if_enabled()

        x_cmd, y_cmd, z_cmd, psi_cmd = self._interpolated_cmd_for_current_step(row)
        self._publish_cmd(x_cmd, y_cmd, z_cmd, psi_cmd, source='RUN_INTERP')
        self._debug_row_log(now, row, vector_selected_now)

        self.cmd_interp_step += 1

        # Stay on this same CSV row until the interpolated command reaches it.
        if self.cmd_interp_step < self.cmd_interpolation_steps:
            return

        # The current row target has now been reached exactly.
        # Handle final/pause logic only after reaching the row, not at the first
        # interpolation sample.
        if row.point_id == self.final_point_id:
            self._start_final_hover(now)
            return

        pause_trigger_now = (
            self.pause_enabled and
            row.point_id in self.pause_point_ids and
            row.point_id not in self._paused_point_ids
        )

        if pause_trigger_now:
            self._paused_point_ids.add(row.point_id)
            self.state = 'PAUSE'
            self.state_start_time = now
            self.get_logger().warn(
                f'Reached pause_point_id={row.point_id}. '
                f'Holding for {self.pause_duration_sec:.3f} seconds with old active vectors. '
                f'After pause, next row will force-select vectors.'
            )
            return

        self._advance_after_completed_row()

    def _run_pause(self, now) -> None:
        pause_row = self.rows[self.row_sequence_index]
        self._publish_active_vectors_if_enabled()
        self._publish_cmd(pause_row.x, pause_row.y, pause_row.z, pause_row.psi, source='PAUSE')

        elapsed = (now - self.state_start_time).nanoseconds * 1e-9
        if elapsed >= self.pause_duration_sec:
            self._advance_after_completed_row(force=True)
            self.state = 'RUN'
            self.state_start_time = now
            if self.force_vector_after_pause:
                self._force_vector_on_next_row = True
            self.get_logger().info('Pause complete. Resuming CSV playback.')

    def _start_final_hover(self, now) -> None:
        self.state = 'FINAL_HOVER'
        self.state_start_time = now
        final_idx = self.point_id_to_index.get(self.final_point_id, len(self.rows) - 1)
        self.row_sequence_index = final_idx
        final_row = self.rows[final_idx]
        self.get_logger().warn(
            f'Reached final point_id={final_row.point_id}. '
            f'Hovering for {self.final_hover_sec:.3f} seconds before landing.'
        )

    def _run_final_hover(self, now) -> None:
        final_row = self.rows[self.row_sequence_index]
        self._publish_active_vectors_if_enabled()
        self._publish_cmd(final_row.x, final_row.y, final_row.z, final_row.psi, source='FINAL_HOVER')

        elapsed = (now - self.state_start_time).nanoseconds * 1e-9
        if elapsed >= self.final_hover_sec:
            self.state = 'LAND'
            self.state_start_time = now
            self.get_logger().warn(
                f'Final hover complete. Landing to z={self.land_z:.6f} '
                f'over {self.landing_duration_sec:.3f} seconds.'
            )

    def _run_land(self, now) -> None:
        final_row = self.rows[self.row_sequence_index]
        elapsed = (now - self.state_start_time).nanoseconds * 1e-9

        if self.landing_duration_sec <= 1e-9:
            alpha = 1.0
        else:
            alpha = elapsed / self.landing_duration_sec
        alpha = max(0.0, min(1.0, alpha))

        z_cmd = final_row.z + alpha * (self.land_z - final_row.z)
        self._publish_active_vectors_if_enabled()
        self._publish_cmd(final_row.x, final_row.y, z_cmd, final_row.psi, source='LAND')

        if alpha >= 1.0:
            self.state = 'LANDED'
            self.state_start_time = now
            self.get_logger().warn('Landing complete. Holding LANDED command.')

    def _run_landed(self, now) -> None:
        final_row = self.rows[self.row_sequence_index]
        self._publish_active_vectors_if_enabled()
        self._publish_cmd(final_row.x, final_row.y, self.land_z, final_row.psi, source='LANDED')

    def _begin_cmd_segment(self, row: PythonRunRow) -> None:
        self.cmd_segment_target_point_id = row.point_id
        self.cmd_interp_step = 0
        self.cmd_segment_start_values = [
            self.last_cmd_x,
            self.last_cmd_y,
            self.last_cmd_z,
            self.last_cmd_psi,
        ]

    def _interpolated_cmd_for_current_step(self, row: PythonRunRow):
        if self.cmd_segment_start_values is None:
            self._begin_cmd_segment(row)

        sx, sy, sz, spsi = self.cmd_segment_start_values
        alpha = float(self.cmd_interp_step + 1) / float(self.cmd_interpolation_steps)
        alpha = max(0.0, min(1.0, alpha))

        x_cmd = sx + alpha * (row.x - sx)
        y_cmd = sy + alpha * (row.y - sy)
        z_cmd = sz + alpha * (row.z - sz)
        psi_cmd = spsi + alpha * (row.psi - spsi)

        return x_cmd, y_cmd, z_cmd, psi_cmd

    def _advance_after_completed_row(self, force: bool = False) -> None:
        self.cmd_interp_step = 0
        self.cmd_segment_start_values = None
        self.cmd_segment_target_point_id = None
        self.row_sequence_index += 1

    # ============================================================
    # Vector selection cadence
    # ============================================================
    def _should_select_vectors_for_current_row(self, row: PythonRunRow) -> bool:
        if self._force_vector_on_next_row:
            self._force_vector_on_next_row = False
            return True

        if row.point_id in self.force_vector_publish_at_point_ids:
            return True

        idx = self.row_sequence_index

        if idx == 0:
            return True

        if self.vector_publish_stride <= 1:
            return True

        if self.publish_vector_at_group_end:
            return ((idx + 1) % self.vector_publish_stride) == 0

        return (idx % self.vector_publish_stride) == 0

    # ============================================================
    # Publishers
    # ============================================================
    def _set_active_vector_row(self, row: PythonRunRow, reason: str) -> None:
        self.active_vector_row = row
        self.active_vector_point_id = row.point_id

        self.get_logger().info(
            f'[VECTOR SELECT] active point_id={row.point_id}, reason={reason}, '
            f'x_vector={self._fmt_vector(row.px)}, '
            f'y_vector={self._fmt_vector(row.py)}, '
            f'z_vector={self._fmt_vector(row.pz)}, '
            f'psi_vector={self._fmt_vector(row.ppsi)}'
        )

        # If not re-publishing active vectors continuously, still publish once
        # immediately when the active vector row changes.
        if not self.republish_active_vectors:
            self._publish_vectors(row)

    def _publish_active_vectors_if_enabled(self) -> None:
        if not self.republish_active_vectors:
            return
        if self.active_vector_row is None:
            return
        self._publish_vectors(self.active_vector_row)

    def _publish_vectors(self, row: PythonRunRow) -> None:
        msg_x = Float64MultiArray()
        msg_y = Float64MultiArray()
        msg_z = Float64MultiArray()
        msg_psi = Float64MultiArray()

        msg_x.data = [float(v) for v in row.px]
        msg_y.data = [float(v) for v in row.py]
        msg_z.data = [float(v) for v in row.pz]
        msg_psi.data = [float(v) for v in row.ppsi]

        # In _run_rows(), this is called before cmd_position.
        self.x_vector_pub.publish(msg_x)
        self.y_vector_pub.publish(msg_y)
        self.z_vector_pub.publish(msg_z)
        self.psi_vector_pub.publish(msg_psi)

    def _publish_cmd(self, x: float, y: float, z: float, yaw: float, source: str) -> None:
        msg = Position()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        msg.yaw = float(yaw)

        self.cmd_pub.publish(msg)

        self.last_cmd_x = float(x)
        self.last_cmd_y = float(y)
        self.last_cmd_z = float(z)
        self.last_cmd_psi = float(yaw)

    # ============================================================
    # Logging helpers
    # ============================================================
    def _debug_row_log(self, now, row: PythonRunRow, vector_selected_now: bool) -> None:
        now_sec = now.nanoseconds * 1e-9
        if (now_sec - self._last_debug_log_time_sec) < self.debug_period_sec:
            return

        self.get_logger().info(
            f'[RUN] seq_idx={self.row_sequence_index}/{len(self.rows)-1}, '
            f'point_id={row.point_id}, interp_step={self.cmd_interp_step+1}/{self.cmd_interpolation_steps}, '
            f'vector_selected_now={vector_selected_now}, '
            f'active_vector_point_id={self.active_vector_point_id}, '
            f'x={row.x:.6f}, y={row.y:.6f}, z={row.z:.6f}, psi={row.psi:.6f}'
        )
        self._last_debug_log_time_sec = now_sec

    def _fmt_vector(self, values: List[float]) -> str:
        return '[' + ', '.join(f'{v:.4g}' for v in values) + ']'


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[PythonRunCmdposVectorPlayer] = None

    try:
        node = PythonRunCmdposVectorPlayer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f'Fatal error: {exc}')
        else:
            print(f'Fatal error: {exc}')
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
