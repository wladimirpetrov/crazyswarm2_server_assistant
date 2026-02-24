#!/usr/bin/env python

from pathlib import Path

from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# --- NEW: simple state cache node ---
class CFState(Node):
    def __init__(self, cf_names):
        super().__init__('cf_state_bridge')
        self.latest_pose = {n: None for n in cf_names}
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        for n in cf_names:
            self.create_subscription(
                PoseStamped, f'/{n}/pose',
                lambda msg, name=n: self._pose_cb(name, msg),
                qos
            )

    def _pose_cb(self, name, msg):
        self.latest_pose[name] = msg
        
def main():
    #import rclpy
    #rclpy.init()
    #node = rclpy.create_node("startup_wait")
    #while node.get_clock().now().nanoseconds == 0:
    #     rclpy.spin_once(node, timeout_sec=0.1)
    #node.destroy_node()
    trajs = []
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs
    
    rclpy.init(args=None)
    cf_names = [cf.name for cf in allcfs.crazyflies]
    state_node = CFState(cf_names)
    exec_ = rclpy.executors.SingleThreadedExecutor()
    exec_.add_node(state_node)
    threading.Thread(target=exec_.spin, daemon=True).start()
    
    traj1 = Trajectory()
    traj1.loadcsv(Path(__file__).parent / 'data/traj_mp_1.csv')
    
    
    
    # enable logging
    allcfs.setParam('usd.logging', 1)
    timeHelper.sleep(2.5)
    TRIALS = 1
    TIMESCALE = 1.0
    for i in range(TRIALS):
        cfs = swarm.allcfs.crazyflies
        cf1, cf2 = cfs[0], cfs[1]
        timeHelper.sleep(1.0)
        
        print("Begin takeoff")
        allcfs.takeoff(targetHeight=1.0, duration=2.0)
        
        print("Uploading trajectory 1...")
        cf1.uploadTrajectory(0,0,traj1)
        print("All trajectories uploaded")
        
        #cf.goTo(np.array([0,0,1.0]),0,2.0)
        #for cf in allcfs.crazyflies:
        #    pos = np.array(cf.initialPosition) + np.array([0, 0, 1.0])
        #    cf.goTo(pos, 0, 2.0)
        #timeHelper.sleep(3.0)

        #for cf in allcfs.crazyflies:
        #    pos = np.array(cf.initialPosition) + np.array([0.0, 0.0, 1.0])
        #    cf.goTo(pos, 0, 2.0)

        timeHelper.sleep(3.0)
        print("start trajectories")
        cf1.startTrajectory(0, timescale=TIMESCALE)
        print("finished start traj")
        timeHelper.sleep(2.0)
        
        
        # --- NEW: wait until we have at least one pose for cf1 ---
        while state_node.latest_pose.get(cf1.name) is None:
            timeHelper.sleep(0.02)

        # Snapshot cf1's current position and send cf2 there (absolute command)
        p1 = state_node.latest_pose[cf1.name].pose.position
        goal = [p1.x, p1.y, p1.z]
        # Use keyword args if your build supports them; otherwise use positional
        try:
            cf2.goTo(goal, yaw=0.0, duration=2.0, relative=False)
        except TypeError:
            cf2.goTo(goal, 0.0, 2.0, False)
        
        timeHelper.sleep(traj1.duration * TIMESCALE)

        allcfs.land(targetHeight=0.06, duration=2.0)
        
        timeHelper.sleep(3.0)

    # disable logging
    allcfs.setParam('usd.logging', 0)

    exec_.shutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
