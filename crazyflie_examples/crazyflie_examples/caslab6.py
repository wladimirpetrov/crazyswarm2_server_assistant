#!/usr/bin/env python

from pathlib import Path

from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
import numpy as np


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
    traj1 = Trajectory()
    traj1.loadcsv(Path(__file__).parent / 'data/traj_mp_1.csv')
    traj2 = Trajectory()
    traj2.loadcsv(Path(__file__).parent / 'data/traj_mp_2.csv')
    traj3 = Trajectory()
    traj3.loadcsv(Path(__file__).parent / 'data/traj_mp_3.csv')
    traj4 = Trajectory()
    traj4.loadcsv(Path(__file__).parent / 'data/traj_mp_45.csv')
    traj6 = Trajectory()
    traj6.loadcsv(Path(__file__).parent / 'data/traj_mp_6.csv')
    traj5 = Trajectory()
    traj5.loadcsv(Path(__file__).parent / 'data/traj_mp_5.csv')
    
    
    #n=4
    #for i in range(n):
    #    traj = Trajectory()
    #    traj.loadcsv(Path(__file__).parent / f'data/traj1{i}b.csv')
    #    trajs.append(traj)
    
    # enable logging
    allcfs.setParam('usd.logging', 1)
    timeHelper.sleep(2.5)
    TRIALS = 1
    TIMESCALE = 1.0
    for i in range(TRIALS):
        cfs = swarm.allcfs.crazyflies
        cf1, cf2, cf3, cf4, cf5, cf6 = cfs[0], cfs[1], cfs[2], cfs[3], cfs[4], cfs[5]
        timeHelper.sleep(1.0)
        
        print("Begin takeoff")
        allcfs.takeoff(targetHeight=1.0, duration=2.0)
        
        print("Uploading trajectory 1...")
        cf1.uploadTrajectory(0,0,traj1)
        print("Uploading trajectory 2...")
        cf2.uploadTrajectory(0,0,traj2)
        print("Uploading trajectory 3...")
        cf3.uploadTrajectory(0,0,traj3)
        print("Uploading trajectory 4...")
        cf4.uploadTrajectory(0,0,traj4)
        print("Uploading trajectory 5...")
        cf5.uploadTrajectory(0,0,traj5)
        print("Uploading trajectory 6...")
        cf6.uploadTrajectory(0,0,traj6)
        
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
        cf2.startTrajectory(0, timescale=TIMESCALE)
        cf3.startTrajectory(0, timescale=TIMESCALE)
        cf4.startTrajectory(0, timescale=TIMESCALE)
        cf5.startTrajectory(0, timescale=TIMESCALE)
        cf6.startTrajectory(0, timescale=TIMESCALE)
        print("finished start traj")



        #timeHelper.sleep(max([t.duration for t in trajs]) * TIMESCALE + 2.0)

        timeHelper.sleep(traj3.duration * TIMESCALE + 2.0)
        
        #allcfs.land(targetHeight=0.06, duration=2.0)
        
        
        cf3.land(targetHeight=0.06, duration=2.0)
        print(traj3.duration * TIMESCALE + 2.0)
        timeHelper.sleep(-traj3.duration*TIMESCALE+traj2.duration*TIMESCALE)
        cf2.land(targetHeight=0.06, duration=2.0)
        print(traj3.duration*TIMESCALE-traj2.duration*TIMESCALE+2.0)
        timeHelper.sleep(-traj2.duration*TIMESCALE+traj1.duration*TIMESCALE)
        print(traj2.duration*TIMESCALE-traj1.duration*TIMESCALE+2.0)
        cf1.land(targetHeight=0.06, duration=2.0)
        timeHelper.sleep(-traj1.duration*TIMESCALE+traj4.duration*TIMESCALE)
        cf4.land(targetHeight=0.06, duration=2.0)
        print(traj1.duration*TIMESCALE-traj4.duration*TIMESCALE+2.0)
        
        timeHelper.sleep(3.0)

    # disable logging
    allcfs.setParam('usd.logging', 0)


if __name__ == '__main__':
    main()
