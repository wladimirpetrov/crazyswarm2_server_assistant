#!/usr/bin/env python

from pathlib import Path

from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
import numpy as np


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    traj1 = Trajectory()
    traj1.loadcsv(Path(__file__).parent / 'data/traj_mp_1a.csv')
    traj2 = Trajectory()
    traj2.loadcsv(Path(__file__).parent / 'data/traj_mp_2a.csv')
    traj3 = Trajectory()
    traj3.loadcsv(Path(__file__).parent / 'data/traj_mp_3a.csv')
    traj4 = Trajectory()
    traj4.loadcsv(Path(__file__).parent / 'data/traj_mp_45a.csv')

    # enable logging
    allcfs.setParam('usd.logging', 1)

    TRIALS = 2
    TIMESCALE = 1.0
    cfs = swarm.allcfs.crazyflies
    cf1, cf2, cf3, cf4 = cfs[0], cfs[1], cfs[2], cfs[3]
    
    print("Uploading trajectory 1...")
    cf1.uploadTrajectory(0,0,traj1)
    print("Uploading trajectory 2...")
    cf2.uploadTrajectory(0,0,traj2)
    print("Uploading trajectory 3...")
    cf3.uploadTrajectory(0,0,traj3)
    print("Uploading trajectory 4...")
    cf4.uploadTrajectory(0,0,traj4)
    print("All trajectories uploaded")
    allcfs.takeoff(targetHeight=1.0, duration=2.0)
    timeHelper.sleep(2.5)
    for i in range(TRIALS):
        
        
        
        cf1.startTrajectory(0, timescale=TIMESCALE)
        cf2.startTrajectory(0, timescale=TIMESCALE)
        cf3.startTrajectory(0, timescale=TIMESCALE)
        if i > 0:
            cf4.startTrajectory(0, timescale=TIMESCALE)

        timeHelper.sleep(traj1.duration * TIMESCALE + 2.0)
        # allcfs.startTrajectory(0, timescale=TIMESCALE, reverse=True)
        # timeHelper.sleep(traj1.duration * TIMESCALE + 2.0)

    allcfs.land(targetHeight=0.06, duration=2.0)
    timeHelper.sleep(3.0)

    # disable logging
    allcfs.setParam('usd.logging', 0)


if __name__ == '__main__':
    main()
