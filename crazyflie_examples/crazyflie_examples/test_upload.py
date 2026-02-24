#!/usr/bin/env python3

from pathlib import Path
from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory
import time

def main():
    # Load a small trajectory
    traj = Trajectory()
    traj.loadcsv(Path(__file__).parent / "data/traj10b.csv")  # Keep it <10 segments

    # Start Crazyswarm
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    print("Sleeping 2s for full connection...")
    timeHelper.sleep(2.0)  # Ensure Crazyflie is initialized

    # Upload and verify trajectory
    try:
        cf = swarm.allcfs.crazyflies[0]
        print("Uploading trajectory...")
        cf.uploadTrajectory(0, 0, traj)
        print("Upload complete!")

        print("Taking off...")
        cf.takeoff(1.0, 2.0)
        timeHelper.sleep(3.0)

        print("Executing trajectory...")
        cf.startTrajectory(0, timescale=1.0)
        timeHelper.sleep(traj.duration + 2.0)

        print("Landing...")
        cf.land(0.06, 2.0)
        timeHelper.sleep(3.0)
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == '__main__':
    main()

