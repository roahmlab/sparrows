import numpy as np

class ArmWaypoint():
    __slots__ = ['pos', 'vel']
    def __init__(self, pos, vel = None):
        self.pos = pos
        self.vel = vel

class GoalWaypointGenerator():
    def __init__(self, qgoal, enforce_vel_radius=0):
        self.qgoal = qgoal
        self.enforce_vel_radius = enforce_vel_radius
    
    def get_waypoint(self, qpos, qvel, qgoal=None):
        if qgoal is not None:
            self.qgoal = qgoal
        if np.all(np.abs(qpos - self.qgoal) < self.enforce_vel_radius):
            print("Adding velocity to waypoint")
            return ArmWaypoint(self.qgoal, np.zeros_like(qvel))
        else:
            return ArmWaypoint(self.qgoal)
        
class CustomWaypointGenerator():
    def __init__(
        self,
        waypoints,
        qgoal,
        enforce_vel_radius=0
    ):
        self.traj = waypoints
        self.num_waypoints = self.traj.shape[0]
        self.waypoint_i: int = 1 # the 0-index waypoint is the starting configuration
        self.counter: int = 0
        self.enforce_vel_radius = enforce_vel_radius
        self.qgoal = qgoal

    def get_waypoint(self, qpos, qvel):    
        if self.waypoint_i < self.num_waypoints - 1:
            if np.linalg.norm(qpos - self.traj[self.waypoint_i]) < 0.5:
                self.waypoint_i += 1
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= 2:
                    self.counter = 0
                    self.waypoint_i += 1
            if self.waypoint_i >= self.num_waypoints - 1:
                waypoint = self.qgoal
            else:
                waypoint = self.traj[self.waypoint_i]
        else:
            waypoint = self.qgoal
        
        if np.all(np.abs(qpos - waypoint) < self.enforce_vel_radius):
            print("Adding velocity to waypoint")
            return ArmWaypoint(waypoint, np.zeros_like(qvel))
        else:
            return ArmWaypoint(waypoint)
