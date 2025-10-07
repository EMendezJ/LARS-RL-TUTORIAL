#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension
import numpy as np
from sensor_msgs.msg import JointState

# Global variables
system_states = JointState()
flag = False
start_time = 0
first = True

# Robot parameters for STRICT 2-link system
m1 = 1.5    # Mass of link 1 ONLY
m2 = 1.5    # Mass of link 2 ONLY
l1 = 0.4    # Length of link 1
l2 = 0.4    # Length of link 2
a = 0.2     # COM position of link 1 (from joint 1)
d = 0.2     # COM position of link 2 (from joint 2)
gravity = 9.8

def callback(data):
    global system_states, flag
    flag = True
    system_states = data

def stop():
    print("Stopping controller")

def compute_M(q2):
    """Inertia matrix for PURE 2-link system"""
    M11 = m1*a**2 + m2*(l1**2 + d**2 + 2*l1*d*np.cos(q2))
    M12 = m2*(d**2 + l1*d*np.cos(q2))
    M21 = M12
    M22 = m2*d**2
    return np.array([[M11, M12], [M21, M22]])

def compute_C(q2, q1_dot, q2_dot):
    """Coriolis matrix for PURE 2-link system"""
    h = -m2*l1*d*np.sin(q2)
    C11 = h*(2*q1_dot + q2_dot)
    C12 = h*q2_dot
    C21 = -h*q1_dot
    C22 = 0.0
    return np.array([[C11, C12], [C21, C22]])

def compute_G(q1, q2):
    """Gravity vector for PURE 2-link system"""
    G1 = (m1*a + m2*l1)*gravity*np.cos(q1) + m2*d*gravity*np.cos(q1 + q2)
    G2 = m2*d*gravity*np.cos(q1 + q2)
    return np.array([G1, G2])

if __name__=='__main__':
    rospy.init_node("DLM_Controller")
    loop_rate = rospy.Rate(200)
    
    # Subscribers and Publishers
    rospy.Subscriber("joint_states", JointState, callback)
    controlInput = rospy.Publisher("dlm_input", Float32MultiArray, queue_size=1)
    setpoint_pub = rospy.Publisher("joint_setpoints", JointState, queue_size=1)
    
    # Message declarations
    u = Float32MultiArray()
    u.layout.dim.append(MultiArrayDimension(label="width", size=1))
    u.layout.dim.append(MultiArrayDimension(label="height", size=2))
    
    setpoint_msg = JointState()
    setpoint_msg.name = ["q1_setpoint", "q2_setpoint"]
    setpoint_msg.position = [0.0, 0.0]
    setpoint_msg.velocity = [0.0, 0.0]
    setpoint_msg.effort = [0.0, 0.0]
    
    # Controller gains
    Kp = np.diag([16.0, 10.0])
    Kd = np.diag([1.0, 1.0])    
    max_torque = 50.0
    
    try:
        while not rospy.is_shutdown():
            if flag:
                current_time = rospy.get_time()
                if first:
                    start_time = current_time
                    first = False
                
                dt = current_time - start_time
                
                # Trajectory generation
                r1 = 1 + 0.3 * np.sin(dt)
                r2 = 0.5 + 0.7 * np.sin(dt)
                r1_dot = 0.3 * np.cos(dt)
                r2_dot = 0.7 * np.cos(dt)
                r1_ddot = -0.3 * np.sin(dt)
                r2_ddot = -0.7 * np.sin(dt)
                
                # Current state
                q1 = system_states.position[0]
                q2 = system_states.position[1]
                q1_dot = system_states.velocity[0]
                q2_dot = system_states.velocity[1]
                
                # Control law
                e = np.array([r1 - q1, r2 - q2])
                e_dot = np.array([r1_dot - q1_dot, r2_dot - q2_dot])
                q_ddot_desired = np.array([r1_ddot, r2_ddot]) + Kd @ e_dot + Kp @ e
                
                M = compute_M(q2)
                C = compute_C(q2, q1_dot, q2_dot)
                G = compute_G(q1, q2)
                
                tau = np.dot(M, q_ddot_desired) + np.dot(C, np.array([q1_dot, q2_dot])) + G
                tau = np.clip(tau, -max_torque, max_torque)
                
                # Publish
                u.data = tau.tolist()
                setpoint_msg.header.stamp = rospy.Time.now()
                setpoint_msg.position = [r1, r2]
                setpoint_msg.velocity = [r1_dot, r2_dot]
                setpoint_msg.effort = q_ddot_desired.tolist()
                
                controlInput.publish(u)
                setpoint_pub.publish(setpoint_msg)
 
            loop_rate.sleep() 
            
    except rospy.ROSInterruptException:
        pass