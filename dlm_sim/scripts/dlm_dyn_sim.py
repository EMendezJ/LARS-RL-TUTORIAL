#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import JointState

class DLM:

  def __init__(self):

    #Set the parameters of the DLM
    self._m1 = rospy.get_param("~link1_mass",1.5)
    self._m2 = rospy.get_param("~link2_mass",1.5)
    self._M1 = rospy.get_param("~link1_end_mass",1.0)
    self._M2 = rospy.get_param("~link2_end_mass",2.0)

    self._l1 = rospy.get_param("~link1_length",0.4)
    self._a = rospy.get_param("~link1_COM_pos",0.2)
    self._l2 = rospy.get_param("~link2_length",0.4)    
    self._d = rospy.get_param("~link2_COM_pos",0.2)

    self._coulombCoef1 = rospy.get_param("~coulomb_coeficient1",0.1)
    self._coulombCoef2 = rospy.get_param("~coulomb_coeficient2",0.1)
    self._viscousCoef1 = rospy.get_param("~viscous_coeficient1",0.01)
    self._viscousCoef2 = rospy.get_param("~viscous_coeficient2",0.01)

    #Set the parameters of the simulation
    self._q1 = rospy.get_param("~q1_init_angle",0.10)
    self._q2 = rospy.get_param("~q2_init_angle",0.1)
    self._q1_dot = rospy.get_param("~q1_dot_0",0.01)
    self._q2_dot = rospy.get_param("~q2_dot_0",0.01)
    self._tau1 = rospy.get_param("~tau1_0",0.0)
    self._tau2 = rospy.get_param("~tau2_0",0.0)
    
    #Set the parameters of the simulation environment
    self.sample_time = rospy.get_param("~sample_time",0.02)
    self._gravity = rospy.get_param("~gravity",9.8)

    # Setup Variables to be used
    self._b = self._l1 - self._a
    self._e = self._l2 - self._d

    self.q = np.array([[self._q1], [self._q2]])
    self.q_dot = np.array([[self._q1_dot], [self._q2_dot]])
    self.q_dot_dot = np.array([[0], [0]])
       
    self.first = True
    self.start_time = 0.0
    self.current_time = 0.0
    self.last_time = 0.0
    self.proc_output = 0.0
    self.tau_input = np.array([[0], [0]])
    
    # Declare the input Message
    self.tau = Float32MultiArray()
    
    # Declare the  process output message
    self.robotJoints = JointState()
    self.robotJoints.header.frame_id = "base"
    self.robotJoints.header.stamp = rospy.Time.now()
    self.robotJoints.name.extend(["joint2", "joint3"])
    self.robotJoints.position.extend([0.0, 0.0])
    self.robotJoints.velocity.extend([0.0, 0.0])
    self.robotJoints.effort.extend([0.0, 0.0])
 
    # Setup the Subscribers
    rospy.Subscriber("dlm_input",Float32MultiArray,self.input_callback)

    #Setup the publishers
    self.state_pub = rospy.Publisher("joint_states", JointState, queue_size=1)

  #Define the callback functions
  def input_callback(self,msg):
    self.tau = msg
    self.tau_input = np.array([[self.tau.data[0]], [self.tau.data[1]]])

  #Define the main RUN function
  def run (self):
    #Variable setup
    if self.first == True:
      self.start_time = rospy.get_time() 
      self.last_time = rospy.get_time()
      self.current_time = rospy.get_time()
      self.first = False
  #System
    else:
      #Define sampling time
      self.current_time = rospy.get_time()
      dt = self.current_time - self.last_time
   
      #Dynamical System Simulation
      if dt >= self.sample_time:  

        self.q += self.q_dot * dt 

        self.M = np.array([[self._M1*np.power(self._l1,2) + self._m1*np.power(self._a,2) + self._M2*(np.power(self._l1,2)+2*np.cos(self.q[1,0])*self._l1*self._l2+np.power(self._l2,2)) + self._m2*(np.power(self._d,2)+2*np.cos(self.q[1,0])*self._l1*self._d+np.power(self._l1,2)), 
                            self._M2 * self._l2 * (self._l2 + self._l1 * np.cos(self.q[1,0])) + self._d * self._m2 * (self._d + self._l1*np.cos(self.q[1,0]))],
                          [self._M2 * self._l2 * (self._l2 + self._l1 * np.cos(self.q[1,0])) + self._d * self._m2 * (self._d + self._l1*np.cos(self.q[1,0])), self._m2*np.power(self._d,2) + self._M2*np.power(self._l2,2)]])
      

        self.C = np.array([[-2*self._l1*self.q_dot[1,0]*np.sin(self.q[1,0])*(self._M2*self._l2+self._d*self._m2), -self._l1* self.q_dot[1,0]*np.sin(self.q[1,0])*(self._M2*self._l2+self._d*self._m2)],
                          [self._l1* self.q_dot[0,0]*np.sin(self.q[1,0])*(self._M2*self._l2+self._d*self._m2), 0.0]])
        
      
        self.G = np.array([[self._m1 * self._a * np.cos(self.q[0,0]) + self._M1 * self._l1 * np.cos(self.q[0,0]) + self._m2 * (self._l1 * np.cos(self.q[0,0]) + self._d * np.cos(self.q[0,0]+self.q[1,0])) + self._M2 * (self._l1 * np.cos(self.q[0,0]) + self._l2 * np.cos(self.q[0,0]+self.q[1,0]))],
                          [(self._m2 * self._d + self._M2 * self._l2) * np.cos(self.q[0,0]+self.q[1,0])]]) * self._gravity         

        self.F = np.array([[self._viscousCoef1 * self.q_dot[0,0] + self._coulombCoef1 * np.sign(self.q_dot[0,0])],[self._viscousCoef2 * self.q_dot[1,0] + self._coulombCoef2 * np.sign(self.q_dot[1,0])]])       

        self.q_dot_dot = np.dot(np.linalg.inv(self.M), (self.tau_input - self.G - np.dot(self.C, self.q_dot) - self.F))
        self.q_dot += self.q_dot_dot * dt

        #Message to publish
        self.robotJoints.header.stamp = rospy.Time.now()
        self.robotJoints.position = self.q
        self.robotJoints.velocity = self.q_dot

        #Publish message
        self.state_pub.publish(self.robotJoints)
        self.last_time = rospy.get_time()
      
  #wrap to pi function
  def wrap_to_Pi(self,theta):
      result = np.fmod((theta + np.pi),(2 * np.pi))
      if(result < 0):
          result += 2 * np.pi
      return result - np.pi

 #Stop Condition
  def stop(self):
  #Setup the stop message (can be the same as the control message)
    print("Stopping")
    total_time = rospy.get_time()-self.start_time
    rospy.loginfo("Total Simulation Time = %f" % total_time)



if __name__=='__main__':

 #Initialise and Setup node
 rospy.init_node("DLM_Sim")
 DLM = DLM()
 
 # Configure the Node
 loop_rate = rospy.Rate(rospy.get_param("~node_rate",200))
 rospy.on_shutdown(DLM.stop)

 print("The DLM simulation is Running")

 try:
  #Run the node
  while not rospy.is_shutdown(): 
   DLM.run()
   loop_rate.sleep()
 
 except rospy.ROSInterruptException:
  pass 