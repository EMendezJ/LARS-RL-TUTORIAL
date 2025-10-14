
1. open isaacsim
2. open lite6_on_table.usd

### Interact with simulated robot

```
ros2 topic pub /joint_command sensor_msgs/msg/JointState "{name: {joint1, joint2, joint3, joint4, joint5, joint6}, position: {3.14, 0.51, 0.5, 1,-0.02,-0.03}, velocity: {0,1,2,3,4,5}, effort: {0,1,2,3,4,5}}"

```
