uint8 UNKNOWN=0
uint8 VEHICLE=1
uint8 PEDESTRIAN=2
uint8 CYCLIST=3
uint8 STATIC=4

sdf_interfaces/Frame frame
uint64 actor_id
uint8 actor_type
geometry_msgs/Pose pose
geometry_msgs/Twist twist
geometry_msgs/Accel acceleration
float32 length_m
float32 width_m
float32 position_stddev_m
float32 velocity_stddev_mps
