uint8 UNKNOWN=0
uint8 FOLLOW=1
uint8 LEFT=2
uint8 RIGHT=3
uint8 STRAIGHT=4
uint8 STOP=5

sdf_interfaces/Frame frame
string route_id
geometry_msgs/PoseStamped[] waypoints
uint8[] maneuver
