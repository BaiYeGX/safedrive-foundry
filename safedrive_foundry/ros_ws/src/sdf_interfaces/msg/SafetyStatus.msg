uint8 UNKNOWN=0
uint8 CLEAR=1
uint8 CAUTION=2
uint8 INTERVENE=3
uint8 EMERGENCY=4

sdf_interfaces/Frame frame
uint8 level
bool oracle_observation
float32 min_ttc_s
float32 max_constraint_slack
float32 risk_upper_bound
string reason_code
