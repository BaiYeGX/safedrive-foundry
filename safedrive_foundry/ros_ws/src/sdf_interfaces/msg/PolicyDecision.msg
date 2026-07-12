uint8 UNKNOWN=0
uint8 CLASSIC_EXPERT=1
uint8 VLA=2
uint8 SAFETY_FALLBACK=3

sdf_interfaces/Frame frame
uint8 policy_kind
string policy_id
string candidate_trajectory_id
float32 confidence
float32 uncertainty
uint64 generated_frame
uint64 planned_frame
