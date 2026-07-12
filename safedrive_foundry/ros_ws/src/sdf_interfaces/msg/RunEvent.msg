uint8 STARTED=1
uint8 FRAME_ACCEPTED=2
uint8 POLICY_SELECTED=3
uint8 SAFETY_INTERVENTION=4
uint8 COMPLETED=5
uint8 FAILED=6

sdf_interfaces/Frame frame
uint64 event_sequence
uint8 event_type
string code
string detail_json
