# G1-01 interface manifest

Schema version: `safedrive.runtime.v1`.

| Interface | SHA-256 |
|---|---|
| `msg/ActorState.msg` | `c0b17bcfa8637c6b18ce70a7f79a85742a5abaae16880949d46cf2debb582116` |
| `msg/ControlCommand.msg` | `f54de58fe1ff8c6db0d7a5fcc7be6d8be1100c7be0fd083bb99dd135698f4050` |
| `msg/EgoState.msg` | `b514b3c1e5635b0f2b495eef43a651c9129ac7619e6820e742c44cca340bd867` |
| `msg/Frame.msg` | `0ec957616a2162a528034fe4511364a193236d91110ddb87c63eeccf3aff20c1` |
| `msg/PolicyDecision.msg` | `2e002deb63102ba6e5dc1fc3e8e3b5f043935c09676d3078c2dbb32c4c04403c` |
| `msg/Route.msg` | `797dcdd2b3ba24c042c2a9de76b64248b826c2ea2a99199de2637932e9613bc7` |
| `msg/RunEvent.msg` | `24cbbb7f2bd2cd249eb16e70348d59b48a3da7f62f1c0bb177f233d5b2c08f1b` |
| `msg/RunIdentity.msg` | `ea8e455cdf210f4627d42d6d81fe24f92f8d28a952093de9aaec5d0b7d7dda7c` |
| `msg/SafetyStatus.msg` | `c5911c262bd3536f94227287621fc1cceae76a1e0e3d4c69cb7557d2205075a4` |
| `msg/Trajectory.msg` | `0084976d4e4c50181a4eb62e405aeae0b26b6c83e9db75148bb9f12a2eac475a` |
| `msg/TrajectoryPoint.msg` | `cc80233422a76ee5f7a8f933e037a29dac8c6accc6670516b964da234fc83fb1` |
| `srv/GetRuntimeProfile.srv` | `80b6d7548dd574349e8db2d56955960fc12e377710761fae46e31c5fc046f078` |

The generated package is `sdf_interfaces`; it has no dependency on the G0
String topic. Formal consumers use these generated types, while the isolated
runtime adapter validates the G0 JSON compatibility boundary.
