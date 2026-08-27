@echo off
echo ===================================================
echo   Starting CARLA 0.9.16 on Town05
echo ===================================================
cd /d E:\CARLA_0.9.16
start CarlaUE4.exe /Game/Carla/Maps/Town05 -dx11 -carla-rpc-port=2000
echo CARLA started! Keep this window or switch to CARLA rendering.
