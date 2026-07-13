# Optional: check Windows conda env carla0916 (not used for G3 VLA)
Write-Host "=== Windows conda envs ==="
conda env list
Write-Host "=== carla0916 probe ==="
conda run -n carla0916 python -c "import sys; print(sys.executable); print(sys.version); import carla; print('carla', getattr(carla,'__file__',ok))"
