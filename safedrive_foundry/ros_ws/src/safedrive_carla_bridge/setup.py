from setuptools import find_packages, setup

package_name = "safedrive_carla_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "sdf = safedrive_carla_bridge.cli:main",
            "carla_status_bridge = safedrive_carla_bridge.bridge_node:main",
            "carla_sync_driver = safedrive_carla_bridge.sync_driver:main",
        ],
    },
)
