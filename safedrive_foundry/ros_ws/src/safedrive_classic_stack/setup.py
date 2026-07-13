from setuptools import setup

package_name = "safedrive_classic_stack"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SafeDrive Foundry",
    maintainer_email="devnull@example.com",
    description="Classic map/route/behavior ROS adapters for SafeDrive Foundry G1-03",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={},
)
