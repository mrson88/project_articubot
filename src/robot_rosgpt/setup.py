from setuptools import setup
from glob import glob
import os

package_name = 'robot_rosgpt'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'webapp'), glob('webapp/*')),
        ('share/' + package_name + '/launch', glob('launch/*launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Anis Koubaa',
    author_email='anis.koubaa@gmail.com',
    maintainer='Anis Koubaa',
    maintainer_email='anis.koubaa@gmail.com',
    keywords=['ROS', 'ChatGPT'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Creative Commons Attribution-NonCommercial License (CC BY-NC)',
        'Programming Language :: Python :: 3.10', #could work with other version. Tested with 3.10
    ],
    description='A ROS2 package for processing and executing unstructured textual commands using ChatGPT in human-robot interaction scenarios.',
    license='Creative Commons Attribution-NonCommercial (CC BY-NC)',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rosgpt = rosgpt.rosgpt:main',
            'rosgptparser_turtlesim = robot_rosgpt.rosgptparser_turtlesim:main',
            'rosgpt_client_node = robot_rosgpt.rosgpt_client_node:main',
            'robot_navigation_node = robot_rosgpt.rosgpt_nav:main',
        ],
    },
)
