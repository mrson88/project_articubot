from setuptools import find_packages, setup
from glob import glob
package_name = 'robot_speech_to_text'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, "submodules"],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*launch.py')),
        ('share/' + package_name + '/submodules', glob('submodules/*.ini')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mrson',
    maintainer_email='mrson@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'speech_to_text_node = robot_speech_to_text.speech_to_text_node:main',
            'speech_to_text_whisper_node = robot_speech_to_text.speech_to_text_whisper_node:main',
        ],
    },
)
