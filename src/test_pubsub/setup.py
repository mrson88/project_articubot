from setuptools import find_packages, setup

package_name = 'test_pubsub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mrson',
    maintainer_email='sonk9d@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'host_data_publisher = test_pubsub.host_data_publisher:main',
            'host_data_subscriber = test_pubsub.host_data_subscriber:main',
        ],
    },
)
