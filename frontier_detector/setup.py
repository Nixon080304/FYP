from setuptools import setup

package_name = 'frontier_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nixon',
    maintainer_email='nixon@todo.todo',
    description='Frontier detection package for active exploration',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'frontier_detector_node = frontier_detector.frontier_detector_node:main',
        ],
    },
)