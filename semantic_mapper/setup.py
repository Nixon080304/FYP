from setuptools import find_packages, setup

package_name = 'semantic_mapper'

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
    maintainer='nixon',
    maintainer_email='nixonedwardwinata2004@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'semantic_mapper_node = semantic_mapper.semantic_mapper_node:main',
        'object_detector_node = semantic_mapper.object_detector_node:main',
        'semantic_projection_node = semantic_mapper.semantic_projection_node:main',
        'semantic_map_node = semantic_mapper.semantic_map_node:main',
        'language_navigation_node = semantic_mapper.language_navigation_node:main',
        ],
    },
)
