from setuptools import setup


package_name = 'logistics_ring_v2'


setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md', 'RING_LATENCY_UPDATE.md',
                                  'RING_REFERENCE_UPDATE.md']),
        ('share/' + package_name + '/launch', [
            'launch/ring2.launch.py', 'launch/material2.launch.py',
            'launch/vision_all.launch.py',
        ]),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='logistics_robot_team',
    maintainer_email='team@example.com',
    description='Ring group detection pipeline v2 for the logistics robot.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'ring_pipeline = logistics_ring_v2.ring_pipeline_node:main',
            'mode_gate = logistics_ring_v2.mode_gate_node:main',
            'gray_node = logistics_ring_v2.gray_node:main',
            'blur_node = logistics_ring_v2.blur_node:main',
            'clahe_node = logistics_ring_v2.clahe_node:main',
            'hough_node = logistics_ring_v2.hough_node:main',
            'posterior_node = logistics_ring_v2.posterior_node:main',
            'validate_node = logistics_ring_v2.validate_node:main',
            'material_all_detect = logistics_ring_v2.material_all_detect_node:main',
            'material_track_detect = logistics_ring_v2.material_track_detect_node:main',
            'material2_sample_capture = logistics_ring_v2.material2_sample_capture:main',
            'material2_color_train = logistics_ring_v2.material2_color_train:main',
            'turntable_detect = logistics_ring_v2.turntable_detect_node:main',
            'stacked_ring_detect = logistics_ring_v2.stacked_ring_detect_node:main',
            'debug_display = logistics_ring_v2.debug_display_node:main',
            'ring_serial_bridge = logistics_ring_v2.ring_serial_bridge_node:main',
        ],
    },
)
