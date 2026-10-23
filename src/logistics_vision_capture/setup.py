from setuptools import setup


package_name = 'logistics_vision_capture'


setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/vision.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='logistics_robot_team',
    maintainer_email='team@example.com',
    description='USB camera capture and chessboard calibration for the logistics robot.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'camera_capture = logistics_vision_capture.camera_capture_node:main',
            'camera_calibrate = logistics_vision_capture.camera_calibrate:main',
            'camera_undistort = logistics_vision_capture.camera_undistort_node:main',
            'number_detect = logistics_vision_capture.number_detect_node:main',
            'ring_detect = logistics_vision_capture.ring_detect_node:main',
            'digit_trainer = logistics_vision_capture.digit_train:main',
            'digit_dataset_capture = logistics_vision_capture.digit_dataset_capture:main',
            'color_sample_capture = logistics_vision_capture.color_sample_capture:main',
            'color_trainer = logistics_vision_capture.color_train:main',
            'material_detect = logistics_vision_capture.material_detect_node:main',
            'material_console = logistics_vision_capture.material_console:main',
        ],
    },
)
