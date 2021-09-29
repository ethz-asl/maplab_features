import rospy

class BaseConfig(object):
    def try_get_param(self, key, default=None):
        rospy.logdebug('[BaseConfig] try_get_param: {key} with default {default}'.format(key=key, default=default))
        return rospy.get_param(key) if rospy.has_param(key) else default

class LidarImageConfig(BaseConfig):
    def __init__(self):
        # image processing
        self.close_point = 1.0
        self.far_point = 50.0
        self.min_intensity = 2
        self.max_intensity = 3000
        self.flatness_range = 1
        self.flatness_intensity = 5

        # Input and output
        self.in_pointcloud_topic = '/os_cloud_node/points'
        self.out_image_topic = '/os_cloud_node/images'

        # LiDAR Settings
        self.fov_up = 50.5
        self.fov_down = 47.5
        self.projection_height = 64
        self.projection_width = 1024

        # Debugging
        self.visualize = False

    def init_from_config(self):
        # image processing
        self.close_point = self.try_get_param("~close_point", self.close_point)
        self.far_point = self.try_get_param("~far_point", self.far_point)
        self.min_intensity = self.try_get_param("~min_intensity", self.min_intensity)
        self.max_intensity = self.try_get_param("~max_intensity", self.max_intensity)
        self.flatness_range = self.try_get_param("~flatness_range", self.flatness_range)
        self.flatness_intensity = self.try_get_param("~flatness_intensity", self.flatness_intensity)

        # input and output
        self.in_pointcloud_topic = self.try_get_param("~in_pointcloud_topic", self.in_pointcloud_topic)
        self.out_image_topic = self.try_get_param("~out_image_topic", self.out_image_topic)

        # LiDAR Settings
        self.fov_up = self.try_get_param("~fov_up", self.fov_up)
        self.fov_down = self.try_get_param("~fov_down", self.fov_down)
        self.projection_height = self.try_get_param("~projection_height", self.projection_height)
        self.projection_width = self.try_get_param("~projection_widt", self.projection_width)

        # Debugging
        self.visualize = self.try_get_param("~visualize", self.visualize)