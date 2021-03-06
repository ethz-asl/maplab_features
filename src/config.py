import rospy

class BaseConfig(object):
    def try_get_param(self, key, default=None):
        rospy.logdebug('[BaseConfig] try_get_param: {key} with default {default}'.format(key=key, default=default))
        return rospy.get_param(key) if rospy.has_param(key) else default

class LidarImageConfig(BaseConfig):
    def __init__(self):
        # Image processing.
        self.close_point = 1.0
        self.far_point = 50.0
        self.min_intensity = 2
        self.max_intensity = 3000
        self.flatness_range = 1
        self.flatness_intensity = 5

        # Input and output.
        self.operation_mode = 'projection'
        self.in_range_image_topic = ''
        self.in_intensity_image_topic = ''
        self.in_pointcloud_topic = '/os_cloud_node/points'
        self.out_image_topic = '/os_cloud_node/images'

        # LiDAR settings.
        self.fov_up = 50.5
        self.fov_down = 47.5
        self.projection_height = 64
        self.projection_width = 1024

        # General settings.
        self.visualize = False
        self.resize_output = True

    def init_from_config(self):
        # Image processing.
        self.close_point = self.try_get_param("~close_point", self.close_point)
        self.far_point = self.try_get_param("~far_point", self.far_point)
        self.min_intensity = self.try_get_param("~min_intensity", self.min_intensity)
        self.max_intensity = self.try_get_param("~max_intensity", self.max_intensity)
        self.flatness_range = self.try_get_param("~flatness_range", self.flatness_range)
        self.flatness_intensity = self.try_get_param("~flatness_intensity", self.flatness_intensity)

        # Input and output.
        self.operation_mode = self.try_get_param("~operation_mode", self.operation_mode)
        self.in_range_image_topic = self.try_get_param("~in_range_image_topic", self.in_range_image_topic)
        self.in_intensity_image_topic = self.try_get_param("~in_intensity_image_topic", self.in_intensity_image_topic)
        self.in_pointcloud_topic = self.try_get_param("~in_pointcloud_topic", self.in_pointcloud_topic)
        self.out_image_topic = self.try_get_param("~out_image_topic", self.out_image_topic)

        # LiDAR settings.
        self.fov_up = self.try_get_param("~fov_up", self.fov_up)
        self.fov_down = self.try_get_param("~fov_down", self.fov_down)
        self.projection_height = self.try_get_param("~projection_height", self.projection_height)
        self.projection_width = self.try_get_param("~projection_widt", self.projection_width)

        # General settings.
        self.visualize = self.try_get_param("~visualize", self.visualize)
        self.resize_output = self.try_get_param("~resize_output", self.resize_output)

class LkConfig(BaseConfig):
    def __init__(self):
        # General settings.
        self.input_topic = '/VersaVIS/cam0/image_raw'
        self.output_topic = '/VersaVIS/cam0/features'

        # Feature extraction settings.
        self.feature_extraction = 'cv'
        self.debug_feature_extraction = False

        # OpenCV settings.
        self.cv_feature_detector = 'sift'
        self.cv_feature_descriptor = 'freak'

        # SURF settings.
        self.surf_hessian_threshold = 400
        self.surf_n_octaves = 4
        self.surf_n_octaves_layers = 3

    def init_from_config(self):
        # General settings.
        self.input_topic = self.try_get_param("~input_topic", self.input_topic)
        self.output_topic = self.try_get_param("~output_topic", self.output_topic)

        # Feature extraction settings.
        self.feature_extraction = self.try_get_param("~feature_extraction", self.feature_extraction)
        self.debug_feature_extraction = self.try_get_param("~debug_extraction", self.debug_feature_extraction)

        # OpenCV settings.
        self.cv_feature_detector = self.try_get_param("~cv_detector_type", self.cv_feature_detector)
        self.cv_feature_descriptor = self.try_get_param("~cv_descriptor_type", self.cv_feature_descriptor)

        # SURF settings.
        self.surf_hessian_threshold = self.try_get_param("~surf_hessian_threshold", self.surf_hessian_threshold)
        self.surf_n_octaves = self.try_get_param("~surf_n_octaves", self.surf_n_octaves)
        self.surf_n_octaves_layers = self.try_get_param("~surf_n_octaves_layers", self.surf_n_octaves_layers)
