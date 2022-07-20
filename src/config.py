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
        self.in_pointcloud_topic = '/os_cloud_node/points'
        self.out_image_topic = '/os_cloud_node/images'

        # LiDAR settings.
        self.projection_height = 64
        self.projection_width = 1024
        self.lidar_calibration = ''

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
        self.in_pointcloud_topic = self.try_get_param("~in_pointcloud_topic", self.in_pointcloud_topic)
        self.out_image_topic = self.try_get_param("~out_image_topic", self.out_image_topic)

        # LiDAR settings.
        self.projection_height = self.try_get_param("~projection_height", self.projection_height)
        self.projection_width = self.try_get_param("~projection_width", self.projection_width)
        self.lidar_calibration = self.try_get_param("~lidar_calibration", self.projection_width)

        # General settings.
        self.visualize = self.try_get_param("~visualize", self.visualize)
        self.resize_output = self.try_get_param("~resize_output", self.resize_output)

class MainConfig(BaseConfig):
    def __init__(self):
        # General settings.
        self.input_topic = ''
        self.output_topic = ''
        self.resize_input_image = 640
        self.debug_detections = False
        self.debug_tracking = False

        # Feature extraction settings.
        self.feature_extraction = 'cv'       # cv, external
        self.cv_feature_detector = 'sift'    # surf, sift
        self.cv_feature_descriptor = 'freak' # freak, brief, sift, surf
        # Do not initialize new features closer than a minimum distnace to
        # the image border. This does not prevent existing features from being
        # tracked there though.
        self.min_distance_to_image_border = 30
        # Do not initialize new feature tracks if they are closer than this
        # threshold to an existing feature track.
        self.mask_redetections_thr_px = 7
        self.pca_descriptors = False
        self.pca_pickle_path = ''

        # SURF settings.
        self.surf_hessian_threshold = 300
        self.surf_n_octaves = 4
        self.surf_n_octaves_layers = 3

        # Feature tracking settings.
        self.feature_tracking = 'lk'   # lk, external
        self.num_tracked_points = 600

        # Lucas Kanade (LK) tracker settings.
        self.lk_window_size = 15
        self.lk_max_level = 2
        self.lk_stop_criteria_eps = 0.01
        self.lk_stop_criteria_steps = 20
        self.lk_max_step_error_px = 2.0
        self.lk_merge_tracks_thr_px = 3

    def init_from_config(self):
        # General settings.
        self.input_topic = self.try_get_param(
            "~input_topic", self.input_topic)
        self.output_topic = self.try_get_param(
            "~output_topic", self.output_topic)
        self.input_topic = [t.strip() for t in self.input_topic.split(',')]
        self.output_topic = [t.strip() for t in self.output_topic.split(',')]
        self.resize_input_image = self.try_get_param(
            "~resize_input_image", self.resize_input_image)
        self.debug_detections = self.try_get_param(
            "~debug_detections", self.debug_detections)
        self.debug_tracking = self.try_get_param(
            "~debug_tracking", self.debug_tracking)

        # Feature extraction settings.
        self.feature_extraction = self.try_get_param(
            "~feature_extraction", self.feature_extraction)
        self.cv_feature_detector = self.try_get_param(
            "~cv_feature_detector", self.cv_feature_detector)
        self.cv_feature_descriptor = self.try_get_param(
            "~cv_feature_descriptor", self.cv_feature_descriptor)
        self.min_distance_to_image_border = self.try_get_param(
            "~min_distance_to_image_border", self.min_distance_to_image_border)
        self.mask_redetections_thr_px = self.try_get_param(
            "~mask_redetections_thr_px", self.mask_redetections_thr_px)
        self.pca_descriptors = self.try_get_param(
            "~pca_descriptors", self.pca_descriptors)
        self.pca_pickle_path = self.try_get_param(
            "~pca_pickle_path", self.pca_pickle_path)

        # SURF settings.
        self.surf_hessian_threshold = self.try_get_param(
            "~surf_hessian_threshold", self.surf_hessian_threshold)
        self.surf_n_octaves = self.try_get_param(
            "~surf_n_octaves", self.surf_n_octaves)
        self.surf_n_octaves_layers = self.try_get_param(
            "~surf_n_octaves_layers", self.surf_n_octaves_layers)

        # Feature tracking settings.
        self.feature_tracking = self.try_get_param(
            "~feature_tracking", self.feature_tracking)
        self.num_tracked_points = self.try_get_param(
            "~num_tracked_points", self.num_tracked_points)

        # Lucas Kanade (LK) tracker settings.
        self.lk_window_size = self.try_get_param(
            "~lk_window_size", self.lk_window_size)
        self.lk_max_level = self.try_get_param(
            "~lk_max_level", self.lk_max_level)
        self.lk_stop_criteria_eps = self.try_get_param(
            "~lk_stop_criteria_eps", self.lk_stop_criteria_eps)
        self.lk_stop_criteria_steps = self.try_get_param(
            "~lk_stop_criteria_steps", self.lk_stop_criteria_steps)
        self.lk_max_step_error_px = self.try_get_param(
            "~lk_max_step_error_px", self.lk_max_step_error_px)
        self.lk_merge_tracks_thr_px = self.try_get_param(
            "~lk_merge_tracks_thr_px", self.lk_merge_tracks_thr_px)
