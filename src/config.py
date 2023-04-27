import rospy

class BaseConfig(object):
    def try_get_param(self, key, default=None):
        rospy.logdebug('[BaseConfig] try_get_param: {key} with default {default}'.format(key=key, default=default))
        return rospy.get_param(key) if rospy.has_param(key) else default

class MainConfig(BaseConfig):
    def __init__(self):
        # General settings.
        self.input_topic = ''
        self.output_topic = ''
        self.resize_input_image = 1.0
        self.debug_detections = False
        self.debug_tracking = False

        # Feature extraction settings.
        self.feature_extraction = 'cv'       # cv, external
        self.cv_feature_detector = ''        # surf, sift
        self.cv_feature_descriptor = ''      # freak, brief, sift, surf
        # Do not initialize new features closer than a minimum distnace to
        # the image border. This does not prevent existing features from being
        # tracked there though.
        self.min_distance_to_image_border = 30
        # Do not initialize new feature tracks if they are closer than this
        # threshold to an existing feature track.
        self.mask_redetections_thr_px = 7
        self.pca_descriptors = False
        self.pca_matrix_path = ''

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
        assert(len(self.input_topic) == len(self.output_topic))

        self.resize_input_image = self.try_get_param(
            "~resize_input_image", self.resize_input_image)
        assert(self.resize_input_image > 0 and self.resize_input_image <= 1.0)
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
        self.pca_matrix_path = self.try_get_param(
            "~pca_matrix_path", self.pca_matrix_path)

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
