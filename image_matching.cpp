#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <npy.hpp>
#include <XoshiroCpp.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener.hpp>
#include <libfreenect2/registration.h>
#include <opencv2/core.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#define ASSET_DIR  "/home/shahe/development/Python/minorproject/assets"
using namespace cv;
using namespace cv::xfeatures2d;

struct similarity_transform_params_t{
	std::vector<KeyPoint> keypoints1;
	std::vector<KeyPoint> keypoints2;
	std::vector<DMatch> matches;
	libfreenect2::Frame* depth_fr1;
	libfreenect2::Frame* depth_fr2;
	int iters = 100;
};

std::unique_ptr<libfreenect2::Freenect2Device::IrCameraParams> ir_params;
std::unique_ptr<libfreenect2::Freenect2Device::ColorCameraParams> color_params;
std::unique_ptr<libfreenect2::Registration> reg;

template<typename T>
std::ostream& operator<<(std::ostream& s, const std::vector<T>& data) {
	s << "[ ";
	for(const auto& i : data) {
		s << i << ", ";
	}
	s << "]" << std::endl;
	return s;
}

// Get rid of matches where the depth map is 0
bool filter_match(const std::vector<unsigned long>& depth_shape,
		const std::vector<float>& depth, int x, int y, int radius=7) {
	int found_zero_depth = 0;
	for(int x_offset = -radius; x_offset <= radius; x_offset++) {
		for(int y_offset = -radius; y_offset <= radius; y_offset++) {
			if(depth[(y + y_offset)*depth_shape[1] + x + x_offset] == 0)
				found_zero_depth++;
		}
	}
	if(found_zero_depth)
		return false;
	return true;
}

std::vector<float> get_pairwise_distances(const std::vector<Eigen::Vector3f>& points) {
	std::vector<float> ret;
	if(points.size() != 3) {
		std::cerr << "Incompatible number of points given" << std::endl;
	}
	for(int i = 0; i < points.size(); i++) {
		for(int j = i + 1; j < points.size(); j++) {
			Eigen::Vector3f diff = points[i] - points[j];
			float squared_diff = diff.transpose() * diff;
			float dist = std::sqrt(squared_diff);
			ret.push_back(dist);
		}
	}
	return ret;
}

Eigen::Vector3f load_keypoint_into_vector(const libfreenect2::Frame* depth_fr,
		const KeyPoint& k) {
	Eigen::Vector3f ret;
	int x = (int)k.pt.x;
	int y = (int)k.pt.y;
	reg->getPointXYZ(depth_fr, y, x, ret.data()[0], ret.data()[1], ret.data()[2]);
	return ret;
}

bool approximately_equal(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2) {
	Eigen::Vector3f diff = v1 - v2;
	float sq_diff = diff.transpose() * diff;
	if(sq_diff < 0.1)
		return true;
	return false;
}

Eigen::Matrix4f get_similarity_transform_ransac(const similarity_transform_params_t& params) {
	XoshiroCpp::Xoshiro256PlusPlus rng(12345);
	Eigen::Matrix4f best_transform;
	size_t best_inlier_cnt = 0;
	for(int i = 0; i < params.iters; i++) {
	//Choose 3 random matches as model
		size_t curr_inlier_cnt = 0;
		Eigen::Matrix4f curr_transform;
		int got_model = 0;
		while(!got_model) {
			std::vector<DMatch> model;
			std::sample(params.matches.begin(), params.matches.end(), std::back_inserter(model),
					3, rng);
			//Points from keypoints1
			std::vector<Eigen::Vector3f> points1;
			//Points from keypoints2
			std::vector<Eigen::Vector3f> points2;

			for(int i = 0; i < 3; i++) {
				points1.push_back(load_keypoint_into_vector(params.depth_fr1,
						params.keypoints1[model[i].queryIdx]));
				points2.push_back(load_keypoint_into_vector(params.depth_fr2,
						params.keypoints2[model[i].trainIdx]));
			}
			int all_distances_equal = 1;
			std::vector<float> pairwise_dist1 = get_pairwise_distances(points1);
			std::vector<float> pairwise_dist2 = get_pairwise_distances(points2);
			for(int i = 0; i < pairwise_dist1.size(); i++) {
				if(std::abs(pairwise_dist1[i] - pairwise_dist2[i]) >= 0.01) {
					all_distances_equal = 0;
				}
			}
			if(all_distances_equal){
				got_model = 1;
			}
			Eigen::Matrix3f pointset1;
			pointset1 << points1[0], points1[1], points1[2];
			
			Eigen::Matrix3f pointset2;
			pointset2 << points2[0], points2[1], points2[2];
			
			curr_transform = Eigen::umeyama(pointset1, pointset2, false);
		}
	//Calculate inliner count
		for(size_t i = 0; i < params.matches.size(); i++) {
			Eigen::Vector3f src_pt = load_keypoint_into_vector(params.depth_fr1, params.keypoints1[params.matches[i].queryIdx]);
			Eigen::Vector3f dest_pt = load_keypoint_into_vector(params.depth_fr2, params.keypoints2[params.matches[i].trainIdx]);
			Eigen::Vector4f src_pt_hom;
			src_pt_hom << src_pt, 1;
			Eigen::Vector4f out_pt_hom = curr_transform * src_pt_hom;
			Eigen::Vector3f out_pt = out_pt_hom.head<3>();
			if(approximately_equal(dest_pt, out_pt)) {
				curr_inlier_cnt++;
			}
		}
		if(curr_inlier_cnt > best_inlier_cnt) {
			std::cout << "Better model found!" << std::endl;
			best_inlier_cnt = curr_inlier_cnt;
			best_transform = curr_transform;
		}
	//Rinse and repeat till n iterations where the model with the highest inlier count is kept
	}
	std::cout << best_transform << std::endl;
	std::cout << "Inlier Count: " << best_inlier_cnt << std::endl;
	return best_transform;
}




int main(void) {
	ir_params = std::make_unique<libfreenect2::Freenect2Device::IrCameraParams>();
	color_params = std::make_unique<libfreenect2::Freenect2Device::ColorCameraParams>();
	ir_params->fx = 365.7117919921875;
	ir_params->fy = 365.7117919921875;
	ir_params->cx = 256.5791931152344;
	ir_params->k1 = 0.08907642215490341;
	ir_params->k2 = -0.2707946002483368;
	ir_params->k3 = 0.09664560109376907;
	ir_params->p1 = 0.0;
	ir_params->p2 = 0.0;
	color_params->fx = 1081.3720703125;
	color_params->fy = 1081.3720703125;
	color_params->cx = 959.5;
	color_params->cy = 539.5;
	reg = std::make_unique<libfreenect2::Registration>(*ir_params, *color_params);

	std::vector<unsigned long> fr1_depth_shape {};
	std::vector<float> fr1_depth {};
	bool fr1_fortran_order;
	npy::LoadArrayFromNumpy(ASSET_DIR "/fr1_depth.npy", fr1_depth_shape,
			fr1_fortran_order, fr1_depth);
	libfreenect2::Frame fr1_depth_frame((size_t)fr1_depth_shape[1],
			(size_t)fr1_depth_shape[0], 4, (unsigned char*)fr1_depth.data());
	fr1_depth_frame.format = libfreenect2::Frame::Float;
	cv::Mat fr1_rgb = cv::imread(ASSET_DIR "/fr1_rgb.png", cv::IMREAD_COLOR);


	std::vector<unsigned long> fr2_depth_shape {};
	std::vector<float> fr2_depth {};
	bool fr2_fortran_order;
	npy::LoadArrayFromNumpy(ASSET_DIR "/fr2_depth.npy", fr2_depth_shape,
			fr2_fortran_order, fr2_depth);
	libfreenect2::Frame fr2_depth_frame((size_t)fr2_depth_shape[1],
			(size_t)fr2_depth_shape[0], 4, (unsigned char*)fr2_depth.data());
	fr2_depth_frame.format = libfreenect2::Frame::Float;
	cv::Mat fr2_rgb = cv::imread(ASSET_DIR "/fr2_rgb.png", cv::IMREAD_COLOR);
	
	int minHessian = 400;
	cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	detector->detectAndCompute(fr1_rgb, cv::noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(fr2_rgb, cv::noArray(), keypoints2, descriptors2);
	
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(
			DescriptorMatcher::BRUTEFORCE);
	std::vector<DMatch> matches;
	std::vector<DMatch> filtered_matches;
	matcher->match(descriptors1, descriptors2, matches);

	for(int i = 0; i < matches.size(); i++) {
		int x1 = keypoints1[matches[i].queryIdx].pt.x;
		int y1 = keypoints1[matches[i].queryIdx].pt.y;
		int x2 = keypoints2[matches[i].trainIdx].pt.x;
		int y2 = keypoints2[matches[i].trainIdx].pt.y;

		if(filter_match(fr1_depth_shape, fr1_depth, x1, y1) &&
				filter_match(fr2_depth_shape, fr2_depth, x2, y2)) {
			filtered_matches.push_back(matches[i]);
		} else {
		}
	}
	std::cout << "Filtered matches: " << filtered_matches.size();
	Mat img_matches;
	drawMatches(fr1_rgb, keypoints1, fr2_rgb, keypoints2, filtered_matches, img_matches);
	imshow("Matches", img_matches);
	waitKey();
	similarity_transform_params_t sim_params;
	sim_params.keypoints1 = keypoints1;
	sim_params.keypoints2 = keypoints2;
	sim_params.depth_fr1 = &fr1_depth_frame;
	sim_params.depth_fr2 = &fr2_depth_frame;
	sim_params.matches = filtered_matches;
	get_similarity_transform_ransac(sim_params);


	return 0;
}
#endif
