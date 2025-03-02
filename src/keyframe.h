/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"
#include "ORBDescriptor.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
#include <tf/transform_datatypes.h>
#include "xfeat.h"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"


#define MIN_LOOP_NUM  15

using namespace Eigen;
using namespace std;
using namespace cv;
using namespace DVision;


class BriefExtractor
{
public:
  virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
  BriefExtractor(const std::string &pattern_file);

  DVision::BRIEF m_brief;
};

class KeyFrame
{
public:
    
	//for xfeat
	KeyFrame(double _time_stamp, int _index,  cv::Mat _image,
			cv::Mat mkpts0_raw_, cv::Mat feats0_raw_, double flow, int _sequence);
	//for orb
	KeyFrame(double _time_stamp, int _index,  cv::Mat _image,std::vector<KeyPoint> keypoints_orb,
	cv::Mat descriptors_orb, double flow, int _sequence);

	KeyFrame(double _time_stamp, int _index,  cv::Mat _image, double flow, int _sequence);



    void undistorImg();
	bool findConnection(KeyFrame* old_kf);

    double computeflow(KeyFrame* old_kf); //wjk added 0226

	void computeWindowBRIEFPoint();
	void computeBRIEFPoint();
	//void extractBrief();
	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);
	bool searchInAera(const BRIEF::bitset window_descriptor,
	                  const std::vector<BRIEF::bitset> &descriptors_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old_norm,
	                  cv::Point2f &best_match,
	                  cv::Point2f &best_match_norm);
	bool searchInAeraORB(const cv::Mat window_descriptor,
	                  const cv::Mat &descriptors_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old_norm,
	                  cv::Point2f &best_match,
	                  cv::Point2f &best_match_norm);
	void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm);

	void searchByORBDes(std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const cv::Mat &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm);

	void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                const std::vector<cv::Point2f> &matched_2d_old_norm,
                                vector<uchar> &status);
	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
	               const std::vector<cv::Point3f> &matched_3d,
	               std::vector<uchar> &status,
	               Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getCameraPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info);
	void R2euler_tf(Matrix3d RR, Eigen::Vector3d& euler);

	Eigen::Vector3d getLoopRelativeT();
	double getLoopRelativeYaw();
	Eigen::Quaterniond getLoopRelativeQ();
	void testprojection();

    Eigen::Vector3d getLoopCamRelativeT();
	Eigen::Quaterniond getLoopCamRelativeQ();
   //ORB
   boost::shared_ptr<ORBdescriptor> currORBDescriptor0_ptr;
   cv::Mat windowOrbDescriptors;
   cv::Mat vOrbDescriptors;
//    XFeat xfeats_;
//    Ptr<ORB> orb1= ORB::create();
   cv::Mat xfeat_des_;
   cv::Mat xfeat_pts_;
   double flow_;

	double time_stamp; 
	int index;
	int local_index;
	int connected_frame_index;
	Eigen::Vector3d vio_T_w_i; 
	Eigen::Matrix3d vio_R_w_i; 
	Eigen::Vector3d T_w_i;
	Eigen::Matrix3d R_w_i;
	Eigen::Vector3d T_w_c_;
	Eigen::Matrix3d R_w_c_;
	Eigen::Matrix<double, 6, 6> covariance_;
	Eigen::Vector3d origin_vio_T;		
	Eigen::Matrix3d origin_vio_R;
	cv::Mat image;
	cv::Mat raw_image;
	cv::Mat thumbnail;
	vector<cv::Point3f> point_3d; 
	vector<cv::Point2f> point_2d_uv;
	vector<cv::Point3f> point_2d_uvd;
	vector<cv::Point2f> point_2d_uv_undistorted_;
	vector<cv::Vec3b> point_2d_uv_color_undistorted_;
	vector<cv::Point2f> point_2d_norm;
	vector<double> point_id;
	vector<cv::KeyPoint> keypoints;
	vector<cv::KeyPoint> keypoints_norm;
	vector<cv::KeyPoint> window_keypoints;
	vector<BRIEF::bitset> brief_descriptors;
	vector<BRIEF::bitset> window_brief_descriptors;
	bool same_to_loop_;
	bool frozen_;
	int last_key_index_;
	int last_key_opt_index_;
	bool has_fast_point;
	int sequence;

	std::vector<cv::KeyPoint> vis_keypoints1_, vis_keypoints2_;
    std::vector<cv::DMatch> vis_matches_ransac_;

	std::vector<KeyPoint> keypoints_orb_;
    cv::Mat descriptors_orb_;

	//for GS
	cv::Mat newCameraMatrix_;
	cv::Mat undistortedImage_;

	cv::Mat color_img_;

	bool has_loop;
	int loop_index = -1;
	double loop_flow = 0.0;
	int loop_index_trans;
	Eigen::Matrix<double, 8, 1 > loop_info;
	Eigen::Matrix<double, 8, 1 > loop_cam_info;
	Eigen::Matrix<double, 8, 1 > loop_info_transpose;
};

