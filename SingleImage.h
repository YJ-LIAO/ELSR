#pragma once
#include <iostream>
#include <vector>
#include<string>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>


void callCrossPt(cv::Mat& lp_Mf, float* lines, int size, float costhre, float dist);

void processImage(cv::Mat CM_Mf, std::vector<cv::Mat> camera_Rts, std::vector<float> cams_focals,
	cv::Mat space_points_Mf, cv::Mat imsizes_Mf_, int i,
	std::vector<std::string> image_names, std::string input_folder,
	Eigen::Vector3d radial, Eigen::Vector2d tangential,
	float costhre, float dist, int inter_support_num, int maxwidth);