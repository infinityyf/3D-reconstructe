#pragma once
#ifndef FF_H
#define FF_H
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <vector>
#include <math.h>
#include <iostream>

void findFeatureMatches(
	cv::Mat& img1,
	cv::Mat& img2,
	std::vector<cv::KeyPoint>& KeyPoints1,
	std::vector<cv::KeyPoint>& KeyPoints2,
	std::vector<cv::DMatch>& matches
);

void VideoReader(std::string filePath);
#endif // !FF_H
