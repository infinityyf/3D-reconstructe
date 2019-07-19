#pragma once
#ifndef FF_H
#define FF_H
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv.hpp>
#include <vector>
#include <math.h>
#include <iostream>
#include <regex>
#include <fstream>

void findFeatureMatches(
	cv::Mat& img1,
	cv::Mat& img2,
	std::vector<cv::KeyPoint>& KeyPoints1,
	std::vector<cv::KeyPoint>& KeyPoints2,
	std::vector<cv::DMatch>& matches
);

void VideoReader(std::string filePath, 
	std::vector<cv::Mat>&, 
	std::vector<cv::Mat>&, 
	cv::Mat&, 
	std::vector<cv::Point3d>&);

void solveRT(
	cv::Mat& cameraMatrix,
	std::vector<cv::KeyPoint>& KeyPoints1,
	std::vector<cv::KeyPoint>& KeyPoints2, 
	std::vector<cv::DMatch>& matches,
	cv::Mat& R,
	cv::Mat& t);

void trianglePoint(
	std::vector<cv::KeyPoint>& KeyPoints1,
	std::vector<cv::KeyPoint>& KeyPoints2,
	std::vector<cv::DMatch>& matches,
	cv::Mat& R1,
	cv::Mat& t1,
	cv::Mat& R2,
	cv::Mat& t2,
	std::vector<cv::Point3d>& points,
	cv::Mat& cameraMatrix
);


cv::Mat getCamera(std::string filePath);

cv::Point2d camCoord(cv::Point3d&, cv::Mat&);
#endif // !FF_H
