#include "findFeature.h"
int main() {
	std::vector<cv::Mat> rotateMatrixs;
	std::vector<cv::Mat> transMatrixs;

	rotateMatrixs.push_back(cv::Mat::eye(3, 3, CV_64F));
	transMatrixs.push_back(cv::Mat::zeros(3, 1, CV_64F));

	cv::Mat cameraMatrix;

	std::vector<cv::Point3d> points; //记录三维点位置

	cameraMatrix = getCamera("E:/vs_workspace/CameraCaliberation/test.txt");

	VideoReader("test1.mp4",rotateMatrixs,transMatrixs, cameraMatrix, points);

	std::cout << points.size() << std::endl;


	return 0;
}