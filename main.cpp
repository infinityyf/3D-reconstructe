#include "findFeature.h"
int main() {
	std::vector<cv::Mat> rotateMatrixs;
	std::vector<cv::Mat> transMatrixs;


	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	rotateMatrixs.push_back(cv::Mat::eye(3, 3, CV_64F));
	transMatrixs.push_back(cv::Mat::zeros(3, 1, CV_64F));

	cv::Mat cameraMatrix;

	std::vector<cv::Point3d> points; //记录三维点位置

	//cameraMatrix = getCamera("E:/vs_workspace/VideoCap/result.txt");
	cameraMatrix = getCamera("E:/vs_workspace/CameraCaliberation/test.txt");


	//PicReader("E:\\vs_workspace\\VideoCap\\caliberate", rotateMatrixs, transMatrixs, cameraMatrix, points);

	//usePNP("E:\\vs_workspace\\VideoCap\\caliberate", rotateMatrixs, transMatrixs, cameraMatrix,6,9,28);
	usePNP("E:\\vs_workspace\\VideoCap\\caliwithPC", rotateMatrixs, transMatrixs, cameraMatrix, 6, 9, 28);


	//VideoReader("test2.mp4",rotateMatrixs,transMatrixs, cameraMatrix, points);

	//std::cout << points.size() << std::endl;

	//生成点云并显示
	//showCloud(points, cloud);


	return 0;
}