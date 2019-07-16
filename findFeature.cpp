#include "findFeature.h"


/*给出两个图片，找到每张图片的关键点，和关键点的匹配*/
void findFeatureMatches(
	cv::Mat& img1,
	cv::Mat& img2,
	std::vector<cv::KeyPoint>& KeyPoints1,
	std::vector<cv::KeyPoint>& KeyPoints2,
	std::vector<cv::DMatch>& matches
) {
	//注意这里使用mat 对描述子进行存储
	cv::Mat Descriptor1, Descriptor2;

	//创建orb特征计算
	cv::Ptr<cv::ORB> orb = cv::ORB::create(500,1.2f,8,31,0,2,cv::ORB::HARRIS_SCORE,31,20);

	orb->detect(img1, KeyPoints1);
	orb->detect(img2, KeyPoints2);

	orb->compute(img1, KeyPoints1, Descriptor1);
	orb->compute(img2, KeyPoints2, Descriptor2);

	std::vector<cv::DMatch> firstMatches;
	cv::BFMatcher matcher(cv::NORM_HAMMING);
	matcher.match(Descriptor1, Descriptor2, firstMatches);

	double min_dist = 10000, max_dist = 0;
	for (cv::DMatch match : firstMatches) {
		double dist = match.distance;
		if (dist < min_dist)min_dist = dist;
		if (dist > max_dist)max_dist = dist;
	}

	for (int i = 0; i < Descriptor1.rows; i++) {
		if (firstMatches[i].distance <= cv::max(2 * min_dist, 30.0)) {
			matches.push_back(firstMatches[i]);
		}
	}

	//将结果绘制在原图上
	cv::drawKeypoints(img1, KeyPoints1, img1);
	cv::drawKeypoints(img2, KeyPoints2, img2);
}

void VideoReader(std::string filePath) {
	//读取视频
	cv::VideoCapture input;
	bool result = input.open(filePath);
	assert(result);

	std::cout << input.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;;
	std::cout << input.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
	cv::Mat frame1, frame2;
	std::vector<cv::KeyPoint>KeyPoints1, KeyPoints2;
	std::vector<cv::DMatch> matches;

	input.read(frame1);
	input.read(frame2);
	if (frame1.empty() || frame2.empty()) return ;
	findFeatureMatches(frame1, frame2, KeyPoints1, KeyPoints2, matches);
	cv::cvtColor(frame1, frame1, cv::COLOR_RGB2GRAY);
	cv::imshow("current frame", frame1);
	cv::waitKey(100);

	while (true) {
		KeyPoints1.clear();
		KeyPoints2.clear();
		matches.clear();
		frame1 = frame2;
		input.read(frame2);
		if (frame1.empty() || frame2.empty()) return;
		findFeatureMatches(frame1, frame2, KeyPoints1, KeyPoints2, matches);
		cv::cvtColor(frame1, frame1, cv::COLOR_RGB2GRAY);
		cv::imshow("current frame", frame1);
		cv::waitKey(100);
	}

	//读取摄像头
	//cv::VideoCapture camera;
	//bool result = camera.open(0);
	//assert(result);

	//cv::Mat frame;
	//while (true){
	//	camera.read(frame);
	//	if (frame.empty())break;
	//	cv::cvtColor(frame, frame, cv::COLOR_RGB2GRAY);
	//	cv::imshow("video", frame);
	//	if (cv::waitKey(20) > 0) break;
	//}
}