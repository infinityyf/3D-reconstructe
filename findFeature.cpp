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

void VideoReader(
	std::string filePath,
	std::vector<cv::Mat>& rotateMatrixs,
	std::vector<cv::Mat>& transMatrixs,
	cv::Mat& cameraMatrix,
	std::vector<cv::Point3d>& points) {
	//读取视频
	cv::VideoCapture input;
	bool result = input.open(filePath);
	assert(result);

	std::cout << input.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;;
	std::cout << input.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
	cv::Mat frame1, frame2;
	std::vector<cv::KeyPoint>KeyPoints1, KeyPoints2;
	std::vector<cv::DMatch> matches;

	//存储旋转和位移信息
	cv::Mat R;
	cv::Mat t;

	input.read(frame1);
	input.read(frame2);
	if (frame1.empty() || frame2.empty()) return ;
	findFeatureMatches(frame1, frame2, KeyPoints1, KeyPoints2, matches);
	cv::cvtColor(frame1, frame1, cv::COLOR_RGB2GRAY);
	cv::imshow("current frame", frame1);
	cv::waitKey(100);

	solveRT(cameraMatrix, KeyPoints1, KeyPoints2, matches, R, t);
	trianglePoint(KeyPoints1, KeyPoints2, matches, rotateMatrixs.back(),transMatrixs.back(), R, t, points, cameraMatrix);
	

	//需要更新两个位姿矩阵vector
	rotateMatrixs.push_back(R.clone());
	transMatrixs.push_back(t.clone());

	while (true) {
		KeyPoints1.clear();
		KeyPoints2.clear();
		matches.clear();
		frame1 = frame2.clone();
		input.read(frame2);
		if (frame1.empty() || frame2.empty()) return;
		findFeatureMatches(frame1, frame2, KeyPoints1, KeyPoints2, matches);
		cv::cvtColor(frame1, frame1, cv::COLOR_RGB2GRAY);
		cv::imshow("current frame", frame1);
		cv::waitKey(100);

		solveRT(cameraMatrix, KeyPoints1, KeyPoints2, matches, R, t);
		trianglePoint(KeyPoints1, KeyPoints2, matches,rotateMatrixs.back(), transMatrixs.back(), R, t, points, cameraMatrix);
		

		//需要更新两个位姿矩阵vector
		rotateMatrixs.push_back(R.clone());
		transMatrixs.push_back(t.clone());

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

void solveRT(
	cv::Mat& cameraMatrix,
	std::vector<cv::KeyPoint>& KeyPoints1,
	std::vector<cv::KeyPoint>& KeyPoints2,
	std::vector<cv::DMatch>& matches, 
	cv::Mat& R, 
	cv::Mat& t) {

	std::vector<cv::Point2d> points1, points2;
	for (cv::DMatch match : matches) {
		points1.push_back(KeyPoints1[match.queryIdx].pt);
		points2.push_back(KeyPoints2[match.trainIdx].pt);
	}

	//计算本质矩阵
	cv::Mat essentialMatrix;
	//essentialMatrix = cv::findEssentialMat(points1, points2, cameraMatrix, cv::RANSAC);
	essentialMatrix = cv::findEssentialMat(points1, points2, cameraMatrix.at<double>(1, 1), cv::Point2d(cameraMatrix.at<double>(0, 2), cameraMatrix.at<double>(1, 2)), cv::RANSAC);
	//分解得到RT
	//cv::recoverPose(essentialMatrix, points1, points2, cameraMatrix, R, t);
	cv::recoverPose(essentialMatrix, points1, points2, R, t, cameraMatrix.at<double>(1, 1), cv::Point2d(cameraMatrix.at<double>(0, 2), cameraMatrix.at<double>(1, 2)));

	//std::cout << "r=" << R << std::endl;
	//std::cout << "t=" << t << std::endl;
}

//计算屏幕坐标
cv::Point2d camCoord(
	cv::Point2f& worldCoord, 
	cv::Mat& cameraMatrix) {
	return cv::Point2d(
		(worldCoord.x-cameraMatrix.at<double>(0,2))/cameraMatrix.at<double>(0,0),
		(worldCoord.y-cameraMatrix.at<double>(1,2))/cameraMatrix.at<double>(1,1));
}

void trianglePoint(
	std::vector<cv::KeyPoint>& KeyPoints1,
	std::vector<cv::KeyPoint>& KeyPoints2,
	std::vector<cv::DMatch>& matches,
	cv::Mat& R1,//第一帧的位姿
	cv::Mat& t1,
	cv::Mat& R2,//第二帧相对第一帧的位姿变化
	cv::Mat& t2,
	std::vector<cv::Point3d>& points,
	cv::Mat& cameraMatrix
) {
	
	//第一个 projmat
	cv::Mat projMatrix1 = (cv::Mat_<double>(3, 4) << 
		R1.at<double>(0, 0), R1.at<double>(0, 1), R1.at<double>(0, 2), t1.at<double>(0, 0), 
		R1.at<double>(1, 0), R1.at<double>(1, 1), R1.at<double>(1, 2), t1.at<double>(1, 0), 
		R1.at<double>(2, 0), R1.at<double>(2, 1), R1.at<double>(2, 2), t1.at<double>(2, 0));

	//std::cout << R1 << std::endl;
	//std::cout << R2 << std::endl;
	
	R2 = R2 * R1;
	t2 = t1 + t2;
	//第二个 projmat
	cv::Mat projMatrix2 = (cv::Mat_<double>(3, 4) <<
		R2.at<double>(0, 0), R2.at<double>(0, 1), R2.at<double>(0, 2), t2.at<double>(0, 0),
		R2.at<double>(1, 0), R2.at<double>(1, 1), R2.at<double>(1, 2), t2.at<double>(1, 0),
		R2.at<double>(2, 0), R2.at<double>(2, 1), R2.at<double>(2, 2), t2.at<double>(2, 0));


	std::vector<cv::Point2d> pts1, pts2;//每个点在帧中的坐标
	for (cv::DMatch match : matches) {
		pts1.push_back(camCoord(KeyPoints1[match.queryIdx].pt, cameraMatrix));
		pts2.push_back(camCoord(KeyPoints2[match.trainIdx].pt, cameraMatrix));
	}

	cv::Mat result;
	cv::triangulatePoints(projMatrix1,projMatrix2,pts1,pts2,result);
	for (int i = 0; i < result.cols; i++) {
		cv::Mat p = result.col(i);
		p /= p.at<double>(3,0);
		cv::Point3d point(p.at<double>(0, 0),p.at<double>(1, 0),p.at<double>(2, 0));
		points.push_back(point);
	}
	std::cout << "solve one pair" << std::endl;
}

cv::Mat getCamera(
	std::string filePath
	) {

	std::ifstream fin(filePath);
	//存储来自文件的字符串流
	std::ostringstream buffer;
	buffer << fin.rdbuf();
	std::string matrix = buffer.str();


	std::vector<double> camera;

	char para[50]; //存储参数
	memset(para, 0, 50);
	int top = -1;
	for (int i = 0; i < matrix.length(); i++) {
		//如果是数字
		if ((matrix.at(i) >= 48 && matrix.at(i) <= 57) || matrix.at(i) == '.') {
			para[++top] = matrix.at(i);
		}
		else if (top != -1) {
			//std::string tmp = std::string(para);
			camera.push_back(atof(para));
			memset(para, 0, 50);
			top = -1;
			
		}
		else continue;
	}

	cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
		camera.at(0), camera.at(1), camera.at(2),
		camera.at(3), camera.at(4), camera.at(5),
		camera.at(6), camera.at(7), camera.at(8)
		);
	std::cout << cameraMatrix << std::endl;

	return cameraMatrix;
}

void PicReader(std::string fileParentPath,
	std::vector<cv::Mat>& rotateMatrixs,
	std::vector<cv::Mat>& transMatrixs,
	cv::Mat& cameraMatrix,
	std::vector<cv::Point3d>& points) {

	

	//读取图片序列
	//使用io.h头文件中的 _find_data_t结构体
	//使用_findfirst,_findnext,_findclose三个函数进行操作

	std::string p;
	intptr_t hFile = 0; //文件句柄
	struct _finddata_t fileinfo; //文件信息
	std::vector<std::string> pictures;
	int index = 0;
	//找到第一个文件
	if ((hFile = _findfirst(p.assign(fileParentPath).append("\\*.jpg").c_str(), &fileinfo)) != 0) {
		do {
			std::string path = p.assign(fileParentPath).append("\\").append(fileinfo.name);
			pictures.push_back(path);
		} while (_findnext(hFile, &fileinfo) != -1);
		_findclose(hFile);
	}

	cv::Mat frame1, frame2;
	std::vector<cv::KeyPoint>KeyPoints1, KeyPoints2;
	std::vector<cv::DMatch> matches;

	//存储旋转和位移信息
	cv::Mat R;
	cv::Mat t;

	frame1 = cv::imread(pictures[index++], cv::IMREAD_UNCHANGED);
	frame2 = cv::imread(pictures[index++], cv::IMREAD_UNCHANGED);

	if (frame1.empty() || frame2.empty()) return;
	findFeatureMatches(frame1, frame2, KeyPoints1, KeyPoints2, matches);
	cv::cvtColor(frame1, frame1, cv::COLOR_RGB2GRAY);
	cv::imshow("current frame", frame1);
	cv::waitKey(1);

	solveRT(cameraMatrix, KeyPoints1, KeyPoints2, matches, R, t);
	trianglePoint(KeyPoints1, KeyPoints2, matches, rotateMatrixs.back(), transMatrixs.back(), R, t, points, cameraMatrix);


	//需要更新两个位姿矩阵vector
	rotateMatrixs.push_back(R.clone());
	transMatrixs.push_back(t.clone());

	while (true) {
		if (index >= pictures.size()) break;

		KeyPoints1.clear();
		KeyPoints2.clear();
		matches.clear();
		frame1 = frame2.clone();
		frame2 = cv::imread(pictures[index++], cv::IMREAD_UNCHANGED);


		if (frame1.empty() || frame2.empty()) return;
		findFeatureMatches(frame1, frame2, KeyPoints1, KeyPoints2, matches);
		cv::cvtColor(frame1, frame1, cv::COLOR_RGB2GRAY);
		cv::imshow("current frame", frame1);
		cv::waitKey(100);

		solveRT(cameraMatrix, KeyPoints1, KeyPoints2, matches, R, t);
		trianglePoint(KeyPoints1, KeyPoints2, matches, rotateMatrixs.back(), transMatrixs.back(), R, t, points, cameraMatrix);


		//需要更新两个位姿矩阵vector
		rotateMatrixs.push_back(R.clone());
		transMatrixs.push_back(t.clone());
	}
}
void usePNP(std::string fileParentPath, std::vector<cv::Mat>& rotateMatrixs , std::vector<cv::Mat>& transMatrixs, cv::Mat& cameraMatrix,int height,int width,float size) {
	//读取图片序列
	//使用io.h头文件中的 _find_data_t结构体
	//使用_findfirst,_findnext,_findclose三个函数进行操作

	std::string p;
	intptr_t hFile = 0; //文件句柄
	struct _finddata_t fileinfo; //文件信息
	std::vector<std::string> pictures;
	//找到第一个文件
	if ((hFile = _findfirst(p.assign(fileParentPath).append("\\*.jpg").c_str(), &fileinfo)) != 0) {
		do {
			std::string path = p.assign(fileParentPath).append("\\").append(fileinfo.name);
			pictures.push_back(path);
		} while (_findnext(hFile, &fileinfo) != -1);
		_findclose(hFile);
	}
	cv::Mat frame;


	//存储旋转和位移信息
	cv::Mat R,r;
	cv::Mat t;
	//为pnp准备数据
	//首先获得标定板内角点的世界坐标，这里把标定板所在平面作为Z轴
	//提取角点，得到像素坐标
	cv::Size chessSize(9, 6);
	cv::Size imageSize;

	//获得角点的世界坐标
	//std::vector<std::vector<cv::Point3f>> objectPoints;
	std::vector<cv::Point3f> eachPoints;
	//获得每张图片角点的像素位置
	//std::vector<std::vector<cv::Point2f>> imagePoints;
	std::vector<cv::Point2f> corners;

	std::vector<cv::Point3f> zcorners;
	std::vector<cv::Point2f> zimg;
	for (int k = 0; k < pictures.size(); k++) {
		zcorners.clear();
		eachPoints.clear();
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				eachPoints.push_back(cv::Point3f(i * size, j * size, 0.0));
			}
		}

		frame = cv::imread(pictures[k], cv::IMREAD_UNCHANGED);

		bool result = cv::findChessboardCorners(frame, chessSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
		//如果寻找角点失败
		if (!result) {
			cv::Mat gray;
			cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
			cv::find4QuadCornerSubpix(gray, corners, chessSize);
		}
		cv::solvePnP(eachPoints, corners, cameraMatrix, cv::Mat(), r, t, false, cv::SOLVEPNP_EPNP);
		cv::Rodrigues(r, R);
		rotateMatrixs.push_back(R);
		transMatrixs.push_back(t);


		cv::Point3f point = cv::Point3f(0, 0, 28);
		zcorners.push_back(point);
		cv::projectPoints(zcorners, r, t, cameraMatrix,cv::Mat(), zimg);
		std::cout << zimg[0] << std::endl;
		
		
		cv::line(frame, corners[0], corners[1], cv::Scalar(255, 0, 0), 1, 8, 0);
		cv::line(frame, corners[0], corners[9], cv::Scalar(0, 255, 0), 1, 8, 0);
		cv::line(frame, corners[0], zimg[0], cv::Scalar(0, 0, 255), 1, 8, 0);
		cv::imshow("test", frame);
		cv::waitKey(3000);
	}
	
	

}