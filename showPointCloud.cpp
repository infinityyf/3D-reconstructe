#include "findFeature.h"

void showCloud(std::vector<cv::Point3d>& points, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	for (cv::Point3d point : points) {
		pcl::PointXYZ p;
		p.x = point.x;
		p.y = point.y;
		p.z = point.z;
		(cloud->points).push_back(p);
	}
	cloud->is_dense = false;
	pcl::io::savePCDFileBinary("save.pcd", *cloud);
	std::cout << cloud->size() << std::endl;

	pcl::visualization::CloudViewer viewer("viewer");
	viewer.showCloud(cloud);
}