#ifndef _TOLDI_H_
#define _TOLDI_H_
#define Pi 3.1415926
#define NULL_POINTID -1
#define TOLDI_NULL_PIXEL 100

typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
typedef pcl::PointXYZ PointInT;

typedef struct
{
    float x;
    float y;
    float z;
} Vertex;

typedef struct
{
    int pointID;
    Vertex x_axis;
    Vertex y_axis;
    Vertex z_axis;
} LRF;

#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <flann/flann.hpp>

//
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

//	Utility functions
bool processCommandLine(int argc, char** argv, std::string &file_cloud, float &support_radius, int &num_voxels, float &smoothing_kernel_width, std::string &file_keypoints, std::string &output_folder);
std::vector<int> readKeypoints(std::string filename);
bool fileExist(const std::string& name);
void saveVector(std::string filename, const std::vector<std::vector<float>> descriptor);
flann::Matrix<float> initializeGridMatrix(const int n, float x_step, float y_step, float z_step);