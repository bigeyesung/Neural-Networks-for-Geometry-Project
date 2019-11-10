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