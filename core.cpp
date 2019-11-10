#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <omp.h>
#include "core.h"


namespace po = boost::program_options;

// Processes command line arguments and returns the default values if the parameters are not specified by the user
bool processCommandLine(int argc, char** argv,
	std::string &file_cloud,
	float &support_radius,
	int &num_voxels,
	float &smoothing_kernel_width,
	std::string &file_keypoints,
	std::string &output_folder)

    {
	try
	{
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "Given a point cloud and a support radius, this code generates a SDV voxel representation for the selected interest points.")
			("fileCloud,f", po::value<std::string>(&file_cloud)->required(), "Input point cloud file in .ply format")
			("supportRadius,r", po::value<float>(&support_radius)->default_value(0.150),
				"Half size of the voxel grid.")
			("numVoxels,n", po::value<int>(&num_voxels)->default_value(16),
				"Number of voxels in a side of the grid. Whole grid is nxnxn.")
			("smoothingKernelWidth,h", po::value<float>(&smoothing_kernel_width)->default_value(1.75),
				"Width of the Gaussia kernel used for smoothing.")
			("fileKeypoints,k", po::value<std::string>(&file_keypoints)->default_value("0"),
				"Path to the file with the indices of the interest points. If 0, SDV voxel grid representation if computed for all the points")
			("outputFolder,o", po::value<std::string>(&output_folder)->default_value("./data/sdv_voxel_grid/"),
				"Output folder path.")
			;
        po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);

		if (vm.count("help"))
		{
			std::cout << desc << "\n";
			return false;
		}

		po::notify(vm);
	}
    catch (std::exception& e)
	{
		std::cerr << "ERROR: " << e.what() << "\n";
		return false;
	}
	catch (...)
	{
		std::cerr << "Unknown error!" << "\n";
		return false;
	}

	return true;
} 

// Reads a file with the interest point indices file
std::vector<int> readKeypoints(std::string filename)
{
    char separator = ' ';
    std::vector<int> result;
    std::string row, item;

    std::ifstream in(filename);

       while (getline(in, row))
    {
        std::stringstream ss(row);
        std::getline(ss, item, separator);
        result.push_back(std::stoi(item.c_str()));
    }

    in.close();
    return result;
}

// Checks if the file exists
bool fileExist(const std::string& name)
{
    std::ifstream f(name.c_str());  // New enough C++ library will accept just name
    return f.is_open();
}

// Saves the descriptor to a binary csv file
void saveVector(std::string filename, const std::vector<std::vector<float>> descriptor)
{
    std::cout << "Saving Features to a CSV file:" << std::endl;
    std::cout << filename << std::endl;

    std::ofstream outFile;
    outFile.open(filename, std::ios::binary);

    float writerTemp;
    for (int i = 0; i < descriptor.size(); i++)
    {
        for (int j = 0; j < descriptor[i].size(); j++)
        {
            writerTemp = descriptor[i][j];
            outFile.write(reinterpret_cast<const char*>(&writerTemp), sizeof(float));
        }
    }
    outFile.close();
}

// Initizales a grid using the step size and the number of voxels per side
flann::Matrix<float> initializeGridMatrix(const int n, float x_step, float y_step, float z_step)
{
    int grid_size = n*n*n;
    flann::Matrix<float> input(new float[grid_size * 3], grid_size, 3);

    float xs = -(n / 2)*x_step + 0.5*x_step;
    float ys = -(n / 2)*y_step + 0.5*y_step;
    float zs = -(n / 2)*z_step + 0.5*z_step;

    for (int i = 0; i < n; i++)
    {
        //move on x axis
        for (int j = 0; j < n; j++)
        {
            //move on y axis
            for (int k = 0; k < n; k++)
            {
                //move on z axis
                input[i + n*j + n*n*k][0] = xs + x_step * i;
                input[i + n*j + n*n*k][1] = ys + y_step * j;
                input[i + n*j + n*n*k][2] = zs + z_step * k;
            }
        }
    }
    return input;
}

// Estimates the Z axis of the local reference frame
void toldiComputeZaxis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Vertex &z_axis, std::vector<float> point_dst)
{
    int i;
    pcl::PointXYZ query_point = cloud->points[0];
    // calculate covariance matrix
    Eigen::Matrix3f Cov = Eigen::Matrix3f::Zero();
    Eigen::Matrix<float, 4, 1> centroid;
    centroid[0] = query_point.x;
    centroid[1] = query_point.y;
    centroid[2] = query_point.z;

    Eigen::Vector4f queryPointVector = query_point.getVector4fMap();
    Eigen::Matrix<float, Eigen::Dynamic, 4> vij(point_dst.size(), 4);
    int valid_nn_points = 0;
    double distance = 0.0;
    double sum = 0.0;

    pcl::computeCovarianceMatrix(*cloud, centroid, Cov);

    EIGEN_ALIGN16 Eigen::Vector3f::Scalar eigen_min;
    EIGEN_ALIGN16 Eigen::Vector3f normal;
    pcl::eigen33(Cov, eigen_min, normal);
    z_axis.x = normal(0);
    z_axis.y = normal(1);
    z_axis.z = normal(2);

      // z-axis sign disambiguity
    float z_sign = 0;
    for (i = 0; i < cloud->points.size(); i++)
    {
        float vec_x = query_point.x - cloud->points[i].x;
        float vec_y = query_point.y - cloud->points[i].y;
        float vec_z = query_point.z - cloud->points[i].z;
        z_sign += (vec_x*z_axis.x + vec_y*z_axis.y + vec_z*z_axis.z);
    }
    if (z_sign < 0)
    {
        z_axis.x = -z_axis.x;
        z_axis.y = -z_axis.y;
        z_axis.z = -z_axis.z;
    }
}

// Estimates the X axis of the local reference frame
void toldiComputeXaxis(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Vertex z_axis, float sup_radius, std::vector<float> point_dst, Vertex &x_axis)
{
    int i, j;
    pcl::PointXYZ query_point = cloud->points[0];
    //
    std::vector<Vertex> vec_proj;
    std::vector<float> dist_weight, sign_weight;//store weights w1,w2
    for (i = 0; i < cloud->points.size(); i++)
    {
        Vertex temp;
        Vertex pq = { cloud->points[i].x - query_point.x,cloud->points[i].y - query_point.y,cloud->points[i].z - query_point.z };
        float proj = z_axis.x*pq.x + z_axis.y*pq.y + z_axis.z*pq.z;
        if (proj >= 0)
            sign_weight.push_back(pow(proj, 2));

              else
            sign_weight.push_back(-pow(proj, 2));
        temp.x = pq.x - proj*z_axis.x;
        temp.y = pq.y - proj*z_axis.y;
        temp.z = pq.z - proj*z_axis.z;
        vec_proj.push_back(temp);
    }

    for (i = 0; i < point_dst.size(); i++)
    {
        float wei_temp = sup_radius - point_dst[i];
        wei_temp = pow(wei_temp, 2);
        dist_weight.push_back(wei_temp);
    }

      Vertex x_axis_temp = { 0.0f,0.0f,0.0f };
    for (i = 0; i < cloud->points.size(); i++)
    {
        float weight_sum = dist_weight[i] * sign_weight[i];
        x_axis_temp.x += weight_sum*vec_proj[i].x;
        x_axis_temp.y += weight_sum*vec_proj[i].y;
        x_axis_temp.z += weight_sum*vec_proj[i].z;
    }
    //Normalization
    float size = sqrt(pow(x_axis_temp.x, 2) + pow(x_axis_temp.y, 2) + pow(x_axis_temp.z, 2));
    x_axis_temp.x /= size;
    x_axis_temp.y /= size;
    x_axis_temp.z /= size;
    x_axis = x_axis_temp;
}

// Estimates the Y axis of the local reference frame
void toldiComputeYaxis(Vertex x_axis, Vertex z_axis, Vertex &y_axis)
{
    Eigen::Vector3f x(x_axis.x, x_axis.y, x_axis.z);
    Eigen::Vector3f z(z_axis.x, z_axis.y, z_axis.z);
    Eigen::Vector3f y;

    y = x.cross(z);//cross product

    y_axis.x = y(0);
    y_axis.y = y(1);
    y_axis.z = y(2);
}

// Compute the lrf accordin to the method from toldi paper, for the points selected with the indices
void toldiComputeLRF(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                     std::vector<int> indices,
                     float sup_radius,
                     float smoothingFactor,
                     std::vector<LRF> &cloud_LRF,
                     std::vector<std::vector <int>>& neighbors,
                     std::vector<std::vector <int>>& neighbors_smoothing_idx,
                     std::vector<std::vector <float>>& neighbors_smoothing_distance)
{
    int i, j, m;
    // Initialize all the variables
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    std::vector<int> point_idx;
    std::vector<float> point_dst;
    kdtree.setInputCloud(cloud);
    pcl::PointXYZ query_point;
    pcl::PointXYZ test;