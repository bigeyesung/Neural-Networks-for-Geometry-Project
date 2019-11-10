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