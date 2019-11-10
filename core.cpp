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