#ifndef _TYPES_HPP_
#define _TYPES_HPP_

#include <vector>
#include <opencv2/core.hpp>

namespace KLT_Tracker{

struct Patch{
    std::vector<cv::Mat> patchPyr;
    std::vector<cv::Mat> Hessian;
    std::vector<cv::Mat> Jcache;
};

}

#endif //_TYPES_HPP_
