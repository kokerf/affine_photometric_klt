#pragma once

#include <Eigen/Dense>
#include "types.hpp"

namespace KLT_Tracker
{

using namespace cv;
using namespace Eigen;

enum {
    USE_INITIAL_FLOW      = 4,
};

//! in this namespace, use those typedef
typedef Matrix<int, 2, 1> Vector2i;
typedef Matrix<float, 2, 1> Vector2f;
typedef Matrix<float, 4, 1> Vector4f;
typedef Matrix<float, 8, 1> Vector8f;
typedef Matrix<float, 2, 2, RowMajor> Matrix2f;
typedef Matrix<float, 8, 8, RowMajor> Matrix8f;

struct LKTInvoker : ParallelLoopBody
{

    LKTInvoker(const Patch *_patches, const Mat &_nextImg,
               const Vector2f *_prevPts, Vector2f *_nextPts, Vector4f *_nextAffines, Vector2f *_nextIllums,
               uchar *_status, float *_err, int _level, int _maxLevel,
               int _maxIter, float _minEpslion, int _flags, float _maxResThreshold);

    void operator()(const Range &range) const;

    const Patch *prevPatches;
    const Mat *nextImg;
    const Vector2f *prevPts;
    Vector2f *nextPts;
    Vector4f *nextAffines;
    Vector2f *nextIllums;
    uchar *status;
    float *err;
    int level;
    int maxLevel;
    int maxIter;
    float minEpslion;
    int flags;
    float maxResThreshold;
};


void getOpticalFlowPyramidPatch(InputArray _pts, const std::vector<cv::Mat> &imgPyr, std::vector<Patch> &pathes, Size winSize, int maxLevel);

void affinePhotometricPyrLKT(const Mat &prevImg, const Mat & nextImg,
                             InputArray _prevPts, InputOutputArray _nextPts,
                             OutputArray _status, OutputArray _err,
                             Size winSize, int maxLevel, int maxIter, float minEpslion,
                             int flags, double maxResThreshold);

}
