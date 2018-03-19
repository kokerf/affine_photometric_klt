//
// Created by neu on 18-3-14.
//

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <Eigen/Dense>
#include "klt.hpp"

#include <opencv2/highgui.hpp>
#include <iostream>

namespace KLT_Tracker{

LKTInvoker::LKTInvoker(const Patch *_prevPatches,
                       const Mat &_nextImg,
                       const Vector2f *_prevPts,
                       Vector2f *_nextPts,
                       Vector4f *_nextAffines,
                       Vector2f *_nextIllums,
                       uchar *_status,
                       float *_err,
                       int _level,
                       int _maxLevel,
                       int _maxIter,
                       float _minEpslion,
                       int _flags,
                       float _maxResThreshold) :
    prevPatches(_prevPatches),
    nextImg(&_nextImg),
    prevPts(_prevPts),
    nextPts(_nextPts),
    nextAffines(_nextAffines),
    nextIllums(_nextIllums),
    status(_status),
    err(_err),
    level(_level),
    maxLevel(_maxLevel),
    maxIter(_maxIter),
    minEpslion(_minEpslion),
    flags(_flags),
    maxResThreshold(_maxResThreshold)
{}

void LKTInvoker::operator()(const Range &range) const
{
    const Mat &I = *nextImg;
    assert(I.type() == CV_8UC1);

    for(int ptidx = range.start; ptidx < range.end; ptidx++)
    {
        const Patch &patch = prevPatches[ptidx];
        const Mat &T = patch.patchPyr[level];
        const Mat &H = patch.Hessian[level];
        const Mat &Jcache = patch.Jcache[level];

        assert(T.type() == CV_32FC1);
        assert(H.type() == CV_32FC1);
        assert(Jcache.type() == CV_32FC1);
        const Size2i winSize(T.cols, T.rows);
        assert(T.cols % 2 == 1 && T.rows % 2 == 1);
        const Vector2i halfWin(T.cols/2, T.rows/2);

        Vector2f prevPt = prevPts[ptidx] * (float) (1. / (1 << level));
        Vector2f nextPt;
        Vector4f affine;
        float alpha;
        float beta;
        if(level == maxLevel)
        {
            if(flags & USE_INITIAL_FLOW)
            {
                nextPt = nextPts[ptidx] * (float) (1. / (1 << level));
                affine = nextAffines[ptidx];
                affine[0] += 1.0f;
                affine[2] += 1.0f;
                alpha = nextIllums[ptidx][0];
                beta = nextIllums[ptidx][1];
            }
            else
            {
                nextPt = prevPt;
                affine << 1.0f, 0.0f, 0.0f, 1.0f;
                alpha = 0.0f;
                beta = 0.0f;
            }
        }
        else
        {
            nextPt = nextPts[ptidx] * 2.f;
            affine = nextAffines[ptidx];
            affine[0] += 1.0f;
            affine[3] += 1.0f;
            alpha = nextIllums[ptidx][0];
            beta = nextIllums[ptidx][1];
        }


        Map<const Matrix<float, Dynamic, Dynamic, RowMajor> > eigH(H.ptr<float>(0), H.rows, H.cols);
        Map<const Matrix<float, Dynamic, Dynamic, RowMajor> > eigJcache(Jcache.ptr<float>(0), Jcache.rows, Jcache.cols);
        CV_Assert(H.cols == H.rows);
        CV_Assert(H.cols == Jcache.cols);
        CV_Assert(Jcache.rows == winSize.area());

        const int stepI = (int) (I.step / I.elemSize1());

//        std::cout << "prevPt" << prevPt << std::endl;
//        std::cout << "nextPt" << nextPt << std::endl;
//
//        cv::imshow("I", I);
//        cv::waitKey(0);

        Vector8f prevDelta;
        float preRes = std::numeric_limits<float>::max();
        int j;
        for(j = 0; j < maxIter; j++)
        {
            Vector2i inextPt = nextPt.cast<int>();

            if(inextPt[0] < -halfWin[0] || inextPt[0] >= I.cols + halfWin[0]||
                inextPt[1] < -halfWin[1] || inextPt[1] >= I.rows + halfWin[1])
            {
                if(level == 0)
                {
                    if(status)
                        status[ptidx] = false;
                    if(err)
                        err[ptidx] = 0;
                }
                break;
            }


//            Mat dest;
//            cv::cvtColor(I, dest, CV_GRAY2RGB);
//            Point2f lt(nextPt[0]+affine[0]*(-halfWin[0]-1)+affine[1]*(-halfWin[0]-1), nextPt[1]+affine[2]*(-halfWin[0]-1)+affine[3]*(-halfWin[0]-1));
//            Point2f rt(nextPt[0]+affine[0]*(halfWin[0]+1)+affine[1]*(-halfWin[0]-1), nextPt[1]+affine[2]*(halfWin[0]+1)+affine[3]*(-halfWin[0]-1));
//            Point2f lb(nextPt[0]+affine[0]*(-halfWin[0]-1)+affine[1]*(halfWin[0]+1), nextPt[1]+affine[2]*(-halfWin[0]-1)+affine[3]*(+halfWin[0]+1));
//            Point2f rb(nextPt[0]+affine[0]*(halfWin[0]+1)+affine[1]*(halfWin[0]+1), nextPt[1]+affine[2]*(halfWin[0]+1)+affine[3]*(halfWin[0]+1));
//            cv::line(dest, lt, rt, cv::Scalar(255,0,0));
//            cv::line(dest, rt, rb, cv::Scalar(255,0,0));
//            cv::line(dest, rb, lb, cv::Scalar(255,0,0));
//            cv::line(dest, lb, lt, cv::Scalar(255,0,0));
//            cv::imshow("dest", dest);
//            Mat src;
//            T.convertTo(src, CV_8UC1);
//            cv::imshow("src", src);
//            cv::waitKey(0);

            Vector8f Jres; Jres.setZero();
            Vector8f delta;
            int n = 0;
            int r, c;
            int vy, vx;
            float res = 0;
            for(r = 0; r < winSize.height; ++r)
            {
                const float *Tptr = T.ptr<float>(r);

                vy = r-halfWin[1];
                for(c = 0; c < winSize.width; ++c, ++n)
                {
                    vx = c-halfWin[0];
                    Vector2f warpPt(affine[0]*vx + affine[1]*vy + nextPt[0],
                                    affine[2]*vx + affine[3]*vy + nextPt[1]);
                    Vector2i iwarpPt = warpPt.cast<int>();

                    if(iwarpPt[0] < 0|| iwarpPt[1] < 0 ||
                        iwarpPt[0] >= I.cols || iwarpPt[1] >= I.rows)
                    {
                        continue;
                    }
                    else
                    {
                        const uchar *Iptr = I.ptr<uchar>(iwarpPt[1]) + iwarpPt[0];

                        float a = warpPt[0] - iwarpPt[0];
                        float b = warpPt[1] - iwarpPt[1];
                        float iw00 = (1.f - a) * (1.f - b);
                        float iw01 = a * (1.f - b);
                        float iw10 =(1.f - a) * b;
                        float iw11 = 1 - iw00 - iw01 - iw10;

                        float diff = (1+alpha) * (Iptr[0] * iw00 + Iptr[1] * iw01 + Iptr[stepI] * iw10 + Iptr[1+stepI] * iw11) + beta - Tptr[c];

                        res += diff * diff;
                        Jres.noalias() += eigJcache.row(n) * diff;
                    }
                }
            }

            delta = eigH.ldlt().solve(Jres);

            Matrix2f dA;
            dA << delta[0] + 1.f, delta[1], delta[2], delta[3] + 1.f;

            Matrix2f dAinv = dA.inverse();

//            dAinv.setIdentity();

            Map<Matrix2f> A(affine.data());

            A = A*dAinv;
            nextPt -= dAinv * Vector2f(delta[4],delta[5]);
            beta -= (alpha+1)*delta[7];
            alpha /= (delta[6]+1);

//            std::cout << j << ", res: " << res/winSize.area() << "， delta: " << delta.transpose() << std::endl;
//            std::cout << "A:\n" << A << std::endl;
//            std::cout << "pt: " << nextPt.transpose() << ", b: " << beta << ", a: " << alpha << std::endl;

            if(delta[4]*delta[4] + delta[5]*delta[5] <= minEpslion)
                break;

            if(j > 0 && std::abs(delta[4] + prevDelta[4]) < 0.01 &&
                std::abs(delta[5] + prevDelta[5]) < 0.01)
            {
                nextPts[ptidx][0] -= delta[4] * 0.5f;
                nextPts[ptidx][1] -= delta[5] * 0.5f;
                break;
            }

            prevDelta = delta;
            preRes = res;
        }

        nextPts[ptidx] = nextPt;
        nextAffines[ptidx] << affine[0]-1.0f, affine[1], affine[2], affine[3]-1.0f;
        nextIllums[ptidx] << alpha, beta;
    }
}

HessianInvoker::HessianInvoker(const Mat &_img, const Mat &_gx, const Mat &_gy, const Vector2f *_pts, Patch *_patches, Size _winSize, int _level, int _maxLevel):
    img(&_img), gx(&_gx), gy(&_gy), pts(_pts), patches(_patches), winSize(_winSize), level(_level), maxLevel(_maxLevel)
{}

void HessianInvoker::operator()(const Range &range) const
{
    const Mat &Tsrc = *img;
    const Mat &Tx = *gx;
    const Mat &Ty = *gy;
    assert(Tsrc.type() == CV_8UC1);
    assert(Tx.type() == CV_32FC1);
    assert(Ty.type() == CV_32FC1);

    const Vector2f halfwin((winSize.width-1)*0.5, (winSize.width-1)*0.5);

    for(int ptidx = range.start; ptidx < range.end; ptidx++)
    {
        Vector2f ptLevel = pts[ptidx] * (1.0f / (1 << level));
        Patch &patch = patches[ptidx];
        if(patch.patchPyr.size() != maxLevel)
            patch.patchPyr.resize(maxLevel+1);
        if(patch.Hessian.size() != maxLevel)
            patch.Hessian.resize(maxLevel+1);
        if(patch.Jcache.size() != maxLevel)
            patch.Jcache.resize(maxLevel+1);

        Mat &T = patch.patchPyr[level];
        Mat &H = patch.Hessian[level];
        Mat &Jcache = patch.Jcache[level];

        T = Mat::zeros(winSize, CV_32FC1);
        H = Mat::zeros(8, 8, CV_32FC1);
        Jcache= Mat::zeros(winSize.area(), 8, CV_32FC1);
        Map<Matrix<float, 8, 8, RowMajor> > eigH((float*)H.data, 8, 8);
        Map<Matrix<float, Dynamic, Dynamic, RowMajor> > eigJcache((float*)Jcache.data, winSize.area(), 8);

        Vector2i iptLevel;
        ptLevel -= halfwin;
        iptLevel[0] = cvRound(ptLevel[0]);
        iptLevel[1] = cvRound(ptLevel[1]);

        float a = ptLevel[0] - iptLevel[0];
        float b = ptLevel[1] - iptLevel[1];
        float iw00 = (1.f - a) * (1.f - b);
        float iw01 = a * (1.f - b);
        float iw10 = (1.f - a) * b;
        float iw11 = 1- iw00 - iw01 - iw10;

        const int stepI = (int)Tsrc.step / Tsrc.elemSize1();

        int x, y;
        int i = 0;
        Vector8f J;
        for(y = 0; y < winSize.height; ++y)
        {
            const uchar *pI = Tsrc.ptr<uchar>(y + iptLevel[1]) + iptLevel[0];
            const float *pdIx = Tx.ptr<float>(y + iptLevel[1]) + iptLevel[0];
            const float *pdIy = Ty.ptr<float>(y + iptLevel[1]) + iptLevel[0];
            float *pP = T.ptr<float>(y);

            for(x = 0; x < winSize.width; ++x, ++i)
            {
                pP[x] = pI[x] * iw00 + pI[x + 1] * iw01 + pI[x + stepI] * iw10 + pI[x + stepI + 1] * iw11;
                float dx = pdIx[x] * iw00 + pdIx[x + 1] * iw01 + pdIx[x + stepI] * iw10 + pdIx[x + stepI + 1] * iw11;
                float dy = pdIy[x] * iw00 + pdIy[x + 1] * iw01 + pdIy[x + stepI] * iw10 + pdIy[x + stepI + 1] * iw11;

                J << x * dx, y * dx, x * dy, y * dy, dx, dy, pP[x], 1;

                eigH.noalias() += J * J.transpose();
                eigJcache.row(i) = J;
            }
        }
    }
}

void getOpticalFlowPyramidPatch(InputArray _pts, const std::vector<cv::Mat> &imgPyr, std::vector<Patch> &pathes, Size winSize, int maxLevel)
{
    Mat ptsMat = _pts.getMat();
    int npoints;
    CV_Assert((npoints = ptsMat.checkVector(2, CV_32F, true)) >= 0);

    assert(winSize.height%2 == 1 && winSize.width%2 == 1);
    const Point2f halfwin((winSize.width-1)*0.5, (winSize.width-1)*0.5);

    const Point2f *pts = ptsMat.ptr<Point2f>();
    pathes.resize(npoints);

    for(int l = 0; l <= maxLevel; ++l)
    {
        Size wholeSize;
        Point ofs;
        imgPyr[l].locateROI(wholeSize, ofs);
        CV_Assert(ofs.x == winSize.width && ofs.y == winSize.height);
        CV_Assert(wholeSize.width == imgPyr[l].cols + 2*winSize.width && wholeSize.height == imgPyr[l].rows + 2*winSize.height);

        Size imgSize = imgPyr[l].size();
        Mat _derivX = Mat::zeros(imgSize.height + winSize.height*2, imgSize.width + winSize.height*2, CV_32FC1);
        Mat _derivY = Mat::zeros(imgSize.height + winSize.height*2, imgSize.width + winSize.height*2, CV_32FC1);

        Mat derivX = _derivX(Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        Mat derivY = _derivY(Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        const double scale = 1.0/32;
        Scharr(imgPyr[l], derivX, CV_32FC1, 1, 0, scale, 0);
        Scharr(imgPyr[l], derivY, CV_32FC1, 0, 1, scale, 0);
        copyMakeBorder(derivX, _derivX, winSize.height, winSize.height, winSize.width, winSize.width, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED);
        copyMakeBorder(derivY, _derivY, winSize.height, winSize.height, winSize.width, winSize.width, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED);

        _derivX.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);
        _derivY.adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);

        parallel_for_(Range(0, npoints), HessianInvoker(imgPyr[l], _derivX, _derivY, (Vector2f*)pts, pathes.data(), winSize, l, maxLevel));
    }
}

void affinePhotometricPyrLKT(const Mat &prevImg, const Mat & nextImg,
                             InputArray _prevPts, InputOutputArray _nextPts,
                             OutputArray _status, OutputArray _err,
                             Size winSize, int maxLevel, int maxIter, float minEpslion,
                             int flags, double maxResThreshold)
{
    CV_Assert(maxLevel >= 0 && winSize.width > 2 && winSize.width % 2 == 1 && winSize.height > 2 && winSize.height % 2 == 1);

    int level = 0, i, npoints;
    Mat prevPtsMat = _prevPts.getMat();
    CV_Assert((npoints = prevPtsMat.checkVector(2, CV_32F, true)) >= 0);

    if(npoints == 0)
    {
        _nextPts.release();
        _status.release();
        _err.release();
        return;
    }

    if(!(flags & USE_INITIAL_FLOW))
        _nextPts.create(prevPtsMat.size(), prevPtsMat.type(), -1, true);

    Mat nextPtsMat = _nextPts.getMat();
    if(flags & USE_INITIAL_FLOW)
        CV_Assert(nextPtsMat.checkVector(2, CV_32F, true) == npoints);

    const Point2f *prevPts = prevPtsMat.ptr<Point2f>();
    Point2f *nextPts = nextPtsMat.ptr<Point2f>();

    _status.create((int) npoints, 1, CV_8U, -1, true);
    Mat statusMat = _status.getMat(), errMat;
    CV_Assert(statusMat.isContinuous());
    uchar *status = statusMat.ptr();
    float *err = 0;

    for(i = 0; i < npoints; i++)
        status[i] = true;

    if(_err.needed())
    {
        _err.create((int) npoints, 1, CV_32F, -1, true);
        errMat = _err.getMat();
        CV_Assert(errMat.isContinuous());
        err = errMat.ptr<float>();
    }

    if(_err.needed())
    {
        _err.create((int) npoints, 1, CV_32F, -1, true);
        errMat = _err.getMat();
        CV_Assert(errMat.isContinuous());
        err = errMat.ptr<float>();
    }

    std::vector<Mat> prevPyr, nextPyr;
    int maxLevel1 = buildOpticalFlowPyramid(prevImg, prevPyr, winSize, maxLevel, false);
    int maxLevel2 = buildOpticalFlowPyramid(nextImg, nextPyr, winSize, maxLevel, false);
    CV_Assert(maxLevel1 == maxLevel2);
    maxLevel = maxLevel1;

    maxIter = std::min(std::max(maxIter, 0), 100);
    minEpslion = std::min(std::max(minEpslion, 0.f), 10.f);
    minEpslion *= minEpslion;

    std::vector<Patch> prevPatches;
    getOpticalFlowPyramidPatch(_prevPts, prevPyr, prevPatches, winSize, maxLevel);

    Mat nextAffines = Mat::zeros(npoints, 8, CV_32FC1);
    Mat nextIllums = Mat::zeros(npoints, 2, CV_32FC1);

    for(level = maxLevel; level >= 0; level--)
    {

        parallel_for_(Range(0, npoints), LKTInvoker(prevPatches.data(), nextPyr[level],
                                                    (Vector2f*)prevPts, (Vector2f*)nextPts,
                                                    (Vector4f*)nextAffines.data,
                                                    (Vector2f*)nextIllums.data,
                                                    status, err, level, maxLevel, maxIter, minEpslion, flags, maxResThreshold));


    }

}

}
