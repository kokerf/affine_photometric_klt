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
        const Mat &Hinv = patch.Hinv[level];
        const Mat &Jcache = patch.Jcache[level];

        assert(T.type() == CV_32FC1);
        assert(Hinv.type() == CV_32FC1);
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


        Map<const Matrix<float, Dynamic, Dynamic, RowMajor> > eigHinv(Hinv.ptr<float>(0), Hinv.rows, Hinv.cols);
        Map<const Matrix<float, Dynamic, Dynamic, RowMajor> > eigJcache(Jcache.ptr<float>(0), Jcache.rows, Jcache.cols);
        CV_Assert(Hinv.cols == Hinv.rows);
        CV_Assert(Hinv.cols == Jcache.cols);
        CV_Assert(Jcache.rows == winSize.area());

        const int stepI = (int) (I.step / I.elemSize1());

//        std::cout << "prevPt" << prevPt << std::endl;
//        std::cout << "nextPt" << nextPt << std::endl;
//
//        cv::imshow("I", I);
//        cv::waitKey(0);

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
//            cv::waitKey(0);

            Vector8f Jres; Jres.setZero();
            int n = 0;
            int r, c;
            int vy, vx;
            float res_sum = 0;
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

                    if(iwarpPt[0] < -winSize.width || iwarpPt[1] < -winSize.height ||
                        iwarpPt[0] >= I.cols + winSize.width || iwarpPt[1] >= I.rows + winSize.height)
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

                        res_sum += diff*diff;

                        Jres.noalias() += eigJcache.row(n) * diff;
                    }
                }
            }


//            std::cout << res_sum << ", Jres: " << Jres.transpose() << std::endl;

            Vector8f delta = eigHinv * Jres;

            Matrix2f dA;
            dA << delta[0] + 1.f, delta[1], delta[2], delta[3] + 1.f;

            Matrix2f dAinv = dA.inverse();

            dAinv.setIdentity();

            Map<Matrix2f> A(affine.data());

//            std::cout << j << "delta: " << delta.transpose() << std::endl;
//            std::cout << "A:\n" << A << "\ninv:\n" << dAinv << std::endl;
//            std::cout << "A*dAinv\n" << A*dAinv << std::endl;

            A = A*dAinv;
            nextPt -= dAinv * Vector2f(delta[4],delta[5]);
            beta -= (alpha+1)*delta[7];
            alpha /= (delta[6]+1);

            nextPts[ptidx] = nextPt;
            nextAffines[ptidx] << affine[0]-1.0f, affine[1], affine[2], affine[3]-1.0f;
            nextIllums[ptidx] << alpha, beta;

//            Mat dest;
//            cv::cvtColor(I, dest, CV_GRAY2RGB);
//            Point2f lt(nextPt[0]-halfWin[0]-1, nextPt[1]-halfWin[0]-1);
//            Point2f rb(nextPt[0]+halfWin[0]+1, nextPt[1]+halfWin[0]+1);
//            cv::rectangle(dest, lt, rb, cv::Scalar(255,0,0));
//            cv::imshow("dest", dest);
//            Mat src;
//            T.convertTo(src, CV_8UC1);
//            cv::imshow("src", src);
//
//            waitKey(0);

            if(delta.dot(delta) <= minEpslion)
                break;

        }
    }
}

void getOpticalFlowPyramidPatch(InputArray _pts, const std::vector<cv::Mat> &imgPyr, std::vector<Patch> &pathes, Size winSize, int maxLevel)
{
    Mat ptsMat = _pts.getMat();
    int npoints;
    CV_Assert((npoints = ptsMat.checkVector(2, CV_32F, true)) >= 0);

    assert(winSize.height%2 == 1 && winSize.width%2 == 1);
    const cv::Point2f halfwin((winSize.width-1)*0.5, (winSize.width-1)*0.5);

    std::vector<Mat> derivXPyr(imgPyr.size());
    std::vector<Mat> derivYPyr(imgPyr.size());
    for(int l = 0; l <= maxLevel; ++l)
    {
        Size wholeSize;
        Point ofs;
        imgPyr[l].locateROI(wholeSize, ofs);
        CV_Assert(ofs.x == winSize.width && ofs.y == winSize.height);
        CV_Assert(wholeSize.width == imgPyr[l].cols + 2*winSize.width && wholeSize.height == imgPyr[l].rows + 2*winSize.height);

        Size imgSize = imgPyr[l].size();
        derivXPyr[l] = Mat::zeros(imgSize.height + winSize.height*2, imgSize.width + winSize.height*2, CV_32FC1);
        derivYPyr[l] = Mat::zeros(imgSize.height + winSize.height*2, imgSize.width + winSize.height*2, CV_32FC1);

        Mat derivX = derivXPyr[l](Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        Mat derivY = derivYPyr[l](Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        const double scale = 1.0/32;
        Scharr(imgPyr[l], derivX, CV_32FC1, 1, 0, scale, 0);
        Scharr(imgPyr[l], derivY, CV_32FC1, 0, 1, scale, 0);
        copyMakeBorder(derivX, derivXPyr[l], winSize.height, winSize.height, winSize.width, winSize.width, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED);
        copyMakeBorder(derivY, derivYPyr[l], winSize.height, winSize.height, winSize.width, winSize.width, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED);

        derivXPyr[l].adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);
        derivYPyr[l].adjustROI(-winSize.height, -winSize.height, -winSize.width, -winSize.width);
    }

    const Point2f *pts = ptsMat.ptr<Point2f>();
    pathes.resize(npoints);
    const cv::Point2f offset(winSize.height, winSize.width);
    for(size_t n = 0; n < npoints; ++n)
    {
        std::vector<Mat> patches(maxLevel+1);
        std::vector<Mat> Hessians(maxLevel+1);
        std::vector<Mat> Jcaches(maxLevel+1);
        const cv::Point2f pt = pts[n];
        for(int l = 0; l <= maxLevel; ++l)
        {
            cv::Point2f ptLevel = pt * (1.0f / (1 << l));
            patches[l] = Mat::zeros(winSize, CV_32FC1);
            Jcaches[l] = Mat::zeros(winSize.area(), 8, CV_32FC1);
            Matrix<float, 8, 8, RowMajor> H;
            Map<Matrix<float, Dynamic, Dynamic, RowMajor> > Jcache((float*)Jcaches[l].data, winSize.area(), 8);

            cv::Point2i iptLevel;
            ptLevel -= halfwin;
            iptLevel.x = cvRound(ptLevel.x);
            iptLevel.y = cvRound(ptLevel.y);

            float a = ptLevel.x - iptLevel.x;
            float b = ptLevel.y - iptLevel.y;
            float iw00 = (1.f - a) * (1.f - b);
            float iw01 = a * (1.f - b);
            float iw10 = (1.f - a) * b;
            float iw11 = 1- iw00 - iw01 - iw10;

            const int stepI = (int)(imgPyr[l].step / imgPyr[l].elemSize1());

            int x, y;
            int i = 0;
            Vector8f J;
            H.setZero();
            for(y = 0; y < winSize.height; ++y)
            {
                const uchar *pI = imgPyr[l].ptr<uchar>(y + iptLevel.y) + iptLevel.x;
                const float *pdIx = derivXPyr[l].ptr<float>(y + iptLevel.y) + iptLevel.x;
                const float *pdIy = derivYPyr[l].ptr<float>(y + iptLevel.y) + iptLevel.x;
                float *pP = patches[l].ptr<float>(y);

                for(x = 0; x < winSize.width; ++x, ++i)
                {
                    pP[x] = pI[x] * iw00 + pI[x + 1] * iw01 + pI[x + stepI] * iw10 + pI[x + stepI + 1] * iw11;
                    float dx = pdIx[x] * iw00 + pdIx[x + 1] * iw01 + pdIx[x + stepI] * iw10 + pdIx[x + stepI + 1] * iw11;
                    float dy = pdIy[x] * iw00 + pdIy[x + 1] * iw01 + pdIy[x + stepI] * iw10 + pdIy[x + stepI + 1] * iw11;

                    J << x * dx, y * dx, x * dy, y * dy, dx, dy, pP[x], 1;

                    H.noalias() += J * J.transpose();
                    Jcache.row(i) = J;
                }

            }

            Hessians[l] = cv::Mat(8, 8, CV_32FC1);
            Map<Matrix<float, 8, 8, RowMajor> > Hinv(Hessians[l].ptr<float>(0));
            Hinv = H.inverse();


//            std::cout << "pt: " << pt << "patch: \n" << patches[l] << ", l " << ptLevel<< std::endl;
//            Mat src;
//            cv::cvtColor(imgPyr[l], src, CV_GRAY2RGB);
//            cv::rectangle(src, ptLevel-Point2f(1,1), ptLevel+halfwin*2+Point2f(1,1), cv::Scalar(255,0,0));
//            cv::imshow("imgPyr", src);
//            Mat temp;
//            patches[l].convertTo(temp, CV_8UC1);
//            imshow("patch", temp);
//            waitKey(0);
        }

        pathes[n].patchPyr = patches;
        pathes[n].Hinv = Hessians;
        pathes[n].Jcache = Jcaches;

//        if(n == 1)
//        {
//            std::cout << "pt: " << pt << "patch: \n" << patches[0] << std::endl;
//            cv::Mat temp;
//            patches[maxLevel].convertTo(temp, CV_8UC1);
//            cv::imshow("patches[maxLevel]", temp);
//            cv::waitKey(0);
//        }
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
