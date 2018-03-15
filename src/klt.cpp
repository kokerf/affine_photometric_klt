//
// Created by neu on 18-3-14.
//

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <Eigen/Dense>
#include "klt.hpp"
#include "types.hpp"


namespace KLT_Tracker{

LKTInvoker::LKTInvoker(const Mat *_prevPatch,
                       const Mat *_prevHessian, //! inv hessian
                       const Mat *_prevJacobian,
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
    prevPatch(_prevPatch),
    prevHessian(_prevHessian),
    prevJacobian(_prevJacobian),
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
        const Mat &T = prevPatch[ptidx];
        const Mat &Hinv = prevHessian[ptidx];
        const Mat &Jcache = prevJacobian[ptidx];
        assert(T.type() == CV_8UC1);
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
                affine << 1.0, 0.0, 1.0, 0,0;
                alpha = 1.0f;
                beta = 0.0f;
            }
        }
        else
        {
            nextPt = nextPts[ptidx] * 2.f;
            affine = nextAffines[ptidx];
            affine[0] += 1.0f;
            affine[2] += 1.0f;
            alpha = nextIllums[ptidx][0];
            beta = nextIllums[ptidx][1];
        }


        Map<const Matrix<float, Dynamic, Dynamic, RowMajor> > eigHinv(Hinv.ptr<float>(0), Hinv.rows, Hinv.cols);
        Map<const Matrix<float, Dynamic, Dynamic, RowMajor> > eigJcache(Jcache.ptr<float>(0), Jcache.rows, Jcache.cols);
        assert(Hinv.cols == Hinv.rows);
        assert(Hinv.cols == Jcache.cols);
        assert(Jcache.rows == winSize.area());

        const int stepI = (int) (I.step / I.elemSize1());

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


            Vector8f Jres; Jres.setZero();
            int n = 0;
            int r, c;
            int vy, vx;
            for(r = 0; r < winSize.height; ++r)
            {
                const uchar *Tptr = T.ptr<uchar>(r);
                const uchar *Iptr = I.ptr<uchar>(r + inextPt[1]) + inextPt[0];

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
                        float a = warpPt[0] - iwarpPt[0];
                        float b = warpPt[1] - iwarpPt[1];
                        float iw00 = cvRound((1.f - a) * (1.f - b));
                        float iw01 = cvRound(a * (1.f - b));
                        float iw10 = cvRound((1.f - a) * b);
                        float iw11 = 1 - iw00 - iw01 - iw10;

                        float diff = alpha * (Iptr[c] * iw00 + Iptr[c+1] * iw01 + Iptr[c+stepI] * iw10 + Iptr[c+stepI] * iw11) + beta - (float)Tptr[c];
                        Jres.noalias() += eigJcache.row(n) * diff;
                    }
                }
            }

            Vector8f delta = eigHinv * Jres;

            float dA11 = delta[0] + 1.f;
            float dA12 = delta[1];
            float dA22 = delta[2] + 1.f;
            float dA21 = delta[3];

            float D =  dA11 * dA22 - dA12 * dA12;

            Matrix2f dAinv; dAinv << dA22/D, -dA21/D, dA12/D, dA11/D;
            Map<Matrix2f> A(affine.data());

            A = A*dAinv;
            nextPt -= dAinv * Vector2f(delta[4],delta[5]);
            beta -= (alpha+1)/delta[7];
            alpha /= (delta[6]+1);

            nextPts[ptidx] = nextPt;
            nextAffines[ptidx] << affine[0]-1.0f, affine[1], affine[2]-1.0f, affine[3];
            nextIllums[ptidx] << alpha, beta;

            if(delta.dot(delta) <= minEpslion)
                break;

        }
    }
}

void getOpticalFlowPyramidPatch(const std::vector<cv::Point2f> &pts, const cv::Mat &img, std::vector<Patch> &pathes, Size winSize, int maxLevel)
{
    assert(winSize.height%2 == 1 && winSize.width%2 == 1);
    const cv::Point2i halfwin((winSize.width-1)*0.5, (winSize.width-1)*0.5);

    std::vector<cv::Mat> imgPyr;
    cv::buildOpticalFlowPyramid(img, imgPyr, winSize, maxLevel, false);

    std::vector<cv::Mat> derivXPyr(imgPyr.size());
    std::vector<cv::Mat> derivYPyr(imgPyr.size());
    for(int l = 0; l <= maxLevel; ++l)
    {
        Size imgSize = imgPyr[l].size();
        derivXPyr[l].resize(imgSize.height + winSize.height*2, imgSize.width + winSize.height*2);
        derivYPyr[l].resize(imgSize.height + winSize.height*2, imgSize.width + winSize.height*2);

        cv::Mat derivX = derivXPyr[l](Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        cv::Mat derivY = derivYPyr[l](Rect(winSize.width, winSize.height, imgSize.width, imgSize.height));
        cv::Scharr(imgPyr[l], derivXPyr[l], CV_32FC1, 1, 0, 1, 0);
        cv::Scharr(imgPyr[l], derivYPyr[l], CV_32FC1, 0, 1, 1, 0);
        cv::copyMakeBorder(derivX, derivXPyr[l], winSize.height, winSize.height, winSize.width, winSize.width, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED);
        cv::copyMakeBorder(derivY, derivYPyr[l], winSize.height, winSize.height, winSize.width, winSize.width, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED);
    }

    const size_t N = pts.size();
    pathes.resize(N);
    const cv::Point2f offset(winSize.height, winSize.width);
    for(size_t n = 0; n < N; ++n)
    {
        std::vector<cv::Mat> patches(maxLevel+1);
        std::vector<cv::Mat> Hessians(maxLevel+1);
        std::vector<cv::Mat> Jcaches(maxLevel+1);
        const cv::Point2f pt = pts[n];
        for(int l = 0; l <= maxLevel; ++l)
        {
            const cv::Point2f ptLevel = pt * (1.0f / (1 << l));
            patches[l] = cv::Mat::zeros(winSize, CV_32FC1);
            Jcaches[l] = cv::Mat::zeros(winSize.area(), 8, CV_32FC1);
            Matrix<float, 8, 8, RowMajor> H;
            Map<Matrix<float, Dynamic, Dynamic, RowMajor> > Jcache((float*)Jcaches[l].data, winSize.area(), 8);

            cv::Point2i iptLevel;
            iptLevel.x = cvRound(ptLevel.x - halfwin.x);
            iptLevel.y = cvRound(ptLevel.y - halfwin.y);

            float a = ptLevel.x - iptLevel.x;
            float b = ptLevel.y - iptLevel.y;
            float iw00 = (1.f - a) * (1.f - b);
            float iw01 = a * (1.f - b);
            float iw10 = (1.f - a) * b;
            float iw11 = 1- iw00 - iw01 - iw10;

            const int stepI = (int)(imgPyr[l].step / imgPyr[l].elemSize1());

            int x, y;
            float u = ptLevel.x - halfwin.x;
            float v = ptLevel.y - halfwin.y;
            int i = 0;
            Vector8f J;
            for(y = 0; y < winSize.height; ++y, ++v)
            {
                const uchar *pI = imgPyr[l].ptr<uchar>(y + iptLevel.y) + iptLevel.x;
                const float *pdIx = derivXPyr[l].ptr<float>(y + iptLevel.y) + iptLevel.x;
                const float *pdIy = derivYPyr[l].ptr<float>(y + iptLevel.y) + iptLevel.x;
                float *pP = patches[l].ptr<float>(0);

                for(x = 0; x < winSize.width; ++x, ++u, ++i)
                {
                    pP[x] = pI[x] * iw00 + pI[x + 1] * iw01 + pI[x + stepI] * iw10 + pI[x + stepI + 1] * iw11;
                    J << u * pdIx[x], v * pdIx[x], u * pdIy[x], v * pdIy[x], pdIx[x], pdIy[x], pI[x], 1;

                    H.noalias() += J * J.transpose();
                    Jcache.row(i) = J;
                }

            }

            Hessians[l] = cv::Mat(8, 8, CV_32FC1);
            Map<Matrix<float, 8, 8, RowMajor> > Hinv(Hessians[l].ptr<float>(0));
            Hinv = H.inverse();
        }

        pathes[n].patchPyr = patches;
        pathes[n].Hinv = Hessians;
        pathes[n].Jcache = Jcaches;
    }

}



}
