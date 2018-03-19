#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/klt.hpp"

int main(int argc, char const *argv[])
{
    if(argc != 3)
    {
        std::cout << "error input!" << std::endl;
        return -1;
    }

    std::string img_str1 = argv[1];
    std::string img_str2 = argv[2];

    cv::Mat img1 = cv::imread(img_str1, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img2 = cv::imread(img_str2, CV_LOAD_IMAGE_GRAYSCALE);

    cv::imshow("img1", img1);
    cv::imshow("img2", img2);

    std::vector<cv::Point2f> pts_prev;
    cv::goodFeaturesToTrack(img1, pts_prev, 100, 0.01, 5);

    cv::Mat show;
    cv::cvtColor(img1, show, CV_GRAY2RGB);
    for(size_t i = 0; i < pts_prev.size(); i++)
    {
        cv::circle(show, pts_prev[i], 3, cv::Scalar(255, 0, 0));
    }

    cv::imshow("show", show);

    cv::waitKey(0);

    std::vector<KLT_Tracker::Patch> pathes;

    std::vector<cv::Mat> imgPyr;
    cv::buildOpticalFlowPyramid(img1, imgPyr, cv::Size(21,21), 3, false);

    KLT_Tracker::getOpticalFlowPyramidPatch(pts_prev, imgPyr, pathes, cv::Size(21,21), 3);

    std::vector<cv::Point2f> pts_next;
    std::vector<uchar> status;
    std::vector<float> error;
    double t0 = cv::getTickCount();
    KLT_Tracker::affinePhotometricPyrLKT(img1, img2, pts_prev, pts_next, status, error, cv::Size(21,21), 3, 30, 0.01, 0, 30);
    double t1 = cv::getTickCount();
    std::cout << "time: " << (t1-t0)/cv::getTickFrequency() << std::endl;

    cv::Mat show_dest;
    cv::cvtColor(img2, show_dest, CV_GRAY2RGB);
    for(size_t i = 0; i < pts_prev.size(); i++)
    {
        if(!status[i])
            continue;

        cv::circle(show_dest, pts_next[i], 3, cv::Scalar(255, 0, 0));
    }

    cv::imshow("dest", show_dest);
    cv::waitKey(0);

    return 0;
}