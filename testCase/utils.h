#pragma once

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct ObjectDetectResult
{
	std::vector<Object> objects;
};
