#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>

#include"detect.h"
#include"utils.h"

#define _SUBDIR       0x10    /* Subdirectory */

cv::Mat draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
            obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return image;
}

int main(int argc, char** argv)
{
	//获取算法版本号
	auto p_version = OD_GetVersion();
	std::cout << p_version << std::endl;

	//模型初始化
	std::string model_path = "../../models/";
	int num_thread = 4;
	void* p_detect = OD_Create(model_path.c_str(), num_thread);
	if (p_detect == nullptr)
	{
		std::cout << "yolox init failed" << std::endl;
		return 0;
	}

    cv::Mat image;
    cv::namedWindow("frame");//创建显示窗口
    cv::VideoCapture capture;
    capture.open(0);
    if (capture.isOpened())
    {
        std::cout <<  "Capture is opened" << std::endl;
        for (;;)
        {
            capture >> image;
            if (image.empty())
            {
                break;
            }

            double ts = (double)cv::getTickCount();
            auto p_det_res = OD_Run(image.data, image.cols, image.rows, image.channels(), p_detect);
            double te = (double)cv::getTickCount();
            std::cout << "infer time: " << (te - ts) / cv::getTickFrequency() * 100 << " ms" << std::endl;

            //可视化
            if (!p_det_res)
            {
                std::cout << "detect no object" << std::endl;
            }
            else
            {
                std::vector<Object> objects = ((ObjectDetectResult*)p_det_res)->objects;
                image = draw_objects(image, objects);
            }
            cv::imshow("frame", image);
            if (27 == cv::waitKey(20))//"ESC"
			    break;

            //单次推理结束后，释放存放推理结果的内存
            OD_ReleaseResult(p_det_res);
        }
    }
    else
    {
        std::cout << "No capture" << std::endl;
    }

	//所有推理结束后，释放模型对象句柄
	OD_Release(p_detect);

    cv::destroyWindow("frame");
    capture.release();

	return 0;
}
