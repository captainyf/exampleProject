#include"detect.h"
#include"yolox.h"
#include<stdio.h>
#include<string>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>


#define OBJECTDETECT_VERSION  "1.23.0605.1"
//版本号命名规则：
//- 4位
//- 用'.'分隔，x.y.z.d
//- 第一位x表示类别序号，1表示算法库，2表示应用程序
//- 第二位y表示年份后两位，例如2021年表示为21
//- 第三位z表示日期，例如5月31日，表示为0531
//- 第四位d表示日内顺序号，表示该库或该执行文件在当日内第几次发布
//例如，a.dll在2021年5月31日第一次发布，版本号为1.21.0531.1

const char* OD_GetVersion()
{
	static const std::string version = std::string("object detect version:[") 
		+ std::string(OBJECTDETECT_VERSION) + std::string("]\n");
	return version.c_str();
}


void* OD_Create(const char* models_path, int num_thread = 1)
{
	//load models
	Yolox* p_det = new Yolox;
	bool flag = p_det->Init(models_path, num_thread);
	if (!flag)
	{
		if (p_det)
		{
			delete p_det;
			p_det = nullptr;
		}
		return nullptr;
	}
	void* p = (void*)p_det;
	return p;
}


void* OD_Run(unsigned char* p_data, int nw, int nh, int channels, void* p_det)
{
	if (p_data == nullptr)		//输入图像为空
	{
		//todo 返回错位类型标识
		return nullptr;
	}
	if (p_det == nullptr)		//推理模型指针为空
	{
		//todo 返回错位类型标识
		return nullptr;
	}

	cv::Mat src;
	if (channels == 3 || channels == 1)
	{
		if (channels == 1)
		{
			cv::Mat src_GRAY = cv::Mat(nh, nw, CV_8UC1, p_data);
			cv::cvtColor(src_GRAY, src, cv::COLOR_GRAY2BGR);
		}
		else
		{
			src = cv::Mat(nh, nw, CV_8UC3, p_data);
		}
	}

	Yolox* p_obj_det = (Yolox*)p_det;
	std::vector<Object> objects = p_obj_det->Run(src);
	if (objects.empty())
	{
		//todo 返回错位类型标识
		return nullptr;
	}

	//根据具体需求，对检测结果后处理
	


	//数据结构转换
	ObjectDetectResult* p_detect_result = new ObjectDetectResult;
	p_detect_result->objects = objects;
	return p_detect_result;
}

void OD_ReleaseResult(void* p_res)
{
	//根据检测结果的数据结构进行修改
	if (p_res != nullptr)
	{
		ObjectDetectResult* p_result = (ObjectDetectResult*)p_res;
		delete p_result;
		p_result = nullptr;
	}
	return;
}

void OD_Release(void* p_det)
{
	if (p_det != nullptr)
	{
		Yolox* p_yolox = (Yolox*)p_det;
		delete p_yolox;
		p_yolox = nullptr;
	}
	return;
}
