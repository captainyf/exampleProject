#pragma once

#include"layer.h"
#include"net.h"
#include"utils.h"

class Yolox {
public:
	Yolox();
	~Yolox();
	void Logger(const char* format, ...);
	bool Init(const std::string model_path, unsigned int num_thread);
	std::vector<Object> Run(const cv::Mat& img);

private:
	FILE* result_txt_;
	bool is_output_console_ = false;		//打印在控制台
	bool is_output_result_txt_ = false;		//写入日志文件
	ncnn::Net yolox_;

private:
	void EnableResultTxt();
	float intersection_area(const Object& a, const Object& b);
	void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right);
	void qsort_descent_inplace(std::vector<Object>& objects);
	void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);
	void generate_grids_and_stride(const int target_size, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides);
	void generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects);


};