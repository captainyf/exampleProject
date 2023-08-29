#pragma once

#ifdef _MSC_VER
#ifdef OBJECTDETECT_EXPORTS
#define OBJECTDETECT_API __declspec(dllexport)
#else
#define OBJECTDETECT_API __declspec(dllimport)
#endif
#else
#define OBJECTDETECT_API __attribute__ ((visibility("default")))
#endif 

#ifdef __cplusplus
extern "C"
{
#endif

	/**
		* @brief		获取算法库版本号
		* @return		const char*: 版本号
	*/
	OBJECTDETECT_API const char* OD_GetVersion();

	/**
		* @brief		创建目标检测模型对象
		* @param		models_path: 模型文件夹models的路径
		* @param		num_thread: ncnn模型推理的线程数
	*/
	OBJECTDETECT_API void* OD_Create(const char* models_path, int num_thread);

	/**
		* @brief		运行目标检测模型，推理单张图像
		* @param		p_data:	OpenCV图像数据指针
		* @param		nw:		图像宽度
		* @param		nh:		图像高度
		* @param		channels: 图像通道数
		* @param		p_det: 目标检测模型对象的句柄，来自OD_Create
		* @return		void* : 检测结果的结构体指针，a sample case (todo )
	*/
	OBJECTDETECT_API void* OD_Run(unsigned char* p_data, int nw, int nh, int channels, void* p_det);

	/**
		* @brief		释放单次目标检测申请的内存
		* @param		p_res: 来自OD_Run返回的结构体指针
	*/
	OBJECTDETECT_API void OD_ReleaseResult(void* p_res);

	/**
		* @brief		释放目标检测对象句柄
		* @param		p_det: 目标检测模型对象的句柄，来自OD_Create
	*/
	OBJECTDETECT_API void OD_Release(void* p_det);

#ifdef __cplusplus
}
#endif

