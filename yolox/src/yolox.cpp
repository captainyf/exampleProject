#include"yolox.h"
#include"utils.h"

#include <stdio.h>
#include <stdarg.h>
#include <vector>

#define YOLOX_NMS_THRESH  0.45 // nms threshold
#define YOLOX_CONF_THRESH 0.25 // threshold of bounding box prob
#define YOLOX_TARGET_SIZE 640  // target image size after resize, might use 416 for small model

// YOLOX use the same focus in yolov5
class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)


Yolox::Yolox()
{
    
}

Yolox::~Yolox()
{
    if (is_output_result_txt_)
    {
        fclose(result_txt_);
    }
}

void Yolox::Logger(const char* format, ...)
{
    if (!(is_output_console_ || is_output_result_txt_)) return;
    char* buffer = (char*)malloc(8192);
    va_list args;
    va_start(args, format);
    vsprintf(buffer, format, args);
    va_end(args);
    if (is_output_console_)
        printf("%s", buffer);
    if (is_output_result_txt_)
        fprintf(result_txt_, "%s", buffer);
    free(buffer);
}

void Yolox::EnableResultTxt()
{
    if (is_output_result_txt_)
    {
        std::string result_txt_path = "log_object_detect.txt";
        result_txt_ = fopen(result_txt_path.c_str(), "w");
    }
}

bool Yolox::Init(const std::string model_path, unsigned int num_thread)
{
    EnableResultTxt();

    Logger("=====Config Vulkan=====\n");
    yolox_.opt.use_vulkan_compute = true;
    // yolox_.opt.use_bf16_storage = true;

    Logger("=====Load NCNN Model=====\n");
    std::string param_path, bin_path;
    param_path = model_path + "/yolox.param";
    bin_path = model_path + "/yolox.bin";
    bool has_param_file = isFileExists(param_path);
    if (!has_param_file)
    {
        Logger("Yolox param file not found: %s\n", param_path.c_str());
        return false;
    }
    bool has_bin_file = isFileExists(bin_path);
    if (!has_bin_file)
    {
        Logger("Yolox bin file not found: %s\n", bin_path.c_str());
        return false;
    }

    Logger("Register YoloV5Focus Layer\n");
    yolox_.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);

    Logger("Load param file\n");
    yolox_.load_param(param_path.c_str());
    Logger("Load bin file\n");
    yolox_.load_model(bin_path.c_str());

    Logger("Config number of thread\n");
    yolox_.opt.num_threads = num_thread;

    return true;
}

std::vector<Object> Yolox::Run(const cv::Mat& img)
{
    cv::Mat src = img.clone();

    int img_w = src.cols;
    int img_h = src.rows;
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)YOLOX_TARGET_SIZE / w;
        w = YOLOX_TARGET_SIZE;
        h = h * scale;
    }
    else
    {
        scale = (float)YOLOX_TARGET_SIZE / h;
        h = YOLOX_TARGET_SIZE;
        w = w * scale;
    }
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(src.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, w, h);

    //pad to YOLOX_TARGET_SIZE rectangle
    int wpad = YOLOX_TARGET_SIZE - w;
    int hpad = YOLOX_TARGET_SIZE - h;
    ncnn::Mat in_pad;
    // different from yolov5, yolox only pad on bottom and right side,
    // which means users don't need to extra padding info to decode boxes coordinate.
    ncnn::copy_make_border(in, in_pad, 0, hpad, 0, wpad, ncnn::BORDER_CONSTANT, 114.f);

    ncnn::Extractor ex = yolox_.create_extractor();
    ex.input("images", in_pad);
    std::vector<Object> proposals;

    {
        ncnn::Mat out;
        ex.extract("output", out);

        static const int stride_arr[] = { 8, 16, 32 }; // might have stride=64 in YOLOX
        std::vector<int> strides(stride_arr, stride_arr + sizeof(stride_arr) / sizeof(stride_arr[0]));
        std::vector<GridAndStride> grid_strides;
        generate_grids_and_stride(YOLOX_TARGET_SIZE, strides, grid_strides);
        generate_yolox_proposals(grid_strides, out, YOLOX_CONF_THRESH, proposals);
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, YOLOX_NMS_THRESH);

    // get object
    int count = picked.size();
    std::vector<Object> objects(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x) / scale;
        float y0 = (objects[i].rect.y) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return objects;
}



float Yolox::intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

void Yolox::qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void Yolox::qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

void Yolox::nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void Yolox::generate_grids_and_stride(const int target_size, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid = target_size / stride;
        for (int g1 = 0; g1 < num_grid; g1++)
        {
            for (int g0 = 0; g0 < num_grid; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

void Yolox::generate_yolox_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;
    const int num_class = feat_blob.w - 5;
    const int num_anchors = grid_strides.size();

    const float* feat_ptr = feat_blob.channel(0);
    for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++)
    {
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;

        // yolox/models/yolo_head.py decode logic
        //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
        //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        float x_center = (feat_ptr[0] + grid0) * stride;
        float y_center = (feat_ptr[1] + grid1) * stride;
        float w = exp(feat_ptr[2]) * stride;
        float h = exp(feat_ptr[3]) * stride;
        float x0 = x_center - w * 0.5f;
        float y0 = y_center - h * 0.5f;

        float box_objectness = feat_ptr[4];
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = feat_ptr[5 + class_idx];
            float box_prob = box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

        } // class loop
        feat_ptr += feat_blob.w;

    } // point anchor loop
}

//int main(int argc, char** argv)
//{
//    if (argc != 2)
//    {
//        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
//        return -1;
//    }
//
//    const char* imagepath = argv[1];
//
//    cv::Mat m = cv::imread(imagepath, 1);
//    if (m.empty())
//    {
//        fprintf(stderr, "cv::imread %s failed\n", imagepath);
//        return -1;
//    }
//
//    std::vector<Object> objects;
//    detect_yolox(m, objects);
//
//    draw_objects(m, objects);
//
//    return 0;
//}
