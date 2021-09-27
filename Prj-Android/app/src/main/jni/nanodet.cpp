// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "nanodet.h"

#include <opencv2/opencv.hpp>

#include "cpu.h"

#include <android/log.h>
#define TAG "JNI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)

#include "model.id.h"
#include "model.mem.h"

NanoDet::NanoDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int NanoDet::load(AAssetManager* mgr, bool use_gpu)
{
    // 把原有的环境清空一下
    nanodet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    nanodet.opt = ncnn::Option();

#if NCNN_VULKAN
    nanodet.opt.use_vulkan_compute = use_gpu;
#endif

//    nanodet.opt.num_threads=2;
    nanodet.opt.num_threads = ncnn::get_big_cpu_count();
    nanodet.opt.blob_allocator = &blob_pool_allocator;
    nanodet.opt.workspace_allocator = &workspace_pool_allocator;

    // 加载模型和设置模型对应的一些参数
    int ret = nanodet.load_param(model_param_bin);
    if(ret != sizeof(model_param_bin)) {
        LOGD("error in load param");
    }
    ret = nanodet.load_model(model_bin);
    if(ret != sizeof(model_bin)) {
        LOGD("error in load model");
    }

    return 0;
}

int NanoDet::detect(const cv::Mat& bgr, cv::Mat& mask, int& t)
{
    double t0 = ncnn::get_current_time();

    cv::Mat src;
    cv::resize(bgr, src, cv::Size(192, 288),0,0,cv::INTER_AREA);
    ncnn::Mat in = ncnn::Mat::from_pixels(src.data, ncnn::Mat::PIXEL_BGR, src.cols, src.rows);
    ncnn::Mat out;

    in.substract_mean_normalize(mean, norm);
    ncnn::Extractor ex = nanodet.create_extractor();
    ex.input(model_param_id::BLOB_inputs_3_288_192, in);
    ex.extract(model_param_id::BLOB_output_1_288_192, out);
    out.substract_mean_normalize(0, norm2);

    mask = cv::Mat::zeros(cv::Size(192, 288), CV_8UC1);
    out.to_pixels(mask.data, ncnn::Mat::PIXEL_GRAY);
    cv::resize(mask, mask, bgr.size(), 0, 0, cv::INTER_CUBIC);

    double t1 = ncnn::get_current_time();
    t = (int)(t1-t0);

    return 0;
}

int NanoDet::draw(cv::Mat& bgr, cv::Mat& mask, int& t)
{
    double t0 = ncnn::get_current_time();

    cv::Mat binForeMask;
    cv::Mat blurForeMask, blurForeMaskF;
    cv::Mat blurBackMask;
    cv::Mat bgrF;
    cv::Mat mix, mixF;


//    blurForeMask = mask.clone();
//    blurForeMask.setTo(0,blurForeMask<127);

    // 计算二值化前景mask
//    cv::threshold(mask, binForeMask, 130, 255, cv::THRESH_BINARY);
    // 计算边沿模糊后的mask

//    cv::blur(binForeMask,blurForeMask,cv::Size(5,5));
//    blurForeMask = binForeMask;
    // 先膨胀小一点
    cv::Mat dilate_element = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));
    cv::dilate(mask,mask,dilate_element);
    // 后腐蚀大一点
    cv::Mat erode_element = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(11,11));
    cv::erode(mask,mask,erode_element);

    cv::threshold(mask, binForeMask, 130, 255, cv::THRESH_BINARY);

    cv::blur(binForeMask,blurForeMask,cv::Size(7,7));


    // 计算取反的模糊后景mask
    blurBackMask = ~blurForeMask;
    // 前后景mask转三通道
    cv::cvtColor(blurForeMask,blurForeMask,cv::COLOR_GRAY2RGB);
    cv::cvtColor(blurBackMask,blurBackMask,cv::COLOR_GRAY2RGB);



    // 前景mask和真实图像进行叠加
    // 第一种叠加做法
//    blurForeMask.convertTo(blurForeMaskF,CV_32FC3,1/255.0,0);
//    bgr.convertTo(bgrF,CV_32FC3);
//    mixF = bgrF.mul(blurForeMaskF);
//    mixF.convertTo(mix,CV_8UC3);
    // 第二种叠加做法
//    blurForeMask.convertTo(blurForeMaskF,CV_16UC3);
//    bgr.convertTo(bgrF,CV_16UC3);
//    mixF = bgrF.mul(blurForeMaskF);
//    mixF.convertTo(mix,CV_8UC3,1/255.0,0);
    // 第三种叠加做法
    blurForeMask.convertTo(blurForeMaskF,CV_16UC3);
    bgr.convertTo(bgrF,CV_16UC3);
    mixF = bgrF.mul(blurForeMaskF);
    mixF = mixF / 255;
    mixF.convertTo(mix,CV_8UC3);





    bgr = mix + blurBackMask;

//    cv::cvtColor(mask,bgr,cv::COLOR_GRAY2RGB);

    double t1 = ncnn::get_current_time();
    t = (int)(t1-t0);

    return 0;
}
