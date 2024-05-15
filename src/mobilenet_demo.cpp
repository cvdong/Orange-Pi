// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rknn_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}
// 从文件中读取二进制模型数据
// 参数：filename：模型文件名，model_size：模型大小
// 返回值：模型数据指针
static unsigned char *load_model(const char *filename, int *model_size)
{
  FILE *fp = fopen(filename, "rb");
  if (fp == nullptr)
  {
    printf("fopen %s fail!\n", filename);
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  int model_len = ftell(fp);
  unsigned char *model = (unsigned char *)malloc(model_len); // 申请模型大小的内存，返回指针
  fseek(fp, 0, SEEK_SET);
  if (model_len != fread(model, 1, model_len, fp))
  {
    printf("fread %s fail!\n", filename);
    free(model);
    return NULL;
  }
  *model_size = model_len;
  if (fp)
  {
    fclose(fp);
  }
  return model;
}

static int rknn_GetTop(float *pfProb, float *pfMaxProb, uint32_t *pMaxClass, uint32_t outputCount, uint32_t topNum)
{
  uint32_t i, j;

#define MAX_TOP_NUM 20
  if (topNum > MAX_TOP_NUM)
    return 0;

  memset(pfMaxProb, 0, sizeof(float) * topNum);
  memset(pMaxClass, 0xff, sizeof(float) * topNum);

  for (j = 0; j < topNum; j++)
  {
    for (i = 0; i < outputCount; i++)
    {
      if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
          (i == *(pMaxClass + 4)))
      {
        continue;
      }

      if (pfProb[i] > *(pfMaxProb + j))
      {
        *(pfMaxProb + j) = pfProb[i];
        *(pMaxClass + j) = i;
      }
    }
  }

  return 1;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
  const int MODEL_IN_WIDTH = 224;
  const int MODEL_IN_HEIGHT = 224;
  const int MODEL_IN_CHANNELS = 3;

  rknn_context ctx = 0;
  int ret;
  int model_len = 0;
  unsigned char *model;

  const char *model_path = argv[1];
  const char *img_path = argv[2];

  if (argc != 3)
  {
    printf("Usage: %s <rknn model> <image_path> \n", argv[0]);
    return -1;
  }

  // ======================= 读取图片 ===================
  cv::Mat orig_img = imread(img_path, cv::IMREAD_COLOR);
  if (!orig_img.data)
  {
    printf("cv::imread %s fail!\n", img_path);
    return -1;
  }
  // BGR to RGB
  cv::Mat orig_img_rgb;
  cv::cvtColor(orig_img, orig_img_rgb, cv::COLOR_BGR2RGB);

  // Resize to MODEL_IN_WIDTH*MODEL_IN_HEIGHT：224*224
  cv::Mat img = orig_img_rgb.clone();
  if (orig_img.cols != MODEL_IN_WIDTH || orig_img.rows != MODEL_IN_HEIGHT)
  {
    printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, MODEL_IN_WIDTH, MODEL_IN_HEIGHT);
    cv::resize(orig_img, img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), 0, 0, cv::INTER_LINEAR);
  }

  // ======================= 初始化RKNN模型 ===================

  // 输入：模型文件名，模型大小
  // 返回值：模型数据指针
  model = load_model(model_path, &model_len);
  // 初始化RKNN模型
  // 输入：ctx：模型句柄，model：模型数据指针，model_len：模型大小，flag：0，reserverd：NULL
  // 返回值：<0：失败
  ret = rknn_init(&ctx, model, model_len, 0, NULL);
  if (ret < 0)
  {
    printf("rknn_init fail! ret=%d\n", ret);
    return -1;
  }

  // ======================= 获取模型输入输出信息 ===================
  // ********** 输入输出数量 **********
  rknn_input_output_num io_num;
  // 使用rknn_query函数获取模型输入输出数量
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC)
  {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  // 打印模型输入输出数量
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  // ********** 输入输出属性 **********
  printf("input tensors:\n");
  // 使用rknn_tensor_attr结构体存储模型输入属性
  rknn_tensor_attr input_attrs[io_num.n_input];
  // 初始化，将input_attrs中前sizeof(input_attrs)个字节用0替换
  memset(input_attrs, 0, sizeof(input_attrs));

  // 遍历模型所有输入（网络可能有多个输入，这里为了兼容多输入，使用for循环遍历）
  for (int i = 0; i < io_num.n_input; i++)
  {
    // 设置模型输入索引
    input_attrs[i].index = i;
    // 使用rknn_query函数获取模型输入信息，存储在input_attrs
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
    {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    // 打印模型输入信息
    dump_tensor_attr(&(input_attrs[i]));

    printf("output tensors:\n");
    // 使用rknn_tensor_attr结构体存储模型输出信息
    rknn_tensor_attr output_attrs[io_num.n_output];
    // 初始化，将output_attrs中前sizeof(output_attrs)个字节用0替换
    memset(output_attrs, 0, sizeof(output_attrs));

    // 遍历模型所有输出
    for (int i = 0; i < io_num.n_output; i++)
    {
      output_attrs[i].index = i;
      ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
      if (ret != RKNN_SUCC)
      {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
      }
      // 打印模型输出信息
      dump_tensor_attr(&(output_attrs[i]));
    }

    // ======================= 设置模型输入 ===================
    // 使用rknn_input结构体存储模型输入信息
    rknn_input inputs[1];
    // 初始化，将inputs中前sizeof(inputs)个字节用0替换
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;                                                     // 设置模型输入索引
    inputs[0].type = RKNN_TENSOR_UINT8;                                      // 设置模型输入类型
    inputs[0].size = img.cols * img.rows * img.channels() * sizeof(uint8_t); // 设置模型输入大小
    inputs[0].fmt = RKNN_TENSOR_NHWC;                                        // 设置模型输入格式：NHWC
    inputs[0].buf = img.data;                                                // 设置模型输入数据

    // 使用rknn_inputs_set函数设置模型输入
    // 输入：ctx：模型句柄，io_num.n_input：模型输入数量，inputs：模型输入信息
    // 返回值：<0：失败
    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0)
    {
      printf("rknn_input_set fail! ret=%d\n", ret);
      return -1;
    }

    // ======================= 推理 ===================
    printf("rknn_run\n");
    // 使用rknn_run函数运行RKNN模型
    // 输入：ctx：模型句柄，nullptr：保留参数
    // 返回值：<0：失败
    ret = rknn_run(ctx, nullptr);
    if (ret < 0)
    {
      printf("rknn_run fail! ret=%d\n", ret);
      return -1;
    }

    // ======================= 获取模型输出 ===================
    // 使用rknn_output结构体存储模型输出信息
    rknn_output outputs[1];
    // 初始化，将outputs中前sizeof(outputs)个字节用0替换
    memset(outputs, 0, sizeof(outputs));
    // 设置模型输出类型为float
    outputs[0].want_float = 1;

    // 使用rknn_outputs_get函数获取模型输出
    // 输入：ctx：模型句柄，1：模型输出数量，outputs：模型输出信息，nullptr：保留参数
    // 返回值：<0：失败
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0)
    {
      printf("rknn_outputs_get fail! ret=%d\n", ret);
      return -1;
    }

    // ======================= 后处理 ===================
    // 遍历模型所有输出
    for (int i = 0; i < io_num.n_output; i++)
    {
      uint32_t MaxClass[5];
      float fMaxProb[5];
      float *buffer = (float *)outputs[i].buf;        // 模型输出数据
      uint32_t sz = outputs[i].size / 4;              // 模型输出大小，除以4是因为模型输出类型为float
      rknn_GetTop(buffer, fMaxProb, MaxClass, sz, 5); // 获取模型输出的Top5

      printf(" --- Top5 ---\n");
      for (int i = 0; i < 5; i++)
      {
        printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);
      }
    }

    // ======================= 释放输出缓冲区 ===================
    // 释放rknn_outputs_get获取的输出
    // 输入：ctx：模型句柄，1：模型输出数量，outputs：模型输出信息（数组）
    // 返回值：<0：失败，>=0：成功
    rknn_outputs_release(ctx, 1, outputs);
    if (ret < 0)
    {
      printf("rknn_outputs_release fail! ret=%d\n", ret);
      return -1;
    }
    else if (ctx > 0)
    {
      // ======================= 释放RKNN模型 ===================
      rknn_destroy(ctx);
    }
    // ======================= 释放模型数据 ===================
    if (model)
    {
      free(model);
    }
    return 0;
  }
}
