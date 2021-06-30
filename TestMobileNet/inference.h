#ifndef INFERRENCE_H
#define INFERRENCE_H


#include"mobilenet.h"

/*神经网络输入图片推断功能实现*/

typedef struct {
    int w;
    int h;
    int c;
    float* data;
} image;

static void make_image(image* img, int w, int h, int c);

//加载要识别的图片数据
static image load_image(char* filename, int W, int H, int channels);

//用于获得imagelist文件中的下一张图片，即下一次推断任务的图片数据
static void get_next_batch(int n, float* X, int* Y, int w, int h, int c, int CLASSE, FILE* fp);

static void get_next_img(float* input, int w, int h, int c, string filename);
//imagenet归一化
static void imagenet_normalize(image* image);

//处理后的图片存为txt，用于在python中验证处理是否正确
static void image_to_file(image* img);

//用于alexnet最后一层的输出比较，对于0-n判断哪个输出神经元的值最大，即最后的识别结果
static int argmax(float* arr, int n);

void mobilenet_inference1(mobilenet* net, char* filename);

void mobilenet_inference(mobilenet* net, int epoch);
#endif