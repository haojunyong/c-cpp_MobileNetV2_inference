#ifndef INFERRENCE_H
#define INFERRENCE_H


#include"mobilenet.h"

/*����������ͼƬ�ƶϹ���ʵ��*/

typedef struct {
    int w;
    int h;
    int c;
    float* data;
} image;

static void make_image(image* img, int w, int h, int c);

//����Ҫʶ���ͼƬ����
static image load_image(char* filename, int W, int H, int channels);

//���ڻ��imagelist�ļ��е���һ��ͼƬ������һ���ƶ������ͼƬ����
static void get_next_batch(int n, float* X, int* Y, int w, int h, int c, int CLASSE, FILE* fp);

static void get_next_img(float* input, int w, int h, int c, string filename);
//imagenet��һ��
static void imagenet_normalize(image* image);

//������ͼƬ��Ϊtxt��������python����֤�����Ƿ���ȷ
static void image_to_file(image* img);

//����alexnet���һ�������Ƚϣ�����0-n�ж��ĸ������Ԫ��ֵ��󣬼�����ʶ����
static int argmax(float* arr, int n);

void mobilenet_inference1(mobilenet* net, char* filename);

void mobilenet_inference(mobilenet* net, int epoch);
#endif