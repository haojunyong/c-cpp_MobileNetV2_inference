#include"inference.h"
#include<time.h>
#include<fstream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include<string>
using namespace std;
/*神经网络输入图片推断功能实现*/
//类别名称
struct label 
{
    string label;
    string class_chinese;

}classes[1000];

struct image_file
{
    string label;
    string filename;
};

//初始化1000个类别名称
static void classes_name_init()
{
    ifstream in("classes_name.txt");
    for (int i = 0; i < 1000; i++)
    {
        in >> classes[i].label>>classes[i].class_chinese;
    }
    in.close();
}

static void make_image(image* img, int w, int h, int c)
{
    /**
     * Make image
     *
     * Input:
     *      w, h, c
     * Output:
     *      img
     * */
    int size;
    img->w = w;
    img->h = h;
    img->c = c;
    size = h * w * c;
    img->data = (float*)malloc(size * sizeof(float));
}

//下载要识别的图片数据
static image load_image(char* filename, int W, int H, int channels)
{
    /**
     * load image from file
     *
     * Input:
     *      filename
     *      channels
     * Return:
     *      image
     * */
    int w, h, c;
    unsigned char* data = stbi_load(filename, &w, &h, &c, channels);
    if (!data)
    {
        printf("Error! Can't load image %s! \n", filename);
        exit(0);
    }
    if (channels)
    {
        c = channels;
    }
    image img;
    make_image(&img, W, H, channels);
    register int dst_idx, src_idx;
    //中心裁剪出224*224的图片大小
    //int w_c = (w - 224) / 2, h_c = (h - 224) / 2;
    //左上裁剪
    //int w_c = 0, h_c = 0;
    for (int k = 0; k < channels; k++)
    {
        for (int j = 0; j < H; j++)
        {
            for (int i = 0; i < W; i++)
            {
                dst_idx = i + W * j + W * H * k;
                src_idx = (i + w * j) * c + k;
                img.data[dst_idx] = (float)data[src_idx]/255;
            }
        }
    }
    free(data);
    return img;
}

//imagenet归一化
//mean=0.485 0.456 0.406 std=0.229 0.224 0.225
static void imagenet_normalize(image* image)
{
    float mean[3] = { 0.485,0.456,0.406 }, std[3] = { 0.229,0.224,0.225 };
    int C = image->c, H = image->h, W = image->w;
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
            {
                image->data[W * H * c + W * h + w] = (image->data[W * H * c + W * h + w] - mean[c]) / std[c];
            }

}

//用于获得imagelist文件中的下一张图片，即下一次推断任务的图片数据
static void get_next_batch(int n, float* X, int* Y, int w, int h, int c, int CLASSE, FILE* fp)
{
    /**
     * sample next batch of data for inference model
     *
     * Input:
     *      n = batchsize 默认为1
     *      w, h, c
     *      CLASSE 0-9
     *      fp
     * Output:
     *      X   [n,c,h,w]
     *      Y   [n]
     * */
    image img;
    make_image(&img, w, h, c);
    int label, l;
    char imgpath[100];
    int imagesize = w * h * c;
    for (int i = 0; i < n; i++)
    {
        if (feof(fp))
            rewind(fp);//rewind 指向文件开头
        fscanf_s(fp, "%d %s", &label, &imgpath, 100);
        Y[i] = label;
        img = load_image(imgpath, w, h, c);
        l = i * imagesize;
        memcpy(X + l, img.data, imagesize * sizeof(float));
    }

}

static void get_next_img(float* input, int w, int h, int c, string filename)
{

    image img;
    int imagesize = w * h * c;
    string s = "test_image/" + filename;
    img = load_image((char*)s.c_str(), w, h, c);
    imagenet_normalize(&img);
    memcpy(input, img.data, imagesize * sizeof(float));
}

static void load_all_img(image_file* file_list, image* all_image, int num,int w,int h,int c)
{
    string f;
    for (int i = 0; i < num; i++)
    {
        f = "test_image/" + file_list[i].filename;
        all_image[i] = load_image((char*)f.c_str(), w, h, c);
    }
}

//处理后的图片存为txt，用于在python中验证处理是否正确
static void image_to_file(image* img)
{
    ofstream out("imagefile2.txt");
    int l = img->h * img->w * img->c;
    for (int i = 0; i < l; i++)
        out << img->data[i]<<endl;
    out.close();
}

//用于最后一层的输出比较，对于0-n判断哪个输出神经元的值最大，即最后的识别结果
static int argmax(float* arr, int n)
{
    /**
     * Return the index of max-value among arr ~ arr+n
     *
     * Input:
     *      arr
     * Output:
     * Return:
     *      the index of max-value
     * */
    int   idx = 0;
    float max = arr[0];
    for (int p = 1; p < n; p++)
    {
        if (arr[p] > max)
        {
            idx = p;
            max = arr[p];
        }
    }
    return idx;
}

//void alexnet_inference(alexnet* net, int epochs)
//{
//    int f = net->batchsize * net->conv1.in_channels * net->conv1.in_w * net->conv1.in_h;
//    net->input = (float*)malloc(f * sizeof(float));
//    int* batch_Y = (int*)malloc(net->batchsize * sizeof(int));
//    int ed = 0;
//    float acc;
//    FILE* fp;
//    printf("start load_image ........\n");
//    fopen_s(&fp, "images.list", "r");
//    printf("start forward_alexnet inference ........\n");
//    for (int e = 0; e < epochs; e++)
//    {
//        printf("-----------------------------%d---------------------------------\n", e + 1);
//
//        get_next_batch(net->batchsize, net->input, batch_Y, net->conv1.in_w, net->conv1.in_h, net->conv1.in_channels, net->fc3.out_units, fp);
//        forward_alexnet(net);
//        int pred = argmax(net->output, OUT_LAYER);
//        free_alexnet_layer(net);
//        for (int i = 0; i < net->batchsize; i++)
//        {
//            printf("prediction: %d  label: %d\n", pred, batch_Y[i]);//用于显示识别结果与真实结果对比
//            if (pred == batch_Y[i])
//            {
//                ed++;
//            }
//        }
//    }
//    acc = (float)ed / (float)epochs * 100.0;
//    printf(" test acc: %0.2f%%\n", acc);
//}

void mobilenet_inference1(mobilenet* net, char* filename)
{
    malloc_mobilenet_pass(net);
    classes_name_init();
    image img;
    clock_t start, finish;
    float duration;
    //printf("start make_image ........\n");
    //make_image(&img, IN_L, IN_L, IN_CHANNELS);
    printf("start load_image ........\n");
    /*测试推断所用时间*/
    start = clock();
    img = load_image(filename, IN_L, IN_L, IN_CHANNELS);
    imagenet_normalize(&img);
    //image_to_file(&img);
    finish = clock();
    duration = (float)((float)finish - (float)start) / CLOCKS_PER_SEC;
    printf("加载一张图片用时：%0.3f s\n", duration);
    net->input = img.data;
    printf("start forward_mobilenet........\n");
    start = clock();
    forward_mobilenet(net);
    int pred = argmax(net->output, F20_CHANNELS);
    //printf("prediction: %d\n", pred);
    cout << "prediction: " << classes[pred].label<<" "<<classes[pred].class_chinese<<endl;
    finish = clock();
    duration = (float)((float)finish - (float)start) / CLOCKS_PER_SEC;
    printf("mobilenet推断一张图片用时：%0.3f s\n", duration);
}

image_file file[2000];
void mobilenet_inference(mobilenet* net, int epochs)
{
    classes_name_init();
    int f = IN_L * IN_L * IN_CHANNELS;
    net->input = (float*)malloc(f * sizeof(float));
    //int* batch_Y = (int*)malloc(net->batchsize * sizeof(int));
    int ed = 0;
    float acc;
    FILE* fp;
    printf("start load_image ........\n");
    ifstream img_list("image_list.txt");
    for (int i = 0; i < epochs; i++)
    {
        img_list >> file[i].filename >> file[i].label;
    }
    img_list.close();
    malloc_mobilenet_pass(net);
    printf("start forward_mobilenet inference ........\n");
    for (int e = 0; e < epochs; e++)
    {
        init_mobilenet_layer(net);
        printf("-----------------------------%d---------------------------------\n", e + 1);
        get_next_img(net->input, IN_L, IN_L, IN_CHANNELS, file[e].filename);
        //get_next_batch(net->batchsize, net->input, batch_Y, net->conv1.in_w, net->conv1.in_h, net->conv1.in_channels, net->fc3.out_units, fp);
        forward_mobilenet(net);
        int pred = argmax(net->output, F20_CHANNELS);
        for (int i = 0; i < net->batchsize; i++)
        {
            cout << "prediction: " << classes[pred].label << " " << classes[pred].class_chinese << " label: " << file[e].label << endl;//用于显示识别结果与真实结果对比
            if (classes[pred].label == file[e].label)
            {
                ed++;
            }
        }
    }
    acc = (float)ed / (float)epochs * 100.0;
    printf(" test acc: %0.2f%%\n", acc);
}