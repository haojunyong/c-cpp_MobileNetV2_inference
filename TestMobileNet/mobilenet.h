#ifndef MOBILENET_H
#define MOBILENET_H


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

//#ifndef TIME_RECORD
//#define TIME_RECORD
//#endif

//定义MobileNet模型结构
//  Definition of model shape

//输出通道数
#define IN_CHANNELS 3
#define F0_CHANNELS 32
#define F1_CHANNELS 16
#define F2_CHANNELS 24
#define F3_CHANNELS 24
#define F4_CHANNELS 32
#define F5_CHANNELS 32
#define F6_CHANNELS 32
#define F7_CHANNELS 64
#define F8_CHANNELS 64
#define F9_CHANNELS 64
#define F10_CHANNELS 64
#define F11_CHANNELS 96
#define F12_CHANNELS 96
#define F13_CHANNELS 96
#define F14_CHANNELS 160
#define F15_CHANNELS 160
#define F16_CHANNELS 160
#define F17_CHANNELS 320
#define F18_CHANNELS 1280
#define F19_CHANNELS 1280
#define F20_CHANNELS 1000

//卷积核大小
#define F0_KERNEL_L 3
#define PW_KERNEL_L 1
#define DW_KERNEL_L 3
#define F18_KERNEL_L 1
#define F19_KERNEL_L 7

//各层步长
#define PW_STRIDES 1
#define F0_STRIDES 2
#define F1_STRIDES 1
#define F2_STRIDES 2
#define F3_STRIDES 1
#define F4_STRIDES 2
#define F5_STRIDES 1
#define F6_STRIDES 1
#define F7_STRIDES 2
#define F8_STRIDES 1
#define F9_STRIDES 1
#define F10_STRIDES 1
#define F11_STRIDES 1
#define F12_STRIDES 1
#define F13_STRIDES 1
#define F14_STRIDES 2
#define F15_STRIDES 1
#define F16_STRIDES 1
#define F17_STRIDES 1
#define F18_STRIDES 1
#define F19_STRIDES 1

//各层填充大小
#define F0_PADDING 1
#define PW_PADDING 0
#define DW_PADDING 1
#define F18_PADDING 0
#define F19_PADDING 0

//各层升维倍数
#define F1_T 1
#define F2_T 6
#define F3_T 6
#define F4_T 6
#define F5_T 6
#define F6_T 6
#define F7_T 6
#define F8_T 6
#define F9_T 6
#define F10_T 6
#define F11_T 6
#define F12_T 6
#define F13_T 6
#define F14_T 6
#define F15_T 6
#define F16_T 6
#define F17_T 6


//feature map大小自动计算
#define IN_L 224



#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))



/*-------------------------各操作定义------------------*/

//定义非线性/激活函数操作结构 
typedef struct nonlinear_op {
    float* input;
    float* output;
    int units;

    int batchsize;
} nonlinear_op;

//定义卷积操作结构：输入、输出、权重、偏置，以及需要用到的输入 输出通道数 卷积核大小 填充大小 步长等
typedef struct conv_op {
    float* input;
    float* output;
    float* weights;
    float* bias;
    float* input_col;

    int in_channels, out_channels;
    int kernel_size; int padding; int stride;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;
    int filter;

    int batchsize;
} conv_op;

//定义bn归一化操作结构，输入、输出、gamma、beta
typedef struct batch_norm_op {
    float* input;
    float* output;
    float* weight;
    float* bias;
    float* mean;
    float* var;
    int channels;
    int w,h;
    int units;
    float eps = 1e-5;

    int batchsize;

}batch_norm_op;

//定义残差连接操作
typedef struct res_connect_op {
    float* input;
    float* add_input;
    float* output;
    int units;
    int in_channels, out_channels;
}res_connect_op;

//定义bottleneck层的操作结构
typedef struct bottleneck_op
{
    float* input;
    float* output;
    int stride;
    int in_channels, out_channels;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;
    int t;//升维倍数
    conv_op conv[3];
    batch_norm_op bn[3];
    nonlinear_op relu6[2];
    res_connect_op res_con;

}bottleneck_op;

//定义最大池化操作结构，包括：输入\输出大小及通道数，池化大小及步长
typedef struct max_pooling_op {
    float* input;
    float* output;
    int channels;
    int kernel_size; int stride;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;

    int batchsize;
} max_pooling_op;

//定义平均池化操作结构，包括：输入\输出大小及通道数，池化大小及步长
typedef struct avg_pooling_op {
    float* input;
    float* output;
    int channels;
    int kernel_size; int stride;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;

    int batchsize;
} avg_pooling_op;


//定义全连接操作结构，包括：输入\输出大小，权重及偏置
typedef struct fc_op {
    float* input;
    float* output;
    float* weights;
    float* bias;
    int in_units, out_units;

    int batchsize;
} fc_op;


//定义mobilenet_v2各层结构（操作）
typedef struct mobilenet {
    float* input;
    float* output;
    int batchsize;

    //第1层 卷积层
    conv_op f0_conv;
    batch_norm_op f0_bn;
    nonlinear_op f0_relu6;

    //第2层 瓶颈层1(省略了第一个pw卷积）
    //conv_op f1_conv[2];
    //batch_norm_op f1_bn[2];
    //nonlinear_op f1_relu6;
    bottleneck_op f1;

    //第3层 瓶颈层2_1
    bottleneck_op f2;

    //第4层 瓶颈层2_2
    bottleneck_op f3;

    //第5层 瓶颈层3_1
    bottleneck_op f4;

    //第6层 瓶颈层3_2
    bottleneck_op f5;

    //第7层 瓶颈层3_3
    bottleneck_op f6;

    //第8层 瓶颈层4_1
    bottleneck_op f7;

    //第9层 瓶颈层4_2
    bottleneck_op f8;

    //第10层 瓶颈层4_3
    bottleneck_op f9;

    //第11层 瓶颈层4_4
    bottleneck_op f10;

    //第12层 瓶颈层5_1
    bottleneck_op f11;

    //第13层 瓶颈层5_2
    bottleneck_op f12;

    //第14层 瓶颈层5_3
    bottleneck_op f13;

    //第15层 瓶颈层6_1
    bottleneck_op f14;

    //第16层 瓶颈层6_2
    bottleneck_op f15;

    //第17层 瓶颈层6_3
    bottleneck_op f16;

    //第18层 瓶颈层7_1
    bottleneck_op f17;

    //第19层 卷积层
    conv_op f18_conv;
    batch_norm_op f18_bn;
    nonlinear_op f18_relu6;

    //第20层 池化层
    avg_pooling_op f19_ap;

    //第21层 全连接层
    fc_op f20_fc;


}mobilenet;



/*-------------------------各操作前向传播实现------------------*/
//常见的激活函数
//relu函数
static void relu_op_forward(nonlinear_op* op);

static void relu6_op_forward(nonlinear_op* op);

static void sigmoid_op_forward(nonlinear_op* op);

//常规卷积
static void conv_op_forward(conv_op* op);

//mobilenet中的dw卷积
static void conv_dw_op_forward(conv_op* op);

//mobilenet中的bn归一化
static void batch_norm_op_forward(batch_norm_op* op);

//mobilenet中的res_connect
static void res_connect_op_forward(res_connect_op* op);

//mobilenet中的bottleneck层
static void bottleneck_op_forward(bottleneck_op* op);

//最大池化操作函数
static void max_pooling_op_forward(max_pooling_op* op);

//平均池化
static void avg_pooling_op_forward(avg_pooling_op* op);

//全连接
static void fc_op_forward(fc_op* op);


/*-------------------------神经网络结构初始化---------------------*/
//初始化各层计算中需要的参数，例如输入/输出通道数，步长，填充大小，卷积核大小等等

//卷积操作参数初始化
static void init_conv_op(conv_op* op, int in_channels, int out_channels, int stride, int padding, int kernel_size, int input_shape, bool is_dw);

//bn操作参数初始化
static void init_bn_op(batch_norm_op* op, int channels, int shape);

//瓶颈层操作参数初始化 t为升维倍数
static void init_bottleneck_op(bottleneck_op* op, int in_channels, int out_channels, int stride, int input_shape, int t);

//池化操作参数初始化
static void init_avg_pool_op(avg_pooling_op* op, int channels, int stride, int kernel_size, int input_shape);

//设置mobilenet网络结构参数
void setup_mobilenet(mobilenet* net);



/*-------------------------神经网络模型参数初始化---------------------*/

//分配模型参数空间
static void calloc_conv_weights(conv_op* op);
static void calloc_bn_weights(batch_norm_op* op);
static void calloc_fc_weights(fc_op* op);
static void calloc_bottleneck_weights(bottleneck_op* op);
void malloc_mobilenet(mobilenet* net);


//从文件中加载已经训练好的该卷积层的对应参数
static void load_conv_weights(conv_op* op, FILE* fp);
static void load_conv_weights(conv_op* op, string filename);
static void load_conv_weights(conv_op* op, ifstream& file);

//从文件中加载已经训练好的bn层的对应参数
static int load_bn_weights(batch_norm_op* op, string filename[], int index);
static void load_bn_weights(batch_norm_op* op, ifstream& file);

//从文件中加载已经训练好的该全连接层的对应参数
static void load_fc_weights(fc_op* op, FILE* fp);
static void load_fc_weights(fc_op* op, string wf, string bf);
static void load_fc_weights(fc_op* op, ifstream& file);

static int load_bottleneck_weights(bottleneck_op* op, string filename[], int index);
static void load_bottleneck_weights(bottleneck_op* op, ifstream& file);

// 加载已经训练好的模型参数
void load_mobilenet(mobilenet* net, string filename);



//释放模型空间
static void free_conv_weights(conv_op* op);
static void free_bn_weights(batch_norm_op* op);
static void free_bottleneck_weights(bottleneck_op* op);
static void free_fc_weights(fc_op* op);
void free_mobilenet(mobilenet* net);


/*-------------------------前向传播实现---------------------*/
//为中间结果分配空间
static void malloc_bottleneck_pass(bottleneck_op* op);
void malloc_mobilenet_pass(mobilenet* net);


void forward_mobilenet(mobilenet* net);

//初始化中间结果
static void init_bottleneck_pass(bottleneck_op* op);
void init_mobilenet_layer(mobilenet* net);

//释放中间结果空间
static void free_bottleneck_layer(bottleneck_op* op);
void free_mobilenet_layer(mobilenet* net);


void start_time_record(string filename);
void end_time_record();
#endif