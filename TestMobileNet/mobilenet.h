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

//����MobileNetģ�ͽṹ
//  Definition of model shape

//���ͨ����
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

//����˴�С
#define F0_KERNEL_L 3
#define PW_KERNEL_L 1
#define DW_KERNEL_L 3
#define F18_KERNEL_L 1
#define F19_KERNEL_L 7

//���㲽��
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

//��������С
#define F0_PADDING 1
#define PW_PADDING 0
#define DW_PADDING 1
#define F18_PADDING 0
#define F19_PADDING 0

//������ά����
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


//feature map��С�Զ�����
#define IN_L 224



#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))



/*-------------------------����������------------------*/

//���������/����������ṹ 
typedef struct nonlinear_op {
    float* input;
    float* output;
    int units;

    int batchsize;
} nonlinear_op;

//�����������ṹ�����롢�����Ȩ�ء�ƫ�ã��Լ���Ҫ�õ������� ���ͨ���� ����˴�С ����С ������
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

//����bn��һ�������ṹ�����롢�����gamma��beta
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

//����в����Ӳ���
typedef struct res_connect_op {
    float* input;
    float* add_input;
    float* output;
    int units;
    int in_channels, out_channels;
}res_connect_op;

//����bottleneck��Ĳ����ṹ
typedef struct bottleneck_op
{
    float* input;
    float* output;
    int stride;
    int in_channels, out_channels;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;
    int t;//��ά����
    conv_op conv[3];
    batch_norm_op bn[3];
    nonlinear_op relu6[2];
    res_connect_op res_con;

}bottleneck_op;

//�������ػ������ṹ������������\�����С��ͨ�������ػ���С������
typedef struct max_pooling_op {
    float* input;
    float* output;
    int channels;
    int kernel_size; int stride;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;

    int batchsize;
} max_pooling_op;

//����ƽ���ػ������ṹ������������\�����С��ͨ�������ػ���С������
typedef struct avg_pooling_op {
    float* input;
    float* output;
    int channels;
    int kernel_size; int stride;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;

    int batchsize;
} avg_pooling_op;


//����ȫ���Ӳ����ṹ������������\�����С��Ȩ�ؼ�ƫ��
typedef struct fc_op {
    float* input;
    float* output;
    float* weights;
    float* bias;
    int in_units, out_units;

    int batchsize;
} fc_op;


//����mobilenet_v2����ṹ��������
typedef struct mobilenet {
    float* input;
    float* output;
    int batchsize;

    //��1�� �����
    conv_op f0_conv;
    batch_norm_op f0_bn;
    nonlinear_op f0_relu6;

    //��2�� ƿ����1(ʡ���˵�һ��pw�����
    //conv_op f1_conv[2];
    //batch_norm_op f1_bn[2];
    //nonlinear_op f1_relu6;
    bottleneck_op f1;

    //��3�� ƿ����2_1
    bottleneck_op f2;

    //��4�� ƿ����2_2
    bottleneck_op f3;

    //��5�� ƿ����3_1
    bottleneck_op f4;

    //��6�� ƿ����3_2
    bottleneck_op f5;

    //��7�� ƿ����3_3
    bottleneck_op f6;

    //��8�� ƿ����4_1
    bottleneck_op f7;

    //��9�� ƿ����4_2
    bottleneck_op f8;

    //��10�� ƿ����4_3
    bottleneck_op f9;

    //��11�� ƿ����4_4
    bottleneck_op f10;

    //��12�� ƿ����5_1
    bottleneck_op f11;

    //��13�� ƿ����5_2
    bottleneck_op f12;

    //��14�� ƿ����5_3
    bottleneck_op f13;

    //��15�� ƿ����6_1
    bottleneck_op f14;

    //��16�� ƿ����6_2
    bottleneck_op f15;

    //��17�� ƿ����6_3
    bottleneck_op f16;

    //��18�� ƿ����7_1
    bottleneck_op f17;

    //��19�� �����
    conv_op f18_conv;
    batch_norm_op f18_bn;
    nonlinear_op f18_relu6;

    //��20�� �ػ���
    avg_pooling_op f19_ap;

    //��21�� ȫ���Ӳ�
    fc_op f20_fc;


}mobilenet;



/*-------------------------������ǰ�򴫲�ʵ��------------------*/
//�����ļ����
//relu����
static void relu_op_forward(nonlinear_op* op);

static void relu6_op_forward(nonlinear_op* op);

static void sigmoid_op_forward(nonlinear_op* op);

//������
static void conv_op_forward(conv_op* op);

//mobilenet�е�dw���
static void conv_dw_op_forward(conv_op* op);

//mobilenet�е�bn��һ��
static void batch_norm_op_forward(batch_norm_op* op);

//mobilenet�е�res_connect
static void res_connect_op_forward(res_connect_op* op);

//mobilenet�е�bottleneck��
static void bottleneck_op_forward(bottleneck_op* op);

//���ػ���������
static void max_pooling_op_forward(max_pooling_op* op);

//ƽ���ػ�
static void avg_pooling_op_forward(avg_pooling_op* op);

//ȫ����
static void fc_op_forward(fc_op* op);


/*-------------------------������ṹ��ʼ��---------------------*/
//��ʼ�������������Ҫ�Ĳ�������������/���ͨ����������������С������˴�С�ȵ�

//�������������ʼ��
static void init_conv_op(conv_op* op, int in_channels, int out_channels, int stride, int padding, int kernel_size, int input_shape, bool is_dw);

//bn����������ʼ��
static void init_bn_op(batch_norm_op* op, int channels, int shape);

//ƿ�������������ʼ�� tΪ��ά����
static void init_bottleneck_op(bottleneck_op* op, int in_channels, int out_channels, int stride, int input_shape, int t);

//�ػ�����������ʼ��
static void init_avg_pool_op(avg_pooling_op* op, int channels, int stride, int kernel_size, int input_shape);

//����mobilenet����ṹ����
void setup_mobilenet(mobilenet* net);



/*-------------------------������ģ�Ͳ�����ʼ��---------------------*/

//����ģ�Ͳ����ռ�
static void calloc_conv_weights(conv_op* op);
static void calloc_bn_weights(batch_norm_op* op);
static void calloc_fc_weights(fc_op* op);
static void calloc_bottleneck_weights(bottleneck_op* op);
void malloc_mobilenet(mobilenet* net);


//���ļ��м����Ѿ�ѵ���õĸþ����Ķ�Ӧ����
static void load_conv_weights(conv_op* op, FILE* fp);
static void load_conv_weights(conv_op* op, string filename);
static void load_conv_weights(conv_op* op, ifstream& file);

//���ļ��м����Ѿ�ѵ���õ�bn��Ķ�Ӧ����
static int load_bn_weights(batch_norm_op* op, string filename[], int index);
static void load_bn_weights(batch_norm_op* op, ifstream& file);

//���ļ��м����Ѿ�ѵ���õĸ�ȫ���Ӳ�Ķ�Ӧ����
static void load_fc_weights(fc_op* op, FILE* fp);
static void load_fc_weights(fc_op* op, string wf, string bf);
static void load_fc_weights(fc_op* op, ifstream& file);

static int load_bottleneck_weights(bottleneck_op* op, string filename[], int index);
static void load_bottleneck_weights(bottleneck_op* op, ifstream& file);

// �����Ѿ�ѵ���õ�ģ�Ͳ���
void load_mobilenet(mobilenet* net, string filename);



//�ͷ�ģ�Ϳռ�
static void free_conv_weights(conv_op* op);
static void free_bn_weights(batch_norm_op* op);
static void free_bottleneck_weights(bottleneck_op* op);
static void free_fc_weights(fc_op* op);
void free_mobilenet(mobilenet* net);


/*-------------------------ǰ�򴫲�ʵ��---------------------*/
//Ϊ�м�������ռ�
static void malloc_bottleneck_pass(bottleneck_op* op);
void malloc_mobilenet_pass(mobilenet* net);


void forward_mobilenet(mobilenet* net);

//��ʼ���м���
static void init_bottleneck_pass(bottleneck_op* op);
void init_mobilenet_layer(mobilenet* net);

//�ͷ��м����ռ�
static void free_bottleneck_layer(bottleneck_op* op);
void free_mobilenet_layer(mobilenet* net);


void start_time_record(string filename);
void end_time_record();
#endif