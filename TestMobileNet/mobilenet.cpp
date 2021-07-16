/**************************************************/
/**本程序可实现的功能：基于imagenet数据集的MobileetV2单样本推断任务**/
/*Author：tc */
/*Owner：Teacher Zhao's Architecture Team */
/*************************************************/
/*
2021.06.09 基本完成mobilenet-v2推断功能，测试运行正常
2021 06.11 修改output内存申请方式
2021.06.14 加入2000张图片的数据集用于推断
*/
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include"opt.h"
#include "mobilenet.h"
#define im2col true;
fstream time_record;
int record = 0;
void start_time_record(string fliename)
{
    time_record.open(fliename, ios::out);
    time_record << "op,time"<<endl;
}

void end_time_record()
{
    time_record.close();
}
/*-------------------------各操作前向传播实现--------------------*/
//常见的激活函数
//relu函数
static void relu_op_forward(nonlinear_op* op)
{
    for (int i = 0; i < (op->units) * (op->batchsize); i++)
    {
        if (op->input[i] <= 0.0)
        {
            op->output[i] = 0.0;
        }
        else
        {
            op->output[i] = op->input[i];
        }
    }
}
//relu6函数
static void relu6_op_forward(nonlinear_op* op)
{
    clock_t start, end;
    start = clock();
    //去掉batchsize
    for (int i = 0; i < op->units; i++)
    {
        if (op->input[i] <= 0.0)
        {
            op->output[i] = 0.0;
        }
        else if (op->input[i] >= 6.0)
        {
            op->output[i] = 6.0;
        }
        else
        {
            op->output[i] = op->input[i];
        }
    }
    end = clock();
    time_record << "relu6," << end-start << endl;
}
//sigmoid函数
static void sigmoid_op_forward(nonlinear_op* op)
{
    for (int i = 0; i < (op->units) * (op->batchsize); i++)
        op->output[i] = 1.0f/ (1.0f + exp(0.0f - op->input[i]));
}

//常规卷积操作函数
static void conv_op_forward(conv_op* op)
{
    clock_t start, end;
    start = clock();
    int Isim2col = im2col;
    if (Isim2col)
    {
        test_im2col(op);
    }
    else
    {
        op->batchsize = 1;
        float* input_p;
        int s = op->stride;//步长
        int p = op->padding;//填充大小
        int iw = op->in_w;
        int ih = op->in_h;
        int iwih = iw * ih;
        int iw1 = op->in_w + 2 * p;
        int ih1 = op->in_h + 2 * p;
        int iwih1 = iw1 * ih1;
        int owoh = op->out_w * op->out_h;
        int k = op->kernel_size;
        int kk = op->kernel_size * op->kernel_size;
        int ikk = op->in_channels * kk;
        int i_iwih = op->in_channels * iwih;
        int i_iwih1 = op->in_channels * iwih1;
        input_p = (float*)calloc(i_iwih1, sizeof(float));//分配填充后的权重空间，初始化为0
        //将输入特征图进行填充操作

        for (int i = 0; i < op->batchsize; i++)
        {
            for (int j = 0; j < op->in_channels; j++)
            {
                for (int ii = 0; ii < op->in_h; ii++)
                {
                    for (int jj = 0; jj < op->in_w; jj++)
                    {
                        //printf("%d\n", i * iwih + j * iwih + ii * ih + jj);
                        input_p[i * i_iwih1 + j * iwih1 + (ii + p) * iw1 + jj + p] = op->input[i * iwih + j * iwih + ii * ih + jj];
                    }
                }
            }
        }
        //卷积操作具体实现
        for (int o_c = 0; o_c < op->out_channels; o_c++)
        {
            for (int o_h = 0; o_h < op->out_h; o_h++)
            {
                for (int o_w = 0; o_w < op->out_w; o_w++)
                {
                    for (int i_c = 0; i_c < op->in_channels; i_c++)
                        for (int i = 0; i < op->kernel_size; i++)
                        {
                            for (int j = 0; j < op->kernel_size; j++)
                            {
                                op->output[o_c * owoh + o_h * op->out_w + o_w] += op->weights[o_c * ikk + i_c * kk + i * k + j] * input_p[i_c * iwih1 + (s * o_h + i) * iw1 + s * o_w + j];
                            }
                        }
                }
            }
        }
        free(input_p);
    }

    end = clock();
    time_record << "conv," << end-start << endl;
    //加偏置操作
    //int o_offset = 0;
    //for (int i = 0; i < op->out_channels; i++)
    //{
    //    float tmp = op->bias[i];
    //    while (o_offset < (i + 1) * owoh)
    //    {
    //        op->output[o_offset] += tmp;
    //        o_offset++;
    //    }
    //}


}

//mobilenet中的dw卷积
static void conv_dw_op_forward(conv_op* op)
{
    //op->batchsize = 1;
    float* input_p;
    int S = op->stride;//步长
    int P = op->padding;//填充大小
    int iw = op->in_w;
    int ih = op->in_h;
    int iwih = iw * ih;
    int iw1 = op->in_w + 2 * P;
    int ih1 = op->in_h + 2 * P;
    int iwih1 = iw1 * ih1;
    int owoh = op->out_w * op->out_h;
    int k = op->kernel_size;
    int kk = op->kernel_size * op->kernel_size;
    int ikk = op->in_channels * kk;
    int i_iwih = op->in_channels * iwih;
    int i_iwih1 = op->in_channels * iwih1;
    input_p = (float*)calloc(i_iwih1, sizeof(float));//分配填充后的权重空间，初始化为0
    clock_t start, end;
    start = clock();
    //将输入特征图进行填充操作
    for (int i = 0; i < op->batchsize; i++)
    {
        for (int j = 0; j < op->in_channels; j++)
        {
            for (int ii = 0; ii < op->in_h; ii++)
            {
                for (int jj = 0; jj < op->in_w; jj++)
                {
                    //printf("%d\n", i * iwih + j * iwih + ii * ih + jj);
                    input_p[i * i_iwih1 + j * iwih1 + (ii + P) * iw1 + jj + P] = op->input[i * iwih + j * iwih + ii * ih + jj];
                }
            }
        }
    }
    //dw卷积操作具体实现
    for (int c = 0; c < op->out_channels; c++)
    {
        for (int o_h = 0; o_h < op->out_h; o_h++)
        {
            for (int o_w = 0; o_w < op->out_w; o_w++)
            {
                    for (int i = 0; i < op->kernel_size; i++)
                    {
                        for (int j = 0; j < op->kernel_size; j++)
                        {
                            op->output[c * owoh + o_h * op->out_w + o_w] += op->weights[c * kk + i * k + j] * input_p[c * iwih1 + (S * o_h + i)* iw1 + S * o_w + j];
                            //output[通道数*map大小 + 当前宽度*每列大小 +当前高度]等价于ouput[c*owoh+o_h*op->out_h+o_w]
                            //weights[当前通道数*卷积核大小 + 当前卷积核座标1*卷积核宽度 + 座标2]
                            //input_p[当前通道数*padding后输入map大小+(步长*当前高度+i）*扩张后高度）+（步长*当前宽度）
                        }
                    }
            }
        }
    }
    end = clock();
    time_record << "dw_conv," << end-start << endl;
    free(input_p);

}

//mobilenet中的bn归一化
static void batch_norm_op_forward(batch_norm_op* op)
{
    int w = op->w, h = op->w;
    int wh = w * h;
    int c = op->channels;
    //每个通道对应一个bn操作
    clock_t start, end;
    start = clock();
    for (int i = 0; i < c; i++)
    {
        //bn归一化
        for (int j = 0; j < wh; j++)
        {
            op->output[i * wh + j] = op->weight[i] * (op->input[i * wh + j] - op->mean[i]) / (sqrt(op->var[i] + op->eps)) + op->bias[i];
            //bn归一化公式gamma*(x-mean)/(sqrt(var+eps))+beta
        }

    }
    end = clock();
    time_record << "bn," << end-start << endl;
}

//mobilenet中的res_connect
static void res_connect_op_forward(res_connect_op* op)
{
    clock_t start, end;
    start = clock();
    int units = op->units;
    for (int i = 0; i < units; i++)
    {
        op->output[i] = op->input[i] + op->add_input[i];
    }
    end = clock();
    time_record << "res," << end-start << endl;
}

static void bottleneck_op_forward(bottleneck_op* op)
{
    //pw
    if (op->t > 1)
    {
        op->conv[0].input = op->input;
        //op->conv[0].output = (float*)calloc(op->conv[0].out_units, sizeof(float));
        conv_op_forward(&(op->conv[0]));
        op->bn[0].input = op->conv[0].output;
        //op->bn[0].output = (float*)malloc(op->bn[0].units * sizeof(float));
        batch_norm_op_forward(&(op->bn[0]));
        //op->relu6[0].output = (float*)malloc(op->relu6[0].units * sizeof(float));
        op->relu6[0].input = op->bn[0].output;
        relu6_op_forward(&(op->relu6[0]));
        op->conv[1].input = op->relu6[0].output;
    
    }
    else op->conv[1].input = op->input;
    //dw
    //op->conv[1].output = (float*)calloc(op->conv[1].out_units, sizeof(float));
    conv_dw_op_forward(&(op->conv[1]));
    op->bn[1].input = op->conv[1].output;
    //op->bn[1].output = (float*)malloc(op->bn[1].units * sizeof(float));
    batch_norm_op_forward(&(op->bn[1]));
    //op->relu6[1].output = (float*)malloc(op->relu6[1].units * sizeof(float));
    op->relu6[1].input = op->bn[1].output;
    relu6_op_forward(&(op->relu6[1]));

    //pw
    op->conv[2].input = op->relu6[1].output;
    //op->conv[2].output = (float*)calloc(op->conv[2].out_units, sizeof(float));
    conv_op_forward(&(op->conv[2]));
    op->bn[2].input = op->conv[2].output;
    //op->bn[2].output = (float*)malloc(op->bn[2].units * sizeof(float));
    batch_norm_op_forward(&(op->bn[2]));

    //残差连接
    if (op->stride == 1 && op->in_units == op->out_units)
    {
        op->res_con.add_input=op->input;
        op->res_con.input = op->bn[2].output;
       // op->res_con.output = (float*)malloc(op->res_con.units * sizeof(float));
        res_connect_op_forward(&(op->res_con));
        op->output = op->res_con.output;
    }

    else op->output = op->bn[2].output;

}
//最大池化
static void max_pooling_op_forward(max_pooling_op* op)
{
    int channels = op->channels;
    int strides = op->stride;
    int pool_size = op->kernel_size;

    int input_offset;
    int output_offset;
    int iwih = op->in_w * op->in_h;
    int owoh = op->out_w * op->out_h;
    clock_t start, end;
    start = clock();
    for (int c = 0; c < channels; c++)
    {
        for (int i = 0; i < op->out_h; i++)
        {
            for (int j = 0; j < op->out_w; j++)
            {
                input_offset = j * strides + i * strides * (op->in_w) + c * iwih;
                float pixel = op->input[input_offset];
                for (int k = 0; k < pool_size; k++)
                {
                    for (int l = 0; l < pool_size; l++)
                    {
                        pixel = MAX(pixel, op->input[c * iwih + (i * strides + k) * op->in_w + j * strides + l]);

                    }
                }
                output_offset = j + i * (op->out_w) + c * owoh;
                op->output[output_offset] = pixel;
            }
        }
    }
}
//平均池化
static void avg_pooling_op_forward(avg_pooling_op* op)
{
    int channels = op->channels;
    int strides = op->stride;
    int pool_size = op->kernel_size;

    int input_offset;
    int output_offset;
    int iwih = op->in_w * op->in_h;
    int owoh = op->out_w * op->out_h;
    clock_t start, end;
    start = clock();
    for (int c = 0; c < channels; c++)
    {
        for (int i = 0; i < op->out_h; i++)
        {
            for (int j = 0; j < op->out_w; j++)
            {
                input_offset = j * strides + i * strides * (op->in_w) + c * iwih;
                //float pixel = op->input[input_offset];
                float sum = 0, mean = 0;
                for (int k = 0; k < pool_size; k++)
                {
                    for (int l = 0; l < pool_size; l++)
                    {
                       // pixel = MAX(pixel, op->input[c * iwih + (i * strides + k) * op->in_h + j * strides + l]);
                        sum += op->input[c * iwih + (i * strides + k) * op->in_w + j * strides + l];

                    }
                }
                mean = sum / (1.0f * pool_size * pool_size);
                output_offset = j + i * (op->out_w) + c * owoh;
                op->output[output_offset] = mean;
            }
        }
    }
    end = clock();
    time_record << "avg_pool," << end-start << endl;
}

//全连接操作函数
static void fc_op_forward(fc_op* op)
{
    clock_t start, end;
    start = clock();
    //op->batchsize = 1;
    for (int b = 0; b < op->batchsize; b++)
    {
        for (int i = 0; i < op->out_units; i++)
        {
            for (int j = 0; j < op->in_units; j++)
            {
                op->output[b * op->out_units + i] += op->input[b * op->in_units + j] * op->weights[i * op->in_units + j];
            }
        }
    }

    for (int j = 0; j < op->batchsize; j++)
    {
        int o_offset = j * op->out_units;

        for (int i = 0; i < op->out_units; i++, o_offset++)
            op->output[o_offset] += op->bias[i];
    }
    end = clock();
    time_record << "fc," << end-start << endl;
}



/*-------------------------神经网络结构初始化---------------------*/

//卷积操作参数初始化
static void init_conv_op(conv_op* op,int in_channels,int out_channels, int stride, int padding, int kernel_size, int input_shape,bool is_dw)
{
    op->batchsize = 1;
    op->in_channels = in_channels;
    op->out_channels = out_channels;
    op->stride = stride;
    op->padding = padding;
    op->kernel_size = kernel_size;
    op->in_w = input_shape;
    op->in_h = input_shape;
    int output_shape = (input_shape - kernel_size + 2 * padding) / stride + 1;
    op->out_w = output_shape;
    op->out_h = output_shape;
    op->in_units = input_shape * input_shape * in_channels;
    op->out_units = output_shape * output_shape * out_channels;
    if (!is_dw) op->filter = kernel_size * kernel_size * in_channels * out_channels;
    else op->filter = kernel_size * kernel_size * out_channels;
}

//bn操作参数初始化
static void init_bn_op(batch_norm_op* op, int channels, int shape)
{
    op->batchsize = 1;
    op->w = shape;
    op->h = shape;
    op->channels = channels;
    op->units = shape * shape * channels;

}

//瓶颈层参数初始化 t为升维倍数
static void init_bottleneck_op(bottleneck_op* op, int in_channels, int out_channels, int stride, int input_shape, int t) 
{
    op->in_channels = in_channels;
    op->out_channels = out_channels;
    op->in_w = input_shape;
    op->in_h = input_shape;
    op->stride = stride;
    op->t = t;
    //pw卷积
    if (t > 1)
    {
        init_conv_op(&(op->conv[0]), in_channels, in_channels * t, PW_STRIDES, PW_PADDING, PW_KERNEL_L, input_shape,false);
        init_bn_op(&(op->bn[0]), op->conv[0].out_channels, op->conv[0].out_w);
        op->relu6[0].units = op->bn[0].units;
        init_conv_op(&(op->conv[1]), in_channels * t, in_channels * t, stride, DW_PADDING, DW_KERNEL_L, op->conv[0].out_w,true);
    }
    else init_conv_op(&(op->conv[1]), in_channels * t, in_channels * t, stride, DW_PADDING, DW_KERNEL_L, input_shape,true);
    
    //dw卷积
    init_bn_op(&(op->bn[1]), op->conv[1].out_channels, op->conv[1].out_w);
    op->relu6[1].units = op->bn[1].units;

    //pw卷积
    init_conv_op(&(op->conv[2]), in_channels * t, out_channels, PW_STRIDES, PW_PADDING, PW_KERNEL_L, op->conv[1].out_w,false);
    init_bn_op(&(op->bn[2]), op->conv[2].out_channels, op->conv[2].out_w);

    int output_shape = op->conv[2].out_w;
    op->out_w = output_shape;
    op->out_h = output_shape;
    op->in_units = input_shape * input_shape * in_channels;
    op->out_units = output_shape * output_shape * out_channels;
    op->res_con.units = op->in_units;

}

static void init_avg_pool_op(avg_pooling_op* op, int channels, int stride, int kernel_size, int input_shape)
{
    op->batchsize = 1;
    op->channels = channels;
    op->kernel_size = kernel_size;
    op->stride = stride;
    
    op->in_w = input_shape;
    op->in_h = input_shape;
    int output_shape = (input_shape - kernel_size) / stride + 1;
    op->out_w = output_shape;
    op->out_h = output_shape;
    op->in_units = input_shape * input_shape * channels;
    op->out_units = output_shape * output_shape * channels;
}

//设置mobilenet网络结构
void setup_mobilenet(mobilenet* net)
{
    net->batchsize = 1;
    //第1层卷积层参数设置 input 224*224*3 output 112*112*32 filter 3*3*3*32 stride=2 padding=1
    init_conv_op(&(net->f0_conv), IN_CHANNELS, F0_CHANNELS, F0_STRIDES, F0_PADDING, F0_KERNEL_L, IN_L,false);
    init_bn_op(&(net->f0_bn), net->f0_conv.out_channels, net->f0_conv.out_w);
    net->f0_relu6.units = net->f0_bn.units;

    //第2层瓶颈层1参数设置 input 112*112*32 output 112*112*16 strides=1 padding=1
    //省略一个pw的bottleneck
    init_bottleneck_op(&(net->f1), F0_CHANNELS, F1_CHANNELS, F1_STRIDES, net->f0_conv.out_w, F1_T);

    //第3层瓶颈层2_1
    init_bottleneck_op(&(net->f2), F1_CHANNELS, F2_CHANNELS, F2_STRIDES, net->f1.out_w,F2_T);

    //第4层瓶颈层2_2
    init_bottleneck_op(&(net->f3), F2_CHANNELS, F3_CHANNELS, F3_STRIDES, net->f2.out_w, F3_T);

    //第5层 瓶颈层3_1
    init_bottleneck_op(&(net->f4), F3_CHANNELS, F4_CHANNELS, F4_STRIDES, net->f3.out_w, F4_T);
    
    //第6层 瓶颈层3_2
    init_bottleneck_op(&(net->f5), F4_CHANNELS, F5_CHANNELS, F5_STRIDES, net->f4.out_w, F5_T);
    
    //第7层 瓶颈层3_3
    init_bottleneck_op(&(net->f6), F5_CHANNELS, F6_CHANNELS, F6_STRIDES, net->f5.out_w, F6_T);
    
    //第8层 瓶颈层4_1
    init_bottleneck_op(&(net->f7), F6_CHANNELS, F7_CHANNELS, F7_STRIDES, net->f6.out_w, F7_T);
    
    //第9层 瓶颈层4_2
    init_bottleneck_op(&(net->f8), F7_CHANNELS, F8_CHANNELS, F8_STRIDES, net->f7.out_w, F8_T);

    //第10层 瓶颈层4_3
    init_bottleneck_op(&(net->f9), F8_CHANNELS, F9_CHANNELS, F9_STRIDES, net->f8.out_w, F9_T);

    //第11层 瓶颈层4_4
    init_bottleneck_op(&(net->f10), F9_CHANNELS, F10_CHANNELS, F10_STRIDES, net->f9.out_w, F10_T);
    
    //第12层 瓶颈层5_1
    init_bottleneck_op(&(net->f11), F10_CHANNELS, F11_CHANNELS, F11_STRIDES, net->f10.out_w, F11_T);

    //第13层 瓶颈层5_2
    init_bottleneck_op(&(net->f12), F11_CHANNELS, F12_CHANNELS, F12_STRIDES, net->f11.out_w, F12_T);

    //第14层 瓶颈层5_3
    init_bottleneck_op(&(net->f13), F12_CHANNELS, F13_CHANNELS, F13_STRIDES, net->f12.out_w, F13_T);

    //第15层 瓶颈层6_1
    init_bottleneck_op(&(net->f14), F13_CHANNELS, F14_CHANNELS, F14_STRIDES, net->f13.out_w, F14_T);

    //第16层 瓶颈层6_2
    init_bottleneck_op(&(net->f15), F14_CHANNELS, F15_CHANNELS, F15_STRIDES, net->f14.out_w, F15_T);

    //第17层 瓶颈层6_3
    init_bottleneck_op(&(net->f16), F15_CHANNELS, F16_CHANNELS, F16_STRIDES, net->f15.out_w, F16_T);

    //第18层 瓶颈层7_1
    init_bottleneck_op(&(net->f17), F16_CHANNELS, F17_CHANNELS, F17_STRIDES, net->f16.out_w, F17_T);

    //第19层 卷积层
    init_conv_op(&(net->f18_conv), F17_CHANNELS, F18_CHANNELS, F18_STRIDES, F18_PADDING, F18_KERNEL_L, net->f17.out_w,false);
    init_bn_op(&(net->f18_bn), net->f18_conv.out_channels, net->f18_conv.out_w);
    net->f18_relu6.units = net->f18_bn.units;

    //第20层 池化层
    init_avg_pool_op(&(net->f19_ap), F19_CHANNELS, F19_KERNEL_L, F19_KERNEL_L, net->f18_conv.out_w); //in_shape=stride=pool_size=7 全局池化

    //第21层 全连接层
    net->f20_fc.batchsize = 1;
    net->f20_fc.in_units = net->f19_ap.out_units;
    net->f20_fc.out_units = F20_CHANNELS;


}



/*-------------------------神经网络模型参数初始化---------------------*/
//分配模型空间
static void calloc_conv_weights(conv_op* op)
{
    op->weights = (float*)calloc(op->filter, sizeof(float));
    //op->bias = (float*)calloc(op->out_channels, sizeof(float));
}
static void calloc_bn_weights(batch_norm_op* op)
{
    op->weight = (float*)malloc(op->channels * sizeof(float));
    op->bias = (float*)malloc(op->channels * sizeof(float));
    op->mean = (float*)malloc(op->channels * sizeof(float));
    op->var = (float*)malloc(op->channels * sizeof(float));

}
static void calloc_fc_weights(fc_op* op)
{
    int f = op->in_units * op->out_units;
    op->weights = (float*)calloc(f, sizeof(float));
    op->bias = (float*)calloc(op->out_units, sizeof(float));
}
static void calloc_bottleneck_weights(bottleneck_op* op)
{
    if (op->t > 1)
    {
        calloc_conv_weights(&(op->conv[0]));
        calloc_bn_weights(&(op->bn[0]));
    }
    calloc_conv_weights(&(op->conv[1]));
    calloc_conv_weights(&(op->conv[2]));
    calloc_bn_weights(&(op->bn[1]));
    calloc_bn_weights(&(op->bn[2]));

}
void malloc_mobilenet(mobilenet* net)
{
    calloc_conv_weights(&(net->f0_conv));
    calloc_bn_weights(&(net->f0_bn));
    calloc_bottleneck_weights(&(net->f1));
    calloc_bottleneck_weights(&(net->f2));
    calloc_bottleneck_weights(&(net->f3));
    calloc_bottleneck_weights(&(net->f4));
    calloc_bottleneck_weights(&(net->f5));
    calloc_bottleneck_weights(&(net->f6));
    calloc_bottleneck_weights(&(net->f7));
    calloc_bottleneck_weights(&(net->f8));
    calloc_bottleneck_weights(&(net->f9));
    calloc_bottleneck_weights(&(net->f10));
    calloc_bottleneck_weights(&(net->f11));
    calloc_bottleneck_weights(&(net->f12));
    calloc_bottleneck_weights(&(net->f13));
    calloc_bottleneck_weights(&(net->f14));
    calloc_bottleneck_weights(&(net->f15));
    calloc_bottleneck_weights(&(net->f16));
    calloc_bottleneck_weights(&(net->f17));
    calloc_conv_weights(&(net->f18_conv));
    calloc_bn_weights(&(net->f18_bn));
    calloc_fc_weights(&(net->f20_fc));


}

//从文件中加载已经训练好的该卷积层的对应参数
static void load_conv_weights(conv_op* op, FILE* fp)
{
    int f = op->out_channels * op->in_channels * op->kernel_size * op->kernel_size;
    fread(op->weights, sizeof(float), f, fp);
   // fread(op->bias, sizeof(float), op->out_channels, fp);
}

static void load_conv_weights(conv_op* op, string filename)
{
    int f = op->filter;
    ifstream in(filename);
    for (int i = 0; i < f; i++)
    {
        in >> op->weights[i];
    }
    in.close();
}

static void load_conv_weights(conv_op* op, ifstream& file)
{
    file.read((char*)op->weights,sizeof(float)*op->filter);
}


static int load_bn_weights(batch_norm_op* op, string filename[],int index)
{
    int i = index;
    ifstream weight_in(filename[i++]),bias_in(filename[i++]),mean_in(filename[i++]), var_in(filename[i++]);
    for (int c = 0; c < op->channels; c++)
    {
        weight_in >> op->weight[c];
        bias_in >> op->bias[c];
        mean_in >> op->mean[c];
        var_in >> op->var[c];
    }
    weight_in.close();
    bias_in.close();
    mean_in.close();
    var_in.close();
    i++;
    return i;
}

static void load_bn_weights(batch_norm_op* op, ifstream& file)
{
    file.read((char*)op->weight, sizeof(float) * op->channels);
    file.read((char*)op->bias, sizeof(float) * op->channels);
    file.read((char*)op->mean, sizeof(float) * op->channels);
    file.read((char*)op->var, sizeof(float) * op->channels);
}

static int load_bottleneck_weights(bottleneck_op* op, string filename[],int index)
{
    int i = index;
    if (op->t > 1)
    {
        load_conv_weights(&(op->conv[0]), filename[i++]);
        i = load_bn_weights(&(op->bn[0]), filename, i);
    }
    load_conv_weights(&(op->conv[1]), filename[i++]);
    i=load_bn_weights(&(op->bn[1]), filename, i);
    load_conv_weights(&(op->conv[2]), filename[i++]);
    i=load_bn_weights(&(op->bn[2]), filename, i);
    return i;
}

static void load_bottleneck_weights(bottleneck_op* op, ifstream& file)
{
    if (op->t > 1)
    {
        load_conv_weights(&(op->conv[0]), file);
        load_bn_weights(&(op->bn[0]), file);
    }
    load_conv_weights(&(op->conv[1]), file);
    load_bn_weights(&(op->bn[1]), file);
    load_conv_weights(&(op->conv[2]), file);
    load_bn_weights(&(op->bn[2]), file);
}

//从文件中加载已经训练好的该全连接层的对应参数
static void load_fc_weights(fc_op* op, FILE* fp)
{
    int f = op->in_units * op->out_units;
    fread(op->weights, sizeof(float), f, fp);
    fread(op->bias, sizeof(float), op->out_units, fp);
}

static void load_fc_weights(fc_op* op, string bf ,string wf)
{
    int f = op->in_units * op->out_units;
    ifstream w_in(wf);
    for (int i = 0; i < f; i++)
    {
        w_in >> op->weights[i];
    }
    w_in.close();
    ifstream b_in(bf);
    for (int i = 0; i < op->out_units; i++)
    {
        float temp;
        b_in >> temp;
        op->bias[i] = temp;
    }
    b_in.close();
}

void load_fc_weights(fc_op* op,  ifstream& file)
{
    file.read((char*)op->weights, sizeof(float) * op->in_units * op->out_units);
    file.read((char*)op->bias, sizeof(float) * op->out_units);
}

// 加载已经训练好的alexnet模型参数
void load_mobilenet(mobilenet* net, string filename)
{
    /*
    * 从txt文件中读取的版本
    string weights_filename[315];
    ifstream fname_in(filename);
    for (int i = 0; i < 315; i++)
    {
        fname_in >> weights_filename[i];
        weights_filename[i] = "./MobileNetV2_pretrained2/" + weights_filename[i] + ".txt";
    }
    fname_in.close();
    int i = 0;
    load_conv_weights(&(net->f0_conv), weights_filename[i++]);
    i = load_bn_weights(&(net->f0_bn), weights_filename, i);
    i = load_bottleneck_weights(&(net->f1), weights_filename, i);
    i = load_bottleneck_weights(&(net->f2), weights_filename, i);
    i = load_bottleneck_weights(&(net->f3), weights_filename, i);
    i = load_bottleneck_weights(&(net->f4), weights_filename, i);
    i = load_bottleneck_weights(&(net->f5), weights_filename, i);
    i = load_bottleneck_weights(&(net->f6), weights_filename, i);
    i = load_bottleneck_weights(&(net->f7), weights_filename, i);
    i = load_bottleneck_weights(&(net->f8), weights_filename, i);
    i = load_bottleneck_weights(&(net->f9), weights_filename, i);
    i = load_bottleneck_weights(&(net->f10), weights_filename, i);
    i = load_bottleneck_weights(&(net->f11), weights_filename, i);
    i = load_bottleneck_weights(&(net->f12), weights_filename, i);
    i = load_bottleneck_weights(&(net->f13), weights_filename, i);
    i = load_bottleneck_weights(&(net->f14), weights_filename, i);
    i = load_bottleneck_weights(&(net->f15), weights_filename, i);
    i = load_bottleneck_weights(&(net->f16), weights_filename, i);
    i = load_bottleneck_weights(&(net->f17), weights_filename, i);
    load_conv_weights(&(net->f18_conv), weights_filename[i++]);
    i = load_bn_weights(&(net->f18_bn), weights_filename, i);
    load_fc_weights(&(net->f20_fc), weights_filename[i++], weights_filename[i++]);
    */
    ifstream file(filename,ios::binary);
    load_conv_weights(&(net->f0_conv), file);
    load_bn_weights(&(net->f0_bn), file);
    load_bottleneck_weights(&(net->f1), file);
    load_bottleneck_weights(&(net->f2), file);
    load_bottleneck_weights(&(net->f3), file);
    load_bottleneck_weights(&(net->f4), file);
    load_bottleneck_weights(&(net->f5), file);
    load_bottleneck_weights(&(net->f6), file);
    load_bottleneck_weights(&(net->f7), file);
    load_bottleneck_weights(&(net->f8), file);
    load_bottleneck_weights(&(net->f9), file);
    load_bottleneck_weights(&(net->f10), file);
    load_bottleneck_weights(&(net->f11), file);
    load_bottleneck_weights(&(net->f12), file);
    load_bottleneck_weights(&(net->f13), file);
    load_bottleneck_weights(&(net->f14), file);
    load_bottleneck_weights(&(net->f15), file);
    load_bottleneck_weights(&(net->f16), file);
    load_bottleneck_weights(&(net->f17), file);
    load_conv_weights(&(net->f18_conv),file);
    load_bn_weights(&(net->f18_bn), file);
    load_fc_weights(&(net->f20_fc), file);
    file.close();


}

//释放模型空间
static void free_conv_weights(conv_op* op)
{
    free(op->weights);
    free(op->bias);
}

static void free_bn_weights(batch_norm_op* op)
{
    free(op->weight);
    free(op->bias);
    free(op->mean);
    free(op->var);
}

static void free_bottleneck_weights(bottleneck_op* op)
{
    free_conv_weights(&(op->conv[0]));
    free_conv_weights(&(op->conv[1]));
    free_conv_weights(&(op->conv[2]));
    free_bn_weights(&(op->bn[0]));
    free_bn_weights(&(op->bn[1]));
    free_bn_weights(&(op->bn[2]));
}
static void free_fc_weights(fc_op* op)
{
    free(op->weights);
    free(op->bias);
}

void free_mobilenet(mobilenet* net)
{
    free_conv_weights(&(net->f0_conv));
    free_bn_weights(&(net->f0_bn));
    free_bottleneck_weights(&(net->f1));
    free_bottleneck_weights(&(net->f2));
    free_bottleneck_weights(&(net->f3));
    free_bottleneck_weights(&(net->f4));
    free_bottleneck_weights(&(net->f5));
    free_bottleneck_weights(&(net->f6));
    free_bottleneck_weights(&(net->f7));
    free_bottleneck_weights(&(net->f8));
    free_bottleneck_weights(&(net->f9));
    free_bottleneck_weights(&(net->f10));
    free_bottleneck_weights(&(net->f11));
    free_bottleneck_weights(&(net->f12));
    free_bottleneck_weights(&(net->f13));
    free_bottleneck_weights(&(net->f14));
    free_bottleneck_weights(&(net->f15));
    free_bottleneck_weights(&(net->f16));
    free_bottleneck_weights(&(net->f17));
    free_conv_weights(&(net->f18_conv));
    free_bn_weights(&(net->f18_bn));
    free_fc_weights(&(net->f20_fc));
}






/*-------------------------神经网络前向传播实现----------------------*/
//mobilenet前向传播过程
void forward_mobilenet(mobilenet* net)
{
    int t = 0;
    //第1层 卷积层
    net->f0_conv.input = net->input;
   // t = net->batchsize * net->f0_conv.out_units;
   // net->f0_conv.output = (float*)calloc(t, sizeof(float));
    conv_op_forward(&(net->f0_conv));
    net->f0_bn.input = net->f0_conv.output;
   // net->f0_bn.output = (float*)malloc(net->batchsize * sizeof(float) * net->f0_bn.units);
    batch_norm_op_forward(&(net->f0_bn));
   // net->f0_relu6.output = (float*)malloc(net->batchsize * sizeof(float) * net->f0_relu6.units);
    net->f0_relu6.input = net->f0_bn.output;
    relu6_op_forward(&(net->f0_relu6));  

    //第2层 瓶颈层1
    net->f1.input = net->f0_relu6.output;
    //net->f1.output = (float*)calloc(net->f1.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f1));

    //第3层 瓶颈层2_1
    net->f2.input = net->f1.output;
   // net->f2.output = (float*)calloc(net->f2.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f2));

    //第4层瓶颈层2_2
    net->f3.input = net->f2.output;
    //net->f3.output = (float*)calloc(net->f3.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f3));

    //第5层 瓶颈层3_1
    net->f4.input = net->f3.output;
   // net->f4.output = (float*)calloc(net->f4.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f4));

    //第6层 瓶颈层3_2
    net->f5.input = net->f4.output;
    //net->f5.output = (float*)calloc(net->f5.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f5));

    //第7层 瓶颈层3_3
    net->f6.input = net->f5.output;
   // net->f6.output = (float*)calloc(net->f6.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f6));

    //第8层 瓶颈层4_1
    net->f7.input = net->f6.output;
   // net->f7.output = (float*)calloc(net->f7.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f7));

    //第9层 瓶颈层4_2
    net->f8.input = net->f7.output;
    //net->f8.output = (float*)calloc(net->f8.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f8));

    //第10层 瓶颈层4_3
    net->f9.input = net->f8.output;
    //net->f9.output = (float*)calloc(net->f9.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f9));

    //第11层 瓶颈层4_4
    net->f10.input = net->f9.output;
    //net->f10.output = (float*)calloc(net->f10.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f10));

    //第12层 瓶颈层5_1
    net->f11.input = net->f10.output;
    //net->f11.output = (float*)calloc(net->f11.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f11));

    //第13层 瓶颈层5_2
    net->f12.input = net->f11.output;
    //net->f12.output = (float*)calloc(net->f12.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f12));

    //第14层 瓶颈层5_3
    net->f13.input = net->f12.output;
   // net->f13.output = (float*)calloc(net->f13.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f13));

    //第15层 瓶颈层6_1
    net->f14.input = net->f13.output;
   // net->f14.output = (float*)calloc(net->f14.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f14));

    //第16层 瓶颈层6_2
    net->f15.input = net->f14.output;
    //net->f15.output = (float*)calloc(net->f15.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f15));

    //第17层 瓶颈层6_3
    net->f16.input = net->f15.output;
    //net->f16.output = (float*)calloc(net->f16.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f16));

    //第18层 瓶颈层7_1
    net->f17.input = net->f16.output;
    //net->f17.output = (float*)calloc(net->f17.out_units, sizeof(float));
    bottleneck_op_forward(&(net->f17));

    //第19层 卷积层
    net->f18_conv.input = net->f17.output;
  //  t = net->batchsize * net->f18_conv.out_units;
   // net->f18_conv.output = (float*)calloc(t, sizeof(float));
    conv_op_forward(&(net->f18_conv));
    net->f18_bn.input = net->f18_conv.output;
   // net->f18_bn.output = (float*)malloc(net->batchsize * sizeof(float) * net->f18_bn.units);
    batch_norm_op_forward(&(net->f18_bn));
    //net->f18_relu6.output = (float*)malloc(net->batchsize * sizeof(float) * net->f18_relu6.units);
    net->f18_relu6.input = net->f18_bn.output;
    relu6_op_forward(&(net->f18_relu6));

    //第20层 池化层
    net->f19_ap.input = net->f18_relu6.output;
   // net->f19_ap.output = (float*)malloc(net->f19_ap.out_units * sizeof(float));
    avg_pooling_op_forward(&(net->f19_ap));

    //第21层 全连接层
    t = net->batchsize * net->f20_fc.out_units;
    //net->f20_fc.output = (float*)calloc(t, sizeof(float));
    net->f20_fc.input = net->f19_ap.output;
    fc_op_forward(&(net->f20_fc));
    net->output = net->f20_fc.output;

    


}

//为中间结果分配空间
static void malloc_bottleneck_pass(bottleneck_op* op)
{
    if (op->t > 1)
    {
        op->conv[0].output = (float*)calloc(op->conv[0].out_units, sizeof(float));
        op->bn[0].output = (float*)calloc(op->bn[0].units, sizeof(float));
        op->relu6[0].output = (float*)calloc(op->relu6[0].units, sizeof(float));
    }
    op->conv[1].output = (float*)calloc(op->conv[1].out_units, sizeof(float));
    op->bn[1].output = (float*)calloc(op->bn[1].units, sizeof(float));
    op->relu6[1].output = (float*)calloc(op->relu6[1].units, sizeof(float));
    op->conv[2].output = (float*)calloc(op->conv[2].out_units, sizeof(float));
    op->bn[2].output = (float*)calloc(op->bn[2].units, sizeof(float));

    if (op->stride == 1 && op->in_units == op->out_units)
    {
        op->res_con.output = (float*)calloc(op->res_con.units, sizeof(float));
    }
}

void malloc_mobilenet_pass(mobilenet* net)
{
    net->f0_conv.output = (float*)calloc(net->f0_conv.out_units, sizeof(float));
    net->f0_bn.output = (float*)calloc(net->f0_bn.units, sizeof(float));
    net->f0_relu6.output = (float*)calloc(net->f0_relu6.units, sizeof(float));
    malloc_bottleneck_pass(&(net->f1));
    malloc_bottleneck_pass(&(net->f2));
    malloc_bottleneck_pass(&(net->f3));
    malloc_bottleneck_pass(&(net->f4));
    malloc_bottleneck_pass(&(net->f5));
    malloc_bottleneck_pass(&(net->f6));
    malloc_bottleneck_pass(&(net->f7));
    malloc_bottleneck_pass(&(net->f8));
    malloc_bottleneck_pass(&(net->f9));
    malloc_bottleneck_pass(&(net->f10));
    malloc_bottleneck_pass(&(net->f11));
    malloc_bottleneck_pass(&(net->f12));
    malloc_bottleneck_pass(&(net->f13));
    malloc_bottleneck_pass(&(net->f14));
    malloc_bottleneck_pass(&(net->f15));
    malloc_bottleneck_pass(&(net->f16));
    malloc_bottleneck_pass(&(net->f17));
    net->f18_conv.output = (float*)calloc(net->f18_conv.out_units, sizeof(float));
    net->f18_bn.output = (float*)calloc(net->f18_bn.units, sizeof(float));
    net->f18_relu6.output = (float*)calloc(net->f18_relu6.units, sizeof(float));
    net->f19_ap.output = (float*)calloc(net->f19_ap.out_units, sizeof(float));
    net->f20_fc.output = (float*)calloc(net->f20_fc.out_units, sizeof(float));
}

//中间结果置为0，用于第二次推测
static void init_bottleneck_pass(bottleneck_op* op)
{
    if (op->t > 1)
    {
        memset(op->conv[0].output, 0.0f, op->conv[0].out_units * sizeof(float));
       // memset(op->bn[0].output, 0.0f, op->bn[0].units * sizeof(float));
       // memset(op->relu6[0].output, 0.0f, op->relu6[0].units * sizeof(float));
    }
    memset(op->conv[1].output, 0.0f, op->conv[1].out_units * sizeof(float));
   // memset(op->bn[1].output, 0.0f, op->bn[1].units * sizeof(float));
   // memset(op->relu6[1].output, 0.0f, op->relu6[1].units * sizeof(float));
    memset(op->conv[2].output, 0.0f, op->conv[2].out_units * sizeof(float));
   // memset(op->bn[2].output, 0.0f, op->bn[2].units * sizeof(float));
    if (op->stride == 1 && op->in_units == op->out_units)
    {
       // memset(op->res_con.output, 0.0f, op->res_con.units * sizeof(float));
    }
}

void init_mobilenet_layer(mobilenet* net)
{
    memset(net->f0_conv.output, 0.0f, net->f0_conv.out_units * sizeof(float));
    //memset(net->f0_bn.output, 0.0f, net->f0_bn.units * sizeof(float));
    //memset(net->f0_relu6.output, 0.0f, net->f0_relu6.units * sizeof(float));
    init_bottleneck_pass(&(net->f1));
    init_bottleneck_pass(&(net->f2));
    init_bottleneck_pass(&(net->f3));
    init_bottleneck_pass(&(net->f4));
    init_bottleneck_pass(&(net->f5));
    init_bottleneck_pass(&(net->f6));
    init_bottleneck_pass(&(net->f7));
    init_bottleneck_pass(&(net->f8));
    init_bottleneck_pass(&(net->f9));
    init_bottleneck_pass(&(net->f10));
    init_bottleneck_pass(&(net->f11));
    init_bottleneck_pass(&(net->f12));
    init_bottleneck_pass(&(net->f13));
    init_bottleneck_pass(&(net->f14));
    init_bottleneck_pass(&(net->f15));
    init_bottleneck_pass(&(net->f16));
    init_bottleneck_pass(&(net->f17));
    memset(net->f18_conv.output, 0.0f, net->f18_conv.out_units * sizeof(float));
    //memset(net->f18_bn.output, 0.0f, net->f18_bn.units * sizeof(float));
   // memset(net->f18_relu6.output, 0.0f, net->f18_relu6.units * sizeof(float));

    //memset(net->f19_ap.output, 0.0f, net->f19_ap.out_units * sizeof(float));
    memset(net->f20_fc.output, 0.0f, net->f20_fc.out_units * sizeof(float));

}

static void free_bottleneck_layer(bottleneck_op* op)
{
    free(op->conv[0].output);
    free(op->conv[1].output);
    free(op->conv[2].output);
    free(op->bn[0].output);
    free(op->bn[1].output);
    free(op->bn[2].output);
    free(op->relu6[0].output);
    free(op->relu6[1].output);
    free(op->res_con.output);
}
void free_mobilenet_layer(mobilenet* net)
{
    free(net->f0_conv.output);
    free(net->f0_bn.output);
    free(net->f0_relu6.output);
    free(net->f1.output);
    free(net->f2.output);
    free(net->f3.output);
    free(net->f4.output);
    free(net->f5.output);
    free(net->f6.output);
    free(net->f7.output);
    free(net->f8.output);
    free(net->f9.output);
    free(net->f10.output);
    free(net->f11.output);
    free(net->f12.output);
    free(net->f13.output);
    free(net->f14.output);
    free(net->f15.output);
    free(net->f16.output);
    free(net->f17.output);
    free(net->f18_conv.output);
    free(net->f18_bn.output);
    free(net->f18_relu6.output);
    free(net->f19_ap.output);
    free(net->f20_fc.output);

}




//int main(int argc, char* argv[])
//{
//    static alexnet net;
//    /*************test && printf **************************/
//    int epochs = 100;//可以根据需要进行修改，数据集上限值为10000
//    char image[100] = "./test_image/1.bmp";//测试集的图片路径
//    char weights_path[100] = "temp.weights";
//    setup_alexnet(&net, 1);//这里默认设置了batchsize=1
//    malloc_alexnet(&net);
//    load_alexnet(&net, weights_path);//加载alexnet已经训练好的权重
//    printf("alexnet setup fininshed. Waiting for inference...\n");
//    alexnet_inference(&net, epochs);//用于测试整个数据集中的图片，图片数量可以自己定义       
//    //alexnet_inference1(&net, image);//用于测试单张图片
//    free_alexnet(&net);
//}