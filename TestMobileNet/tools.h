#ifndef TOOLS_H
#define TOOLS_H
#include<iostream>
#include<fstream>
#include"mobilenet.h"
using namespace std;
//参数转为二进制文件存储
static void write_conv_weights(ofstream &file, conv_op* op);
static void write_bn_weights(ofstream &file, batch_norm_op* op);
static void write_bottleneck_weights(ofstream &file, bottleneck_op* op);
static void write_fc_weights(ofstream &file, fc_op* op);
void weights_to_binary_file(ofstream &file,mobilenet* net);



#endif