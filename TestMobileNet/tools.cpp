#include"tools.h"
//static long long weights_num=0;
static void write_conv_weights(ofstream &file, conv_op* op)
{
	file.write((char*)op->weights, sizeof(float) * op->filter);
	//cout << "convfilter=" <<op->filter<< endl;
	//weights_num += op->filter;
}

static void write_bn_weights(ofstream &file, batch_norm_op* op)
{
	file.write((char*)op->weight, sizeof(float) * op->channels);
	file.write((char*)op->bias, sizeof(float) * op->channels);
	file.write((char*)op->mean, sizeof(float) * op->channels);
	file.write((char*)op->var, sizeof(float) * op->channels);
	//cout << "bn=" << op->channels * 4<<endl;
	//weights_num +=op->channels * 4;
}

static void write_bottleneck_weights(ofstream &file, bottleneck_op* op)
{
	if (op->t > 1)
	{
		write_conv_weights(file, &(op->conv[0]));
		write_bn_weights(file, &(op->bn[0]));
	}
	write_conv_weights(file, &(op->conv[1]));
	write_bn_weights(file, &(op->bn[1]));
	write_conv_weights(file, &(op->conv[2]));
	write_bn_weights(file, &(op->bn[2]));
}

static void write_fc_weights(ofstream& file, fc_op* op)
{
	file.write((char*)op->weights, sizeof(float) * op->in_units * op->out_units);
	file.write((char*)op->bias, sizeof(float) * op->out_units);
	//weights_num += op->in_units * op->out_units;
	//weights_num += op->out_units;
	//cout << "fc_weights=" << op->in_units * op->out_units << endl;
	//cout << "fc_bias=" << op->out_units << endl;
}

void weights_to_binary_file(ofstream &file, mobilenet* net)
{
	write_conv_weights(file,&(net->f0_conv));
	write_bn_weights(file, &(net->f0_bn));
	write_bottleneck_weights(file, &(net->f1));
	write_bottleneck_weights(file, &(net->f2));
	write_bottleneck_weights(file, &(net->f3));
	write_bottleneck_weights(file, &(net->f4));
	write_bottleneck_weights(file, &(net->f5));
	write_bottleneck_weights(file, &(net->f6));
	write_bottleneck_weights(file, &(net->f7));
	write_bottleneck_weights(file, &(net->f8));
	write_bottleneck_weights(file, &(net->f9));
	write_bottleneck_weights(file, &(net->f10));
	write_bottleneck_weights(file, &(net->f11));
	write_bottleneck_weights(file, &(net->f12));
	write_bottleneck_weights(file, &(net->f13));
	write_bottleneck_weights(file, &(net->f14));
	write_bottleneck_weights(file, &(net->f15));
	write_bottleneck_weights(file, &(net->f16));
	write_bottleneck_weights(file, &(net->f17));
	write_conv_weights(file, &(net->f18_conv));
	write_bn_weights(file, &(net->f18_bn));
	write_fc_weights(file, &(net->f20_fc));
	//cout << "weights_num = " << weights_num<<endl;
}
