#include"mobilenet.h"
#include"inference.h"
#include"tools.h"
#include"opt.h"
#include<iostream>
#include<fstream>
#include <cassert>
using namespace std;
#define VNAME(value) (#value)

int main()
{
	static mobilenet net;
	setup_mobilenet(&net);
	malloc_mobilenet(&net);
	cout << "start loading mobilenet weights....." << endl;
	load_mobilenet(&net, "MobileNetV2_pretrained.data");
	cout << "weights loading finished" << endl;
	cout << "Waiting for inference..." << endl;
	//cout << "start writing...." << endl;
	//ofstream out_weights("MobileNetV2_pretrained.data", ios::binary);
	//weights_to_binary_file(out_weights, &net);
	//out_weights.close();
	//cout << "end writing...." << endl;

	//----单张图片推断----//
	//char image[100] = "./test_image/n0761348000000310.jpg";
	//mobilenet_inference1(&net, image);

	//----多张图片推断----//
	clock_t start, end;
	start = clock();
	start_time_record("test_10.csv");
	mobilenet_inference(&net, 10);//测试图片（小于等于2000张)

	end_time_record();
	end = clock();
	cout << "\ntime used: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;

	//----im2col测试----

//	test_im2col(&(net.f0_conv));

	free_mobilenet(&net);
	free_mobilenet_layer(&net);
	//return 0;


	

}