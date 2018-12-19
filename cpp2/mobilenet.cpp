#include "config.h"
#include "convlayer.h"
#include "pose_decoder.h"
#include <stdio.h>
#include <cmath>
#include <time.h>
#include <sys/time.h>
#include <time.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

const char* Heatmap2_weightdir = "./wbin/heatmap_2_weights.bin";
const char* Heatmap2_biasdir = "./wbin/heatmap_2_biases.bin";

const char* offset2_weightdir = "./wbin/offset_2_weights.bin";
const char* offset2_biasdir = "./wbin/offset_2_biases.bin";

const char* disp2_fwd_weightdir = "./wbin/displacement_fwd_2_weights.bin";
const char* disp2_fwd_biasdir = "./wbin/displacement_fwd_2_biases.bin";

const char* disp2_bwd_weightdir = "./wbin/displacement_bwd_2_weights.bin";
const char* disp2_bwd_biasdir = "./wbin/displacement_bwd_2_biases.bin";

void sigmoidmul(paratype* dataforsigm, int num) __attribute__ ((noinline));
void sigmoidmul(paratype* dataforsigm, int num)
{
	for (int i = 0; i < num; i++)
	{
		dataforsigm[i] = 1 / (1 + expf(-dataforsigm[i]));
	}
}

int main()
{       
        timeval ts_start;
	timeval ts_curr;
        timeval ts_end;
	float timeuse;
        gettimeofday(&ts_start, NULL);
	paratype* graphin, * graghout,*graphtest,
		*heatmap,*offset2, *displacement_fwd, *displacement_bwd,
		*heatmap_2_weight, *heatmap_2_bias,
		*offset_2_weight, *offset_2_bias, *displacement_fwd_weight, *displacement_fwd_bias, *displacement_bwd_weight, *displacement_bwd_bias;
	
	graphin = new paratype[513 * 513 * 3];
	heatmap = new paratype[Heatmap_outH*Heatmap_outH*Heatmap_Channel];
	offset2 = new paratype[offset_2_outH*offset_2_outH*offset_2_Channel];
	displacement_fwd = new paratype[disp_outH*disp_outH*disp_Channel];
	displacement_bwd = new paratype[disp_outH*disp_outH*disp_Channel];
	heatmap_2_weight = new paratype[Heatmap_outH * Heatmap_outH * Heatmap_Channel];
	heatmap_2_bias = new paratype[Heatmap_Channel];
	offset_2_weight = new paratype[offset_2_outH * offset_2_outH * offset_2_Channel];
	offset_2_bias = new paratype[offset_2_Channel];
	displacement_fwd_weight = new paratype[disp_outH * disp_outH * disp_Channel];
	displacement_fwd_bias = new paratype[disp_Channel];
	displacement_bwd_weight = new paratype[disp_outH * disp_outH * disp_Channel];
	displacement_bwd_bias = new paratype[disp_Channel];
	
	FILE *fr = fopen("./testimage.bin", "rb");
	fread(graphin, sizeof(paratype), 513 * 513 * 3, fr);
	fclose(fr);

	
	int convoutsize = Conv7_poioutH * Conv7_poioutH* Conv13_poiChannel;
	int checksize = offset_2_outH * offset_2_outH*offset_2_Channel;
	
	graphtest = new paratype[checksize];
	fr = fopen("./check/testoffset.bin", "rb");
	fread(graphtest, sizeof(paratype), checksize, fr);
	fclose(fr);
	fr = fopen(Heatmap2_weightdir, "rb");
	fread(heatmap_2_weight, sizeof(paratype), Heatmap_outH * Heatmap_outH * Heatmap_Channel, fr);
	fclose(fr);
	fr = fopen(Heatmap2_biasdir, "rb");
	fread(heatmap_2_bias, sizeof(paratype), Heatmap_Channel, fr);
	fclose(fr);
	fr = fopen(offset2_weightdir, "rb");
	fread(offset_2_weight, sizeof(paratype), offset_2_outH * offset_2_outH * offset_2_Channel, fr);
	fclose(fr);
	fr = fopen(offset2_biasdir, "rb");
	fread(offset_2_bias, sizeof(paratype), offset_2_Channel, fr);
	fclose(fr);
	fr = fopen(disp2_fwd_weightdir, "rb");
	fread(displacement_fwd_weight, sizeof(paratype), disp_outH * disp_outH * disp_Channel, fr);
	fclose(fr);
	fr = fopen(disp2_fwd_weightdir, "rb");
	fread(displacement_fwd_bias, sizeof(paratype), disp_Channel, fr);
	fclose(fr);
	fr = fopen(disp2_bwd_weightdir, "rb");
	fread(displacement_bwd_weight, sizeof(paratype), disp_outH * disp_outH * disp_Channel, fr);
	fclose(fr);
	fr = fopen(disp2_bwd_weightdir, "rb");
	fread(displacement_bwd_weight, sizeof(paratype), disp_Channel, fr);
	fclose(fr);

	graghout = new paratype[convoutsize];

 	gettimeofday(&ts_end, NULL);
        timeuse=  float(ts_end.tv_sec - ts_start.tv_sec) +
                     float(ts_end.tv_usec)*1e-6 - float(ts_start.tv_usec)*1e-6;
        printf("convoutused:%f\n",timeuse);
        gettimeofday(&ts_curr, NULL);

	convmodule(graphin, graghout);
	printf("convcomplet!\n");
	
        gettimeofday(&ts_end, NULL);
        timeuse=  float(ts_end.tv_sec - ts_curr.tv_sec) +
                     float(ts_end.tv_usec)*1e-6 - float(ts_curr.tv_usec)*1e-6;
        printf("convoutused:%f\n",timeuse);
        gettimeofday(&ts_curr, NULL);

	outlayer_norelu(IN graghout, heatmap_2_weight, heatmap_2_bias, heatmap, Conv13_poiChannel, Conv13_poioutH, Heatmap_Kernel, Heatmap_Channel, Heatmap_outH, Heatmap_Stride);
	
        gettimeofday(&ts_end, NULL);
        timeuse=  float(ts_end.tv_sec - ts_curr.tv_sec) +
                     float(ts_end.tv_usec)*1e-6 - float(ts_curr.tv_usec)*1e-6;
        printf("convoutused:%f\n",timeuse);
        gettimeofday(&ts_curr, NULL);

        sigmoidmul(heatmap, Heatmap_outH*Heatmap_outH*Heatmap_Channel);

	outlayer_norelu(IN graghout, offset_2_weight, offset_2_bias, offset2, Conv13_poiChannel, Conv13_poioutH, offset_2_Kernel, offset_2_Channel, offset_2_outH, offset_2_Stride);

	outlayer_norelu(IN graghout, displacement_fwd_weight, displacement_fwd_bias, displacement_fwd, Conv13_poiChannel, Conv13_poioutH, disp_Kernel, disp_Channel, disp_outH, disp_Stride);

	outlayer_norelu(IN graghout, displacement_bwd_weight, displacement_bwd_bias, displacement_bwd, Conv13_poiChannel, Conv13_poioutH, disp_Kernel, disp_Channel, disp_outH, disp_Stride);

        //here we get the four map with format HWC in array

	int checkoffset = (12*33)* 17;
	int numcheck = 17;
	printf("networkoutput\n");
	for (int i = 0; i <numcheck; i++)
	{
		printf("%f,", offset2[checkoffset +i]);

	}
	printf("\nout\n");
	for (int i = 0; i <numcheck; i++)
	{
		printf("%f,", graphtest[checkoffset + i]);

	}
	printf("\n");

	PoseDecoder decoder;
	decoder.heatmap = heatmap;
	decoder.offset = offset2;
	decoder.disFwd = displacement_fwd;
	decoder.disBwd = displacement_bwd;

	cv::Mat3f img(513, 513, reinterpret_cast<cv::Vec3f*>(graphin));
	// cv::Mat1f img(513, 3*513, (float*)graphin);
	// double i_min, i_max;
	// cv::minMaxIdx(img, &i_min, &i_max);
	// printf("min:%f max:%f\n", i_min, i_max);
	cv::Mat3f img2 = (img+1)/2;
	

	decoder.DecodeSinglePose();

	for (int c=0; c<Heatmap_Channel; c++)
	{
		double x = decoder.results[c].x;
		double y = decoder.results[c].y;
		printf("%f %f\n", x, y);
		cv::circle(img2, cv::Point(x, y), 3, cv::Scalar(0,1,0));
	}

	cv::imshow("img", img2);
	cv::waitKey();
	
	for (int c=0; c<Heatmap_Channel; c++)
	{
		decoder.show_heatmap(c);

		cv::waitKey();
	}

	

	
	return 0;
}