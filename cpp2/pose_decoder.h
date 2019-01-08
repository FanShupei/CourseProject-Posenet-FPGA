#include "config.h"
#include <cassert>
#include <opencv2/core.hpp>

struct pos2_t
{
    int x;
    int y;

    pos2_t(): x(0), y(0) {}

    pos2_t(int _x, int _y):
        x(_x), y(_y) {}
};

struct float2_t
{
    float_t x;
    float_t y;

    float2_t(): x(0), y(0) {}

    float2_t(float_t _x, float_t _y):
        x(_x), y(_y) {}
};

const float_t SCORE_THRESHOLD = 0.05;
const int OUTPUT_STRIDE = 16;

struct PoseDecoder
{
    const float_t* heatmap;
    const float_t* offset;
    const float_t* disFwd;
    const float_t* disBwd;

    bool valid[Heatmap_Channel];
    float2_t results[Heatmap_Channel];

    float_t get_score(int c, int y, int x)
    {
        assert(0<=c && c<=Heatmap_Channel);
        assert(0<=y && y<=Heatmap_outH);
        assert(0<=x && x<=Heatmap_outH);
        return heatmap[(y*Heatmap_outH+x)*Heatmap_Channel+c];
    }

    float2_t get_offset(int c, pos2_t pos)
    {
        assert(0<=c && c<=Heatmap_Channel);
        assert(0<=pos.y && pos.y<=Heatmap_outH);
        assert(0<=pos.x && pos.x<=Heatmap_outH);
        const float_t* column = &offset[(pos.y*offset_2_outH+pos.x)*offset_2_Channel];
        return float2_t(column[c+offset_2_Channel/2], column[c]);
    }

    void DecodeSinglePose();

    void overlay_skeleton(cv::Mat& img);

    void show_heatmap(int c);
};