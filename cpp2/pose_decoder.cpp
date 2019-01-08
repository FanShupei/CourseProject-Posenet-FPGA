#include "config.h"
#include "pose_decoder.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void PoseDecoder::DecodeSinglePose()
{
    const float_t SCORE_THRESHOLD = 0.05;

    pos2_t part_pos[Heatmap_Channel];
    float_t part_scores[Heatmap_Channel];

    memset(valid, 0, sizeof(valid));

    memset(part_pos, 0, sizeof(part_pos));
    memset(part_scores, 0, sizeof(part_scores));

    for (int y=0; y<Heatmap_outH; y++)
        for (int x=0; x<Heatmap_outH; x++)
            for (int c=0; c<Heatmap_Channel; c++)
            {
                float_t score = get_score(c, y, x);
                // printf("%d %d %d %f\n", c, x, y, score);
                if (score >= SCORE_THRESHOLD && score > part_scores[c])
                {
                    valid[c] = true;
                    part_scores[c] = score;
                    part_pos[c].x = x;
                    part_pos[c].y = y;
                }
            }

    for (int c=0; c<Heatmap_Channel; c++)
    {
        float2_t ofs = get_offset(c, part_pos[c]);
        results[c].x = part_pos[c].x*OUTPUT_STRIDE+ofs.x;
        results[c].y = part_pos[c].y*OUTPUT_STRIDE+ofs.y;
    }
}

void PoseDecoder::overlay_skeleton(cv::Mat& img)
{
    int len = (Heatmap_outH-1) * OUTPUT_STRIDE + 1;
    assert(img.cols == len && img.rows == len);

    for (int c=0; c<Heatmap_Channel; c++)
    {
        if (valid[c])
        {
            float_t x = results[c].x;
		    float_t y = results[c].y;
		    cv::circle(img, cv::Point(x, y), 3, cv::Scalar(0,0,1), CV_FILLED);
        }
    }

    const int LINE_COUNT=16;

    const int L1[LINE_COUNT] =
    {
        0, 1, 0, 2,
        5, 5, 7, 6, 8,
        5, 6, 11,
        11, 13, 12, 14
    };
    const int L2[LINE_COUNT] =
    {
        1, 3, 2, 4,
        6, 7, 9, 8, 10,
        11, 12, 12,
        13, 15, 14, 16
    };

    for (int i=0; i<LINE_COUNT; i++)
    {
        int c1 = L1[i], c2 = L2[i];
        if (valid[c1] && valid[c2])
        {
            float_t x1 = results[c1].x;
            float_t y1 = results[c1].y;
            float_t x2 = results[c2].x;
            float_t y2 = results[c2].y;

            cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,0,1), 1);
        }
    }
}

void PoseDecoder::show_heatmap(int c)
{
    cv::Mat1f hm(Heatmap_outH, Heatmap_outH);
    for (int y=0; y<Heatmap_outH; y++)
        for (int x=0; x<Heatmap_outH; x++)
            hm(y, x) = get_score(c, y, x);

    cv::imshow("heatmap", hm);
}

