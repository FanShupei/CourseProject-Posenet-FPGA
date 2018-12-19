#include "config.h"
#include "pose_decoder.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

void PoseDecoder::DecodeSinglePose()
{
    pos2_t part_pos[Heatmap_Channel];
    float_t part_scores[Heatmap_Channel];

    memset(part_pos, 0, sizeof(part_pos));
    memset(part_scores, 0, sizeof(part_scores));

    for (int y=0; y<Heatmap_outH; y++)
        for (int x=0; x<Heatmap_outH; x++)
            for (int c=0; c<Heatmap_Channel; c++)
            {
                float_t score = get_score(c, y, x);
                // printf("%d %d %d %f\n", c, x, y, score);
                if (score > part_scores[c])
                {
                    part_scores[c] = score;
                    part_pos[c].x = x;
                    part_pos[c].y = y;
                }
            }

    for (int c=0; c<Heatmap_Channel; c++)
    {
        float2_t ofs = get_offset(c, part_pos[c]);
        results[c].x = part_pos[c].x*16+ofs.x;
        results[c].y = part_pos[c].y*16+ofs.y;
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

