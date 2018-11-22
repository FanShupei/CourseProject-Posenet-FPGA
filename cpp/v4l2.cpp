#include "v4l2.h"
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/ioctl.h>

#include <linux/videodev2.h>

#include "string.h"
#include "assert.h"
#include <string>
#include <vector>
#include <algorithm>

using std::string;
using std::vector;

#define CLEAR(x) memset(&(x), 0, sizeof(x))

struct buffer {
        void   *start;
        size_t  length;
};

struct Camera_V4L2: public ICamera
{
    int fd;
    CameraProp G_prop;

    vector<buffer> buffers;

    int init_device(CameraProp const& prop);
    bool assert_capability();
    int init_userp(uint32_t size);
public:
    int open(std::string const& deviceName, CameraProp const& prop) override;
    int stream_on() override;
    void stream_off() override;
    void mainloop(std::function<Callback> const& callback) override;
    const CameraProp& get_prop() override;
};

int Camera_V4L2::open(std::string const& deviceName, CameraProp const& prop)
{
    fd = ::open(deviceName.c_str(), O_RDWR);
    if (fd == -1)
    {
        perror("open device");
        return fd;
    }

    return init_device(prop);
}

bool Camera_V4L2::assert_capability()
{
    struct v4l2_capability cap;

    if (-1 == ioctl(fd, VIDIOC_QUERYCAP, &cap))
    {
        perror("ioctl VIDIOC_QUERYCAP");
        return false;
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "not video capture device\n");
        return false;
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "not support streaming i/o\n");
        return false;
    }

    return true;
}

int Camera_V4L2::init_userp(uint32_t buffer_size)
{
    struct v4l2_requestbuffers req;
    CLEAR(req);

    req.count  = 4;
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_USERPTR;

    if (-1 == ioctl(fd, VIDIOC_REQBUFS, &req)) {
        perror("VIDIOC_REQBUFS");
        return -1;
    }

    buffers.resize(4);

    for (auto& buf: buffers)
    {
        buf.length = buffer_size;
        buf.start = malloc(buffer_size);
    }

    return 0;
}

int Camera_V4L2::init_device(CameraProp const& prop)
{
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;
    unsigned int min;

    if (!assert_capability())
        return -1;

    CLEAR(fmt);
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = prop.width;
    fmt.fmt.pix.height      = prop.height;
    fmt.fmt.pix.pixelformat = prop.fourcc;
    fmt.fmt.pix.field       = V4L2_FIELD_ANY;

    if (-1 == ioctl(fd, VIDIOC_S_FMT, &fmt))
    {
        perror("VIDIOC_S_FMT");
        return -1;
    }

    G_prop.width     = fmt.fmt.pix.width;
    G_prop.height    = fmt.fmt.pix.height;
    G_prop.fourcc    = fmt.fmt.pix.pixelformat;
    G_prop.framesize = fmt.fmt.pix.sizeimage;

    if (-1 == init_userp(fmt.fmt.pix.sizeimage))
        return -1;

    return 0;
}

int Camera_V4L2::stream_on()
{
    for (int i = 0; i < buffers.size(); ++i)
    {
        struct v4l2_buffer buf;
        CLEAR(buf);

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_USERPTR;
        buf.index = i;
        buf.m.userptr = (unsigned long)buffers[i].start;
        buf.length = buffers[i].length;

        if (-1 == ioctl(fd, VIDIOC_QBUF, &buf))
        {
            perror("VIDIOC_QBUF");
            return -1;
        }       
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == ioctl(fd, VIDIOC_STREAMON, &type))
    {
        perror("VIDIOC_STREAMON");
        return -1;
    }
        
    return 0;
}

void Camera_V4L2::stream_off()
{
    // TODO
}

void Camera_V4L2::mainloop(std::function<Callback> const& callback)
{
    bool continue_flag = true;
    while (continue_flag)
    {
        struct v4l2_buffer buf;
        CLEAR(buf);

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_USERPTR;

        if (-1 == ioctl(fd, VIDIOC_DQBUF, &buf))
        {
            perror("VIDIOC_DQBUF");
            return;
        }

        assert(std::any_of(buffers.begin(), buffers.end(),
            [&buf](const buffer& buf2)
            {
                return buf.m.userptr == (unsigned long)buf2.start && 
                    buf.length == buf2.length;
            }));

        if (!callback((void *)buf.m.userptr, G_prop))
            continue_flag = false;

        if (-1 == ioctl(fd, VIDIOC_QBUF, &buf))
        {
            perror("VIDIOC_QBUF");
            exit(-1);
        }
    }        
}

const CameraProp& Camera_V4L2::get_prop()
{
    return G_prop;
}

CameraPtr create_v4l2_camera()
{
    return std::make_unique<Camera_V4L2>();
}