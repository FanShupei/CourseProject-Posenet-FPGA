#include "v4l2.h"
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#pragma GCC diagnostic ignored "-Wignored-attributes"

#include <arm_compute/core/ITensor.h>
#include <arm_compute/runtime/CL/CLTensor.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#include <arm_compute/core/CL/kernels/CLColorConvertKernel.h>

using std::unique_ptr;
using arm_compute::CLTensor;

class Proc
{
    int counter = 0;
    CLTensor tensor_yuv;
    CLTensor tensor_bgr;
    arm_compute::CLColorConvertKernel cvt_kernel;
public:
    Proc();
    bool operator()(void* data, CameraProp const& prop);

    static void init_CL_context();
};

int main()
{
    Proc::init_CL_context();
    Proc proc;

    CameraProp prop;
    prop.width  = 640;
    prop.height = 480;
    prop.fourcc = 0;
    prop.fps = 30;

    CameraPtr cam = create_v4l2_camera();

    if (-1 == cam->open("/dev/video0", prop))
    {
        fprintf(stderr, "open failed");
        return -1;
    }

    printf("open successful\n");

    if (-1 == cam->stream_on())
    {
        fprintf(stderr, "stream on failed\n");
        return -1;
    }

    printf("stream on successful\n");

    cam->mainloop(std::ref(proc));

    return 0;
}

using namespace std;
using namespace cv;
using namespace arm_compute;

void Proc::init_CL_context()
{
    // CLScheduler::get().default_init();
    auto& schd = CLScheduler::get();
    // schd.default_init();
    int idx = 1;
    // scanf("%d", &idx);

    vector<cl::Platform> platforms;
    vector<cl::Device> platform_devices;
    cl::Platform::get(&platforms);
    cl::Platform            p = platforms[idx];
    p.getDevices(CL_DEVICE_TYPE_ALL, &platform_devices);
    cl::Device device = platform_devices[0];

    printf("device number: %d\n", platform_devices.size());

    cl_context_properties properties[] =
    {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(p()),
        0
    };
    cl::Context ctx = cl::Context(device, properties);
    cl::CommandQueue queue = cl::CommandQueue(ctx, device);
    

    schd.init(ctx, queue, device);
    schd.set_target(GPUTarget::BIFROST);
    CLKernelLibrary::get().init("./cl_kernels/", ctx, device);
}

Proc::Proc()
{
    tensor_yuv.allocator()->init(TensorInfo(480, 640, Format::YUYV422));
    tensor_bgr.allocator()->init(TensorInfo(480, 640, Format::RGB888));
    tensor_yuv.allocator()->allocate();
    tensor_bgr.allocator()->allocate();

    printf("tensor yuv size: %d\n", tensor_yuv.info()->total_size());
    printf("tensor rgb size: %d\n", tensor_bgr.info()->total_size());

    cvt_kernel.configure(&tensor_yuv, &tensor_bgr);
}


bool Proc::operator()(void* data, CameraProp const& prop)
{
    auto& scheduler = CLScheduler::get();

    tensor_yuv.map();
    memcpy(tensor_yuv.buffer(), data, prop.framesize);
    tensor_yuv.unmap();

    scheduler.enqueue(cvt_kernel, false);

    tensor_bgr.map();
    {
        Mat3b img_rgb(480, 640, reinterpret_cast<Vec3b*>(tensor_bgr.buffer()));
        imshow("rgb", img_rgb);
    }
    tensor_bgr.unmap();

    printf("show successful\n");

    return waitKey(1) != 'q';
}