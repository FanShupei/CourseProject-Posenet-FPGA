#include <memory>
#include <functional>

struct CameraProp
{
    int width;
    int height;
    uint32_t fourcc;
    uint32_t framesize;
    int fps;
};

class ICamera
{
public:
    using Callback = bool (void* data, CameraProp const& prop);
    virtual ~ICamera() { }
    virtual int open(std::string const& deviceName, CameraProp const& prop) = 0;
    virtual int stream_on() = 0;
    virtual void stream_off() = 0;
    virtual void mainloop(std::function<Callback> const& callback) = 0;
    virtual const CameraProp& get_prop() = 0;
};

using CameraPtr = std::unique_ptr<ICamera>;

CameraPtr create_v4l2_camera();