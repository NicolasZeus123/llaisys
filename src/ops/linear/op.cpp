#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, in, weight, bias);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() &&  weight->isContiguous() && bias->isContiguous(), "Linear: all tensors must be contiguous.");
    } else {
        CHECK_SAME_DEVICE(out, in, weight);
        CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
        ASSERT(out->isContiguous() && in->isContiguous() &&  weight->isContiguous(), "Linear: all tensors must be contiguous.");
    }

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType())
    {
    case LLAISYS_DEVICE_CPU:
        if (bias != nullptr) {
            return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), in->shape()[0], in->shape()[1], weight->shape()[0], weight->dtype());
        } else {
            return cpu::linear(out->data(), in->data(), weight->data(), nullptr, in->shape()[0], in->shape()[1], weight->shape()[0], weight->dtype());
        }
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
