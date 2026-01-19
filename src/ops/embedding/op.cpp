#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous.");

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType())
    {
    case LLAISYS_DEVICE_CPU:
        /* code */
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->shape()[0], weight->shape()[1]);
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
