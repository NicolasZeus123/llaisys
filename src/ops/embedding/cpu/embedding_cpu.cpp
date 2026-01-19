#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t shape_index, size_t width){
    for (size_t i = 0; i < shape_index; i++) {
        for (size_t j = 0; j < width; j++) {
            out[i*width+j] = weight[index[i]*width+j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t shape_index, size_t width) {
    switch (type)
    {
    case LLAISYS_DTYPE_F32:
        return embedding_(
            reinterpret_cast<float *>(out), 
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const float *>(weight),
            shape_index, width
        );
    case LLAISYS_DTYPE_BF16:
        return embedding_(
            reinterpret_cast<llaisys::bf16_t *>(out), 
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            shape_index, width
        );
    case LLAISYS_DTYPE_F16:
        return embedding_(
            reinterpret_cast<llaisys::fp16_t *>(out), 
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            shape_index, width
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu