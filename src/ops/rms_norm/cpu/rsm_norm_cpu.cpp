#include "rsm_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rsm_norm_(T *out, const T *in, const T* weight, float eps, size_t height, size_t width) {
    for (size_t i = 0; i < height; i++) {
        float div = 0;
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) { 
            for (size_t j = 0; j < width; j++) {
                float in_temp = llaisys::utils::cast<float>(in[i*width+j]);
                div += in_temp * in_temp;
            }
        } else { 
            for (size_t j = 0; j < width; j++) {
                div += in[i*width+j] * in[i*width+j];
            }
        }
        div /= width;
        div += eps;
        div = sqrt(div);
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            for (size_t j = 0; j < width; j++) {
                out[i*width+j] = llaisys::utils::cast<T>(
                    llaisys::utils::cast<float>(weight[j]) * llaisys::utils::cast<float>(in[i*width+j]) / div
                );
            }
        } else {
            for (size_t j = 0; j < width; j++) {
                out[i*width+j] = weight[j] * in[i*width+j] / div;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rsm_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t type, size_t height, size_t width){
    switch (type)
    {
    case LLAISYS_DTYPE_F32:
        return rsm_norm_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            eps, height, width
        );
    case LLAISYS_DTYPE_BF16:
        return rsm_norm_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            eps, height, width
        );
    case LLAISYS_DTYPE_F16:
        return rsm_norm_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            eps, height, width
        );
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu