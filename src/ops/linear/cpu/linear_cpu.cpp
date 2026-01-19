#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t height_in, size_t width_in, size_t height_weight) {
    for (size_t i = 0; i < height_in; i++) {
        for (size_t j = 0; j < height_weight; j++) {
            // 计算第(i,j)处的out
            float sum_temp = 0;
            if (bias != nullptr) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    sum_temp = llaisys::utils::cast<float>(bias[j]);
                } else {
                    sum_temp = bias[j];
                }
            } else {
                sum_temp = 0;
            }
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                for (size_t k = 0; k < width_in; k++) {
                    sum_temp += llaisys::utils::cast<float>(in[i*width_in+k]) * 
                                llaisys::utils::cast<float>(weight[j*width_in+k]);
                }
            } else {
                for (size_t k = 0; k < width_in; k++) {
                    // out[i][j] += in[i][k] * weight[j][k]
                    sum_temp += in[i*width_in+k] * weight[j*width_in+k];
                }
            }
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[i*height_weight+j] = llaisys::utils::cast<T>(sum_temp);
            } else {
                out[i*height_weight+j] = sum_temp;
            }
        }
    }
}

namespace llaisys::ops::cpu {
    void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias, size_t height_in, size_t width_in, size_t height_weight, llaisysDataType_t type){
        switch (type)
        {
        case LLAISYS_DTYPE_F32:
            return linear_(
                reinterpret_cast<float *>(out),
                reinterpret_cast<const float *>(in),
                reinterpret_cast<const float *>(weight),
                reinterpret_cast<const float *>(bias),
                height_in, width_in, height_weight
            );
        case LLAISYS_DTYPE_BF16:
            return linear_(
                reinterpret_cast<llaisys::bf16_t *>(out),
                reinterpret_cast<const llaisys::bf16_t *>(in),
                reinterpret_cast<const llaisys::bf16_t *>(weight),
                reinterpret_cast<const llaisys::bf16_t *>(bias),
                height_in, width_in, height_weight
            );
        case LLAISYS_DTYPE_F16:
            return linear_(
                reinterpret_cast<llaisys::fp16_t *>(out),
                reinterpret_cast<const llaisys::fp16_t *>(in),
                reinterpret_cast<const llaisys::fp16_t *>(weight),
                reinterpret_cast<const llaisys::fp16_t *>(bias),
                height_in, width_in, height_weight
            );
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
} // namespace llaisys::ops::cpu