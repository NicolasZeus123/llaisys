#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seqlen, size_t nhead, size_t d) {
    for (size_t seq = 0; seq < seqlen; seq++) {
        int64_t p = pos_ids[seq];
        for (size_t i = 0; i < nhead; i++) {
            size_t offset_a = seq*nhead*d+i*d;
            size_t offset_b = offset_a + d/2;
            for (size_t j = 0; j < d/2; j++) {
                double fai = p * std::pow(static_cast<double>(theta), -2* static_cast<double>(j)/d); 
                float cos_fai = std::cos(fai);
                float sin_fai = std::sin(fai);

                float a, b;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a = llaisys::utils::cast<float>(in[offset_a+j]);
                    b = llaisys::utils::cast<float>(in[offset_b+j]);
                } else {
                    a = in[offset_a+j];
                    b = in[offset_b+j];
                }
                float a_out = a*cos_fai-b*sin_fai;
                float b_out = b*cos_fai+a*sin_fai;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[offset_a+j] = llaisys::utils::cast<T>(a_out);
                    out[offset_b+j] = llaisys::utils::cast<T>(b_out);
                } else {
                    out[offset_a+j] = a_out;
                    out[offset_b+j] = b_out;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t d) {
        switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}