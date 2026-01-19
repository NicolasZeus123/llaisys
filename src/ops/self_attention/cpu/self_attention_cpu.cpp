#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale, size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv) {
    float inf = -std::numeric_limits<float>::infinity(); // -inf
    size_t past_len = total_len - seqlen;
    for (size_t i = 0; i < seqlen; i++) {
        // 当前每个新token
        for (size_t h_q = 0; h_q < nhead; h_q++) {
            // 每个query head
            size_t h_kv = h_q / (nhead / nkvhead); // 对应的kv head

            // 第一步：计算 scores(seqlen * total_len)
            std::vector<float> score(total_len, 0);
            for (size_t j = 0; j < total_len; j++) {
                score[j] = inf; // j > past_len + i时被掩盖
                if (j <= past_len + i) {
                    // 不掩盖
                    float dot = 0;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        for (size_t l = 0; l < d; l++) {
                            dot += llaisys::utils::cast<float>(q[i*nhead*d + h_q*d + l]) * llaisys::utils::cast<float>(k[j*nkvhead*d + h_kv*d + l]);
                        }
                    } else {
                        for (size_t l = 0; l < d; l++) {
                            dot += q[i*nhead*d + h_q*d + l] * k[j*nkvhead*d + h_kv*d + l];
                        }
                    }
                    score[j] = dot*scale;
                }
            }
            // 第二步，score进行SOFTMAX
            float max_score = inf;
            for (auto s: score) max_score = std::max(s, max_score);
            float sum_exp = 0, exp_val = 0;
            std::vector<float> alpha(total_len, 0);
            for (size_t j = 0; j < total_len; j++) {
                exp_val = std::exp(score[j]-max_score);
                alpha[j] = exp_val;
                sum_exp += exp_val;
            }
            for (size_t j = 0; j < total_len; j++) alpha[j] /= sum_exp;

            // 第三步，causalsoftmax(A)乘V
            for (size_t m = 0; m < dv; m++) {
                float out = 0;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    for (size_t j = 0; j < total_len; j++) { 
                        out += alpha[j] * llaisys::utils::cast<float>(v[j*nkvhead*dv + h_kv*dv + m]);
                    }
                    attn_val[i*nhead*dv + h_q*dv + m] = llaisys::utils::cast<T>(out);
                } else {
                    for (size_t j = 0; j < total_len; j++) { 
                        out += alpha[j] * v[j*nkvhead*dv + h_kv*dv + m];
                    }
                    attn_val[i*nhead*dv + h_q*dv + m] = out;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type, float scale, size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), scale, seqlen, nhead, d, total_len, nkvhead, dv);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q), reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v), scale, seqlen, nhead, d, total_len, nkvhead, dv);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q), reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v), scale, seqlen, nhead, d, total_len, nkvhead, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu