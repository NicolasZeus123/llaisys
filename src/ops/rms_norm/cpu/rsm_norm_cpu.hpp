#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rsm_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps, llaisysDataType_t type, size_t height, size_t width);
}