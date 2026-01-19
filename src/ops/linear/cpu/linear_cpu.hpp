#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias, size_t height_in, size_t width_in, size_t height_weight, llaisysDataType_t type);
}