/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once

#include "Core/Macros.h"

#include <cstdint>
#include <limits>

namespace USTC_CG
{
namespace math
{

HD_USTC_CG_API uint16_t float32ToFloat16(float value);
HD_USTC_CG_API float float16ToFloat32(uint16_t value);

struct float16_t
{
    float16_t() = default;

    float16_t(uint32_t sign, uint32_t exponent, uint32_t fraction)
        : mBits((sign & 0x01) << 15 | (exponent & 0x1f) << 10 | (fraction & 0x03ff))
    {}

    explicit float16_t(float value) : mBits(float32ToFloat16(value)) {}

    template<typename T>
    explicit float16_t(T value) : mBits(float32ToFloat16(static_cast<float>(value)))
    {}

    operator float() const { return float16ToFloat32(mBits); }

    static constexpr float16_t fromBits(uint16_t bits) { return float16_t(bits, FromBits); }
    uint16_t toBits() const { return mBits; }

    bool operator==(const float16_t other) const { return mBits == other.mBits; }
    bool operator!=(const float16_t other) const { return mBits != other.mBits; }
    bool operator<(const float16_t other) const { return static_cast<float>(*this) < static_cast<float>(other); }
    bool operator<=(const float16_t other) const { return static_cast<float>(*this) <= static_cast<float>(other); }
    bool operator>(const float16_t other) const { return static_cast<float>(*this) > static_cast<float>(other); }
    bool operator>=(const float16_t other) const { return static_cast<float>(*this) >= static_cast<float>(other); }

    float16_t operator+() const { return *this; }
    float16_t operator-() const { return fromBits(mBits ^ 0x8000); }

    // TODO: Implement math operators in native fp16 precision. For now using fp32.
    float16_t operator+(const float16_t other) const { return float16_t(static_cast<float>(*this) + static_cast<float>(other)); }
    float16_t operator-(const float16_t other) const { return float16_t(static_cast<float>(*this) - static_cast<float>(other)); }
    float16_t operator*(const float16_t other) const { return float16_t(static_cast<float>(*this) * static_cast<float>(other)); }
    float16_t operator/(const float16_t other) const { return float16_t(static_cast<float>(*this) / static_cast<float>(other)); }

    float16_t operator+=(const float16_t other) { return *this = *this + other; }
    float16_t operator-=(const float16_t other) { return *this = *this - other; }
    float16_t operator*=(const float16_t other) { return *this = *this * other; }
    float16_t operator/=(const float16_t other) { return *this = *this / other; }

    constexpr bool isFinite() const noexcept { return exponent() < 31; }
    constexpr bool isInf() const noexcept { return exponent() == 31 && mantissa() == 0; }
    constexpr bool isNan() const noexcept { return exponent() == 31 && mantissa() != 0; }
    constexpr bool isNormalized() const noexcept { return exponent() > 0 && exponent() < 31; }
    constexpr bool isDenormalized() const noexcept { return exponent() == 0 && mantissa() != 0; }

private:
    enum Tag
    {
        FromBits
    };

    constexpr float16_t(uint16_t bits, Tag) : mBits(bits) {}

    constexpr uint16_t mantissa() const noexcept { return mBits & 0x3ff; }
    constexpr uint16_t exponent() const noexcept { return (mBits >> 10) & 0x001f; }

    uint16_t mBits;
};

#if FALCOR_MSVC
#pragma warning(push)
#pragma warning(disable : 4455) // disable warning about literal suffixes not starting with an underscore
#elif FALCOR_CLANG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuser-defined-literals"
#elif FALCOR_GCC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wliteral-suffix"
#endif

/// h suffix for "half float" literals.
inline float16_t operator""h(long double value)
{
    return float16_t(static_cast<float>(value));
}

#if FALCOR_MSVC
#pragma warning(pop)
#elif FALCOR_CLANG
#pragma clang diagnostic pop
#elif FALCOR_GCC
#pragma GCC diagnostic pop
#endif

} // namespace math
} // namespace USTC_CG

namespace std
{

template<>
class numeric_limits<USTC_CG::math::float16_t>
{
public:
    static constexpr bool is_specialized = true;
    static constexpr USTC_CG::math::float16_t min() noexcept { return USTC_CG::math::float16_t::fromBits(0x0200); }
    static constexpr USTC_CG::math::float16_t max() noexcept { return USTC_CG::math::float16_t::fromBits(0x7bff); }
    static constexpr USTC_CG::math::float16_t lowest() noexcept { return USTC_CG::math::float16_t::fromBits(0xfbff); }
    static constexpr int digits = 11;
    static constexpr int digits10 = 3;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr int radix = 2;
    static constexpr USTC_CG::math::float16_t epsilon() noexcept { return USTC_CG::math::float16_t::fromBits(0x1200); }
    static constexpr USTC_CG::math::float16_t round_error() noexcept { return USTC_CG::math::float16_t::fromBits(0x3c00); }
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr float_denorm_style has_denorm = denorm_absent;
    static constexpr bool has_denorm_loss = false;
    static constexpr USTC_CG::math::float16_t infinity() noexcept { return USTC_CG::math::float16_t::fromBits(0x7c00); }
    static constexpr USTC_CG::math::float16_t quiet_NaN() noexcept { return USTC_CG::math::float16_t::fromBits(0x7fff); }
    static constexpr USTC_CG::math::float16_t signaling_NaN() noexcept { return USTC_CG::math::float16_t::fromBits(0x7dff); }
    static constexpr USTC_CG::math::float16_t denorm_min() noexcept { return USTC_CG::math::float16_t::fromBits(0); }
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr float_round_style round_style = round_to_nearest;
};
} // namespace std
