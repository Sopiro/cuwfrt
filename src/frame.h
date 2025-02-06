#pragma once

#include "api.cuh"
#include "common.h"

namespace cuwfrt
{

inline __cpu_gpu__ Float CosTheta(const Vec3& w)
{
    return w.z;
}

inline __cpu_gpu__ Float Cos2Theta(const Vec3& w)
{
    return Sqr(w.z);
}

inline __cpu_gpu__ Float AbsCosTheta(const Vec3& w)
{
    return std::abs(w.z);
}

inline __cpu_gpu__ Float Sin2Theta(const Vec3& w)
{
    return fmax(Float(0), 1 - Cos2Theta(w));
}

inline __cpu_gpu__ Float SinTheta(const Vec3& w)
{
    return std::sqrt(Sin2Theta(w));
}

inline __cpu_gpu__ Float TanTheta(const Vec3& w)
{
    return SinTheta(w) / CosTheta(w);
}

inline __cpu_gpu__ Float Tan2Theta(const Vec3& w)
{
    return Sin2Theta(w) / Cos2Theta(w);
}

inline __cpu_gpu__ Float CosPhi(const Vec3& w)
{
    Float sin_theta = SinTheta(w);
    return (sin_theta == 0) ? 1 : Clamp(w.x / sin_theta, -1, 1);
}

inline __cpu_gpu__ Float SinPhi(const Vec3& w)
{
    Float sin_theta = SinTheta(w);
    return (sin_theta == 0) ? 0 : Clamp(w.y / sin_theta, -1, 1);
}

inline __cpu_gpu__ Float SphericalTheta(const Vec3& v)
{
    return std::acos(Clamp(v.y, -1, 1));
}

inline __cpu_gpu__ Float SphericalPhi(const Vec3& v)
{
    Float r = std::atan2(v.z, v.x);
    return r < 0 ? r + two_pi : r;
}

inline __cpu_gpu__ Vec3 SphericalDirection(Float theta, Float phi)
{
    Float sin_theta = std::sin(theta);
    return Vec3(std::cos(phi) * sin_theta, std::cos(theta), std::sin(phi) * sin_theta);
}

inline __cpu_gpu__ Vec3 SphericalDirection(Float sin_theta, Float cos_theta, Float phi)
{
    return Vec3(std::cos(phi) * sin_theta, cos_theta, std::sin(phi) * sin_theta);
}

inline __cpu_gpu__ Vec3 SphericalDirection(Float sin_theta, Float cos_theta, Float sin_phi, Float cos_phi)
{
    return Vec3(cos_phi * sin_theta, cos_theta, sin_phi * sin_theta);
}

// Assume it's a standard shading coordinate system
inline __cpu_gpu__ bool SameHemisphere(const Vec3& a, const Vec3& b)
{
    return a.z * b.z > 0;
}

inline __cpu_gpu__ void CoordinateSystem(const Vec3& v1, Vec3* v2, Vec3* v3)
{
    Float sign = std::copysign(1.0f, v1.z);
    Float a = -1 / (sign + v1.z);
    Float b = v1.x * v1.y * a;
    *v2 = Vec3(1 + sign * (v1.x * v1.x) * a, sign * b, -sign * v1.x);
    *v3 = Vec3(b, sign + (v1.y * v1.y) * a, -v1.y);
}

// Represents orthonormal coordinate frame
struct Frame
{
    __cpu_gpu__ static Frame FromXZ(const Vec3& x, const Vec3& z);
    __cpu_gpu__ static Frame FromXY(const Vec3& x, const Vec3& y);
    __cpu_gpu__ static Frame FromX(const Vec3& x);
    __cpu_gpu__ static Frame FromY(const Vec3& y);
    __cpu_gpu__ static Frame FromZ(const Vec3& z);

    Frame() = default;
    __cpu_gpu__ Frame(const Vec3& n);
    __cpu_gpu__ Frame(const Vec3& x, const Vec3& y, const Vec3& z);

    // Convert from local coordinates to world coordinates
    __cpu_gpu__ Vec3 FromLocal(const Vec3& v) const;

    // Convert from world coordinates to local coordinates
    __cpu_gpu__ Vec3 ToLocal(const Vec3& v) const;

    __cpu_gpu__ Vec3& operator[](int32 i);
    __cpu_gpu__ Vec3 operator[](int32 i) const;

    Vec3 x, y, z;
};

__cpu_gpu__ inline Frame Frame::FromXZ(const Vec3& x, const Vec3& z)
{
    return Frame(x, Cross(z, x), z);
}

__cpu_gpu__ inline Frame Frame::FromXY(const Vec3& x, const Vec3& y)
{
    return Frame(x, y, Cross(x, y));
}

__cpu_gpu__ inline Frame Frame::FromX(const Vec3& x)
{
    Vec3 y, z;
    CoordinateSystem(x, &y, &z);
    return Frame(x, y, z);
}

__cpu_gpu__ inline Frame Frame::FromY(const Vec3& y)
{
    Vec3 z, x;
    CoordinateSystem(y, &z, &x);
    return Frame(x, y, z);
}

__cpu_gpu__ inline Frame Frame::FromZ(const Vec3& z)
{
    Vec3 x, y;
    CoordinateSystem(z, &x, &y);
    return Frame(x, y, z);
}

__cpu_gpu__ inline Frame::Frame(const Vec3& n)
    : z{ n }
{
    CoordinateSystem(n, &x, &y);
}

__cpu_gpu__ inline Frame::Frame(const Vec3& x, const Vec3& y, const Vec3& z)
    : x{ x }
    , y{ y }
    , z{ z }
{
}

__cpu_gpu__ inline Vec3& Frame::operator[](int32 i)
{
    return (&x)[i];
}

__cpu_gpu__ inline Vec3 Frame::operator[](int32 i) const
{
    return (&x)[i];
}

__cpu_gpu__ inline Vec3 Frame::FromLocal(const Vec3& v) const
{
    return v.x * x + v.y * y + v.z * z;
}

__cpu_gpu__ inline Vec3 Frame::ToLocal(const Vec3& v) const
{
    return Vec3(Dot(v, x), Dot(v, y), Dot(v, z));
}

} // namespace cuwfrt
