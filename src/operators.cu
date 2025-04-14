/*#include <math.h>

#include "operators.cuh"

float2 operator+(const float2& a, const float2& b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float3 operator+(const float3& a, const float& b){
    return make_float3(a.x + b, a.y + b, a.z + b);
}

float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float2 operator*(const float2& a, const float2& b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

float2 operator*(const float& a, const float2& b) {
    return make_float2(a * b.x, a * b.y);
}

float3 operator*(const float& a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

float2 operator*(const float& a, const int2& b){
    return make_float2(a * b.x, a * b.y);
}

float3 normalize(const float3& v) {
    float len = sqrt(dot(v, v));
    return make_float3(v.x / len, v.y / len, v.z / len);
}

float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
*/