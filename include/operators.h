#ifndef OPERATORS_H
#define OPERATORS_H

#include <cuda_runtime.h>

float3 normalize(const float3& v);

float2 operator+(const float2& a, const float2& b);
float3 operator+(const float3& a, const float3& b);
float3 operator-(const float3& a, const float3& b);

float2 operator*(const float& a, const float2& b);
float2 operator*(const float2& a, const float2& b);

float3 operator*(const float& a, const float3& b);


float3 cross(const float3& a, const float3& b);

float dot(const float3& a, const float3& b);
int cross(int2 a, int2 b);
float cross(float2 a, float2 b);

#endif