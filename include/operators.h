#ifndef OPERATORS_H
#define OPERATORS_H

#include <cuda_runtime.h>

__host__ __device__ float3 operator*(const float& a, const float3& b);
__host__ __device__ float3 operator+(const float3& a, const float3& b);
__host__ __device__ float3 operator-(const float3& a, const float3& b);
__host__ __device__ float3 cross(const float3& a, const float3& b);
__host__ __device__ float dot(const float3& a, const float3& b);
__host__ __device__ float3 normalize(const float3& v);
__host__ __device__ int crossProduct(int2 a, int2 b);
__host__ __device__ float crossProduct(float2 a, float2 b);

#endif