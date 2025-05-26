#ifndef MESH_H
#define MESH_H

#include <vector> 
#include <string>
#include <cuda_runtime.h>

#include "primitives.cuh"

template<typename T>
struct Triangle{
    T v[3];
};

typedef Triangle<float3> Triangle3D;
typedef Triangle<float2> Triangle2D;
typedef Triangle<uint2> Triangle2Di;

struct Material {
    bool light = false;
};

struct Mesh{
    size_t size;

    Triangle3D* triangles;
    Triangle2D* uvs;
    Triangle2Di* uvProj;
    float3* normals;
    Material* materials;
};

struct CpuMesh : public Mesh{
    std::vector<Triangle3D> vTriangles;
    std::vector<Triangle2D> vUvs;
    std::vector<Triangle2Di> vUvProj;
    std::vector<float3> vNormals;
    std::vector<Material> vMaterials;

    static CpuMesh fromObj(std::string filename, uint2 fBufferSize);
};

struct CudaMesh : public Mesh{
    CudaMesh (CpuMesh& mesh);
    void free();
};


uint2 projectUV(float2 uv, uint2 fSize);

Triangle2Di projectUV(Triangle2D& uv, uint2 fSize);

float3 calculateNormal(Triangle3D& t);

__host__ __device__ int edgeFunction(const uint2& a, const uint2& b, const uint2& c);

__device__ __host__ float3 getBarCoord(uint2 pos, Triangle2Di& t);

__device__ __host__ bool barCoordInside(float3 barCoord);

__device__ __host__ float3 applyBarCoord(float3 coord, const Triangle3D& t);

#endif