#ifndef PATHTRACER_H
#define PATHTRACER_H

#include "primitives.cuh"
#include "mesh.cuh"
#include "framebuffer.cuh"

const int NOT_FOUND = -1;

struct Ray {
    float3 origin;
    float3 direction;  // Must be normalized
};

float3 pathtrace(Ray& ray, Mesh& mesh, Framebuffer& f);
float3 hemisphereSampling(float3 normal, float2 seed);
float rayTriangleDist(Ray& ray, Triangle3D& t);

#endif