#ifndef SCAN_H
#define SCAN_H

#include <cuda_runtime.h>

#include "mesh.cuh"
#include "framebuffer.cuh"

struct QueueElement {
    uint2 uv;
    float3 pos;
    float3 normal;
};

struct WorkingQueue{
    size_t size;
    size_t capacity;
    QueueElement* elements;

    WorkingQueue(uint2 fSize);
    ~WorkingQueue();
};

void computeGI(CpuMesh& mesh, Framebuffer& f);
__global__ void computePixelQueue(int k, CudaMesh mesh, WorkingQueue queue, size_t* size);

#endif