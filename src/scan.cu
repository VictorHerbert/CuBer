#include "scan.cuh"

#include "mesh.cuh"
#include "framebuffer.cuh"

#include <iostream>

WorkingQueue::WorkingQueue(uint2 fSize){
    this->capacity = fSize.x * fSize.y;
    cudaMalloc(&this->elements, this->capacity * sizeof(QueueElement));
}

WorkingQueue::~WorkingQueue(){
    cudaFree(elements);
}

__global__ void computePixelQueue(int k, CudaMesh mesh, WorkingQueue queue, size_t* size) {
    //printf("T [%d, %d, %d]>", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("T [%u, %u, %u]> %lu %u\n", threadIdx.x, threadIdx.y, threadIdx.z, mesh.size, k);

    int triIdx = blockIdx.x;
    if (triIdx >= mesh.size) return;

    Triangle3D tri = mesh.triangles[triIdx];
    Triangle2Di uv = mesh.uvProj[triIdx];

    uint minX = min(uv.v[0].x, min(uv.v[1].x, uv.v[2].x));
    uint minY = min(uv.v[0].y, min(uv.v[1].y, uv.v[2].y));
    uint maxX = max(uv.v[0].x, max(uv.v[1].x, uv.v[2].x));
    uint maxY = max(uv.v[0].y, max(uv.v[1].y, uv.v[2].y));

    printf("X: %d %d\n", minX, maxX);
    printf("Y: %d %d\n", minY, maxY);

    uint2 pos;

    for (pos.x = threadIdx.x + minX; pos.x <= maxX; pos.x += blockDim.x) {
        for (pos.y = threadIdx.y + minY; pos.y <= maxY; pos.y += blockDim.y) {

            float3 barCoord = getBarCoord(pos, uv);

            if (barCoordInside(barCoord)) {
                int index = atomicAdd((unsigned long long*) &queue.size, (unsigned long long) 1);
                queue.elements[queue.size] = {
                    pos,
                    applyBarCoord(barCoord, tri),
                    mesh.normals[triIdx]
                };

                printf("%d %d added to queue\n", pos.x, pos.y);
            }
        }
    }
}
