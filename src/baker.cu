#include "baker.cuh"

#include "scan.cuh"


void computeLightmap(CpuMesh& mesh, Framebuffer& f){
    WorkingQueue queue(f.size);
    CudaMesh cudaMesh(mesh);

    size_t* sizePtr;
    cudaMallocManaged(&sizePtr, sizeof(size_t));
    *sizePtr = 0;

    computePixelQueue<<<2,3>>>(cudaMesh, queue, sizePtr);
    cudaDeviceSynchronize();

    queue.size = *sizePtr;

    std::vector<QueueElement> cpuQueue(queue.size);
    cudaMemcpy(cpuQueue.data(), queue.elements, queue.size * sizeof(QueueElement), cudaMemcpyDeviceToHost);    

    for(QueueElement el : cpuQueue){
        f.putPixel(make_int2(el.uv.x, el.uv.y), {255, 255, 255});
    }


    cudaMesh.free();
    queue.free();

    return;
}