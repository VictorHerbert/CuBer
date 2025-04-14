#include "baker.cuh"

#include "scan.cuh"

void computeGI(CpuMesh& mesh, Framebuffer& f){
    WorkingQueue queue(f.size);
    CudaMesh cudaMesh(mesh);

    computePixelQueue<<<2,2>>>(cudaMesh, queue);

    std::vector<QueueElement> cpuQueue(queue.size);
    cudaMemcpy(queue.elements, cpuQueue.data(), queue.size * sizeof(QueueElement), cudaMemcpyDeviceToHost);

    return;
}