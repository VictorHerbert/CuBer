#include "baker.cuh"

#include "scan.cuh"


void computeGI(CpuMesh& mesh, Framebuffer& f){
    WorkingQueue queue(f.size);
    CudaMesh cudaMesh(mesh);

    size_t* size;
    cudaMallocManaged(&size, sizeof(size_t));
    *size = 0;

    printf("Mesh size: %d\n", mesh.size); 

    computePixelQueue<<<2,3>>>(666, cudaMesh, queue, size);
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

    printf("QueueSize: %d\n", *size);

    std::vector<QueueElement> cpuQueue(queue.size);
    cudaMemcpy(queue.elements, cpuQueue.data(), queue.size * sizeof(QueueElement), cudaMemcpyDeviceToHost);

    return;
}