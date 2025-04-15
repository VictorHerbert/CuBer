#include "mesh.cuh"

#include <cuda_runtime.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

#include "third_party/helper_math.h"

uint2 projectUV(float2 uv, uint2 fSize){
    return {
        uv.x * fSize.x,
        uv.y * fSize.y
    };
}

Triangle2Di projectUV(Triangle2D& uv, uint2 fSize){
    return {
        projectUV(uv.v[0], fSize),
        projectUV(uv.v[1], fSize),
        projectUV(uv.v[2], fSize)
    };
}

float3 calculateNormal(Triangle<float3>& t) {
    float3 edge1 = t.v[1] - t.v[0];
    float3 edge2 = t.v[2] - t.v[0];
    float3 normal = cross(edge1, edge2);
    return normalize(normal);
}

int edgeFunction(const uint2& a, const uint2& b, const uint2& c) {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

bool barCoordInside(float3 barCoord) {
    return (barCoord.x >= 0 && barCoord.y >= 0 && barCoord.z >= 0) || (barCoord.x <= 0 && barCoord.y <= 0 && barCoord.z <= 0);
}

__device__ __host__ float3 getBarCoord(uint2 pos, Triangle2Di& t){
    return {
        edgeFunction(t.v[1], t.v[2], pos),
        edgeFunction(t.v[2], t.v[0], pos),
        edgeFunction(t.v[0], t.v[1], pos)
    };
}

float3 applyBarCoord(float3 coord, const Triangle3D& t){
    
}

CpuMesh CpuMesh::fromObj(std::string filename, uint2 fBufferSize){
    std::ifstream file;
    file.open(filename);

    //if (!file)
    //    throw ERROR

    CpuMesh mesh;

    std::vector<float3> vertexes;
    std::vector<float2> uv;
    std::string cmd;
    //float3 currPoint;

    std::string line;

    while (std::getline(file, line)){
        std::replace(line.begin(), line.end(), '/', ' ');
        std::istringstream iss(line);
        iss >> cmd;

        if(cmd == "#")
            continue;
        else if(cmd == "mtllib")
            continue;
        else if(cmd == "o")
            continue;
        else if(cmd == "v"){
            //vertexes.push_back
            vertexes.push_back(float3());
            iss >>
                (vertexes.end()-1)->x >>
                (vertexes.end()-1)->y >>
                (vertexes.end()-1)->z;
        }
        else if(cmd == "vn")
            continue;
        else if(cmd == "vt"){
            uv.push_back(float2());
            iss >>
                (uv.end()-1)->x >>
                (uv.end()-1)->y;
        }
        else if(cmd == "s")
            continue;
        else if(cmd == "usemtl")
            continue;
        else if(cmd == "f"){
            Triangle3D vTriangle;
            Triangle2D uvTriangle;

            for(int i = 0; i < 3; i++){
                int v, vt, vn;
                iss >> v >> vt >> vn;
                vTriangle.v[i] = vertexes[v-1];
                uvTriangle.v[i] = uv[vt-1];
                //std::cout << v-1 << ": " << t.v[i].x << " " << t.v[i].y << " " << t.v[i].z << std::endl;
            }

            mesh.vTriangles.push_back(vTriangle);
            mesh.vUvs.push_back(uvTriangle);            
            mesh.vUvProj.push_back(projectUV(uvTriangle, fBufferSize));
            mesh.vNormals.push_back(calculateNormal(vTriangle));
        }
        else{
            std::cout << "Unrecognized instruction:" << cmd << std::endl;
            break;
        }
    }
    file.close();


    mesh.triangles = mesh.vTriangles.data();
    mesh.uvs = mesh.vUvs.data();
    mesh.uvProj = mesh.vUvProj.data();
    mesh.normals = mesh.vNormals.data();
    mesh.size = mesh.vTriangles.size();
    
    return mesh;
}

CudaMesh::CudaMesh(CpuMesh& mesh){
    cudaMalloc(&this->triangles, sizeof(Triangle3D) * mesh.vTriangles.size());
    cudaMalloc(&this->uvs, sizeof(Triangle2D) * mesh.vUvs.size());
    cudaMalloc(&this->uvProj, sizeof(Triangle2Di) * mesh.vUvProj.size());
    cudaMalloc(&this->normals, sizeof(float3) * mesh.vNormals.size());
    cudaMalloc(&this->materials, sizeof(Material) * mesh.vMaterials.size());

    cudaMemcpy(this->triangles, mesh.vTriangles.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(this->uvs, mesh.vUvs.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(this->uvProj, mesh.vUvProj.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(this->normals, mesh.vNormals.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(this->materials, mesh.vMaterials.data(), size, cudaMemcpyHostToDevice);

    this->size = mesh.size;
}

CudaMesh::~CudaMesh(){
    cudaFree(this->triangles);
    cudaFree(this->uvs);
    cudaFree(this->uvProj);
    cudaFree(this->normals);
    cudaFree(this->materials);
}