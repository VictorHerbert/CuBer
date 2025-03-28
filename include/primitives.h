#ifndef SCENE_H
#define SCENE_H

#include <string>
#include <cuda_runtime.h>


typedef int Material;

struct  __attribute__((packed)) Pixel{
    uchar1 r,g,b;
};

struct Triangle {
    float3 v[3];
    float2 uv[3];
    float3 normal;
    Material mat;
    int id;
};

struct Mesh{
    size_t size;
    Triangle* triangles;
};

struct Ray {    
    float3 origin;
    float3 direction;  // Must be normalized
};

struct Triangle2D{
    int2 v[3];
};

struct Camera{
    float3 pos;
    float3 plane;
    float3 dir;    
};

struct Framebuffer{
    float2 size;
    Pixel* data;
    float* zbuffer;

    const int channels = 3;

    void save(std::string filename);    
};

void putPixel(Framebuffer& f, int2 pos, Pixel& color);

#endif