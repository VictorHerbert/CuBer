#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <string>

#include "primitives.cuh"
#include "mesh.cuh"

struct Framebuffer{
    uint2 size;
    Color* data;
    const int channels = 3;

    void save(std::string filename);
    void putPixel(int2 pos, Color color);
    void drawLine(uint2 src, uint2 dest, Color col);
    void wireframeTriangle(Triangle2Di& t, Color col);
};

#endif