#include "framebuffer.cuh"

#include <string>
#include "primitives.cuh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"

void Framebuffer::save(std::string filename){
    stbi_write_png(filename.c_str(), this->size.x, this->size.y, this->channels, this->data, this->size.x * this->channels);
}

void Framebuffer::putPixel(int2 pos, Color col) {
    if (pos.x < 0 || pos.x >= this->size.x || pos.y < 0 || pos.y >= this->size.y)
        return;

    pos.y = (this->size.y - 1) - pos.y;

    int index = pos.x + pos.y * this->size.x;
    this->data[index] = col;
}


void Framebuffer::drawLine(uint2 src, uint2 dest, Color col) {
    int x0 = src.x, y0 = src.y;
    int x1 = dest.x, y1 = dest.y;

    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        this->putPixel({x0, y0}, col);

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
    }
}

void Framebuffer::wireframeTriangle(Triangle2Di& t, Color col) {
    // Draw the edges of the triangle using drawLine
    this->drawLine(t.v[0], t.v[1], col);
    this->drawLine(t.v[1], t.v[2], col);
    this->drawLine(t.v[2], t.v[0], col);
}