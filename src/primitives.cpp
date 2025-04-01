#include "primitives.h"

#include <string>
#include <iostream>

#include "operators.h"
#include "stb_image_write.h"

void Framebuffer::save(std::string filename){
    stbi_write_png(filename.c_str(), this->size.x, this->size.y, this->channels, this->data, this->size.x * this->channels);
}

void putPixel(Framebuffer& f, int2 pos, Pixel& color) {
    if (pos.x < 0 || pos.x >= f.size.x || pos.y < 0 || pos.y >= f.size.y)
        return;
    
    pos.y = (f.size.y - 1) - pos.y;

    int index = pos.x + pos.y * f.size.x;
    f.data[index] = color;
}

int2 Triangle2D::applyUV(float2 uv){
    float z = 1 - uv.x - uv.y;
    float2 fpos = uv.x*this->v[0] + uv.y*this->v[1] + z*this->v[2];
    return {fpos.x, fpos.y};
}