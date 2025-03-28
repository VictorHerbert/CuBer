#include "primitives.h"

#include <string>

#include "stb_image_write.h"

Framebuffer::Framebuffer(int width, int height, Pixel* data){
    this->width = width;
    this->height = height;
    this->data = data;    
}

void Framebuffer::to_file(std::string filename){
    stbi_write_png(filename.c_str(), this->width, this->height, this->channels, this->data, this->width * this->channels);
}

void putPixel(Framebuffer& f, int2 pos, Pixel& color, float depth) {
    if (pos.x < 0 || pos.x >= f.width || pos.y < 0 || pos.y >= f.height)
        return;
    
    pos.y = (f.height - 1) - pos.y;

    int index = pos.x + pos.y * f.width;
    f.data[index] = color;
}