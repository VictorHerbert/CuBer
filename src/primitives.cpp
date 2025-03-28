#include "primitives.h"

#include <string>
#include <iostream>

#include "stb_image_write.h"

void Framebuffer::save(std::string filename){
    stbi_write_png(filename.c_str(), this->size.x, this->size.y, this->channels, this->data, this->size.x * this->channels);
}

void putPixel(Framebuffer& f, int2 pos, Pixel& color) {
    if (pos.x < 0 || pos.x >= f.size.x || pos.y < 0 || pos.y >= f.size.y)
        return;
    
    pos.y = (f.size.y - 1) - pos.y;

    //std::cout << pos.x << " " << pos.y << std::endl;

    int index = pos.x + pos.y * f.size.x;
    f.data[index] = color;
}