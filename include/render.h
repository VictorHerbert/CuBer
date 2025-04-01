#ifndef RENDER_H
#define RENDER_H

#include "primitives.h"

#include <cuda_runtime.h>


void drawLine(Framebuffer& f, int2 src, int2 dest);

void wireframeTriangle(Triangle2D& t, Framebuffer& f, Pixel& p);
void wireframeTriangle(Triangle& t, Framebuffer& f, Camera& c);

void drawTriangle(Triangle2D& t, Framebuffer& f);
void drawTriangle(Triangle& t, Framebuffer& f, Camera& c);


#endif