#ifndef PATHTRACER_H
#define PATHTRACER_H

#include "primitives.h"


void pathtraceTriangle(Triangle& t, Framebuffer& f);
Pixel getLight(Ray& ray);

#endif