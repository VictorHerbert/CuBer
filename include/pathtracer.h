#ifndef PATHTRACER_H
#define PATHTRACER_H

#include "primitives.h"

bool rayTriangleIntersect(Ray& ray, Triangle& t, float3& barCoord);

float3 getBarCoord(Triangle2D& t, int2 p);

void pathtrace(Ray& ray, Mesh& mesh, Framebuffer& f);

#endif