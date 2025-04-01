#ifndef PATHTRACER_H
#define PATHTRACER_H

#include "primitives.h"

bool rayTriangleIntersect(Ray& ray, Triangle& t, float3& barCoord);

float3 getBarCoord(Triangle2D& t, int2 p);

float3 sphereSampling(float3 seed);

void pathtrace(Ray& ray, Mesh& mesh, Framebuffer& f);

#endif