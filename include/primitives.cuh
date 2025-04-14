#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <limits>

const float INF = std::numeric_limits<float>::infinity();

struct Color{
    uchar1 r,g,b;
};

Color gamma_correction(float3 light);

#endif