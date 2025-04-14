#include "pathtracer.cuh"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#include "primitives.cuh"
#include "third_party/helper_math.h"

float rayTriangleDist(Ray& ray, Triangle3D& t) {
    float3 edge1 = t.v[1]-t.v[0];
    float3 edge2 = t.v[2]-t.v[0];

    float3 h = cross(ray.direction, edge2);
    float a = dot(edge1, h);

    if (std::abs(a) < 1e-6f)
        return INF;

    float f = 1.0f / a;
    float3 s = ray.origin - t.v[0];
    float u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f)
        return INF;

    float3 q = cross(s, edge1);
    float v = f * dot(ray.direction, q);

    if (v < 0.0f || u + v > 1.0f)
        return INF;

    return f * dot(edge2, q);
}

float3 hemisphereSampling(float3 normal, float2 seed) {
    float z = 1.0 - 2.0 * seed.x;
    float r = sqrt(1.0 - seed.y * seed.y);

    float angle = M_PI * seed.y;
    float3 dir = {r * cos(angle), r * sin(angle), z};

    return dot(dir, normal) < 0.0 ? -dir : dir;
}

float randFloat(){
    return (float) rand()/RAND_MAX;
}

int ray_count = 1e2;

/*
void iterateTriangle(Mesh& mesh, Framebuffer& framebuffer, int idx){
    Triangle2Di uv = mesh.uvProj[idx];

    // Sort vertices by Y-coordinate (bottom to top)
    if (uv.v[0].y > uv.v[1].y) std::swap(uv.v[0], uv.v[1]);
    if (uv.v[1].y > uv.v[2].y) std::swap(uv.v[1], uv.v[2]);
    if (uv.v[0].y > uv.v[1].y) std::swap(uv.v[0], uv.v[1]);

    // Step through the triangle row by row (scanline)
    for (int y = uv.v[0].y; y <= uv.v[2].y; ++y) {
        // Find the left and right x boundaries for the current scanline
        int xLeft = uv.v[0].x, xRight = uv.v[0].x;

        if (y >= uv.v[1].y) {
            // Second half of triangle: interpolate between v[1] and v[2]
            float alpha = float(y - uv.v[1].y) / float(uv.v[2].y - uv.v[1].y);
            xLeft = int((1 - alpha) * uv.v[0].x + alpha * uv.v[1].x);
            xRight = int((1 - alpha) * uv.v[0].x + alpha * uv.v[2].x);
        } else {
            // First half of triangle: interpolate between v[0] and v[1]
            float alpha = float(y - uv.v[0].y) / float(uv.v[1].y - uv.v[0].y);
            xLeft = int((1 - alpha) * uv.v[0].x + alpha * uv.v[1].x);
            xRight = int((1 - alpha) * uv.v[0].x + alpha * uv.v[2].x);
        }

        // Iterate through each pixel between xLeft and xRight at this y level
        for (int x = std::min(xLeft, xRight); x <= std::max(xLeft, xRight); ++x) {
            // Calculate barycentric coordinates for the current pixel
            if(idx == 15 && false){
                Color white = {255, 255, 255};
                framebuffer.putPixel({x,y}, white);
                break;
            }

            framebuffer.putPixel({x,y}, {255, 255, 0});
            break;

            float3 worldCoord = getWorldCoordinates({x,y}, uv, mesh.triangles[idx]);
            //getLight(worldCoord, )

            float3 light = {0,0,0};
            for(int i = 0; i < ray_count; i++){
                float2 seed = {randFloat(), randFloat()};
                float3 dir = hemisphereSampling(mesh.normals[idx], seed);
                Ray ray = {worldCoord + 1e-6*dir, dir};

                light = light + pathtrace(ray, mesh, framebuffer);
            }
            light = light/static_cast<float>(ray_count);
            light *= 255;
            //Color pixel = gamma_correction(light);
            Color c = {light.x, light.y, light.z};
            framebuffer.putPixel({x,y}, {255, 255, 255});
        }
    }
}*/

float3 pathtrace(Ray& ray, Mesh& mesh, Framebuffer& f){
    int argMin = NOT_FOUND;
    float distMin = INF;
    for(int i = 0; i < mesh.size; i++){
        float dist = rayTriangleDist(ray, mesh.triangles[i]);
        if(dist < distMin){
            argMin = i;
            distMin = dist;
        }
    }
    
    if(argMin != NOT_FOUND){
        if(argMin == 15) // Just to test
            return {1, 1, 1};
    }
    return {0,0,0};
}