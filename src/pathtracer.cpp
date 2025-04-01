#include "pathtracer.h"

#include <algorithm>
#include <cuda_runtime.h>

#include "primitives.h"
#include "operators.h"

#include <cmath>

#include <iostream>

bool rayTriangleIntersect(Ray& ray, Triangle& t, float3& barCoord, float& d) {
    float3 edge1 = t.v[1]-t.v[0];
    float3 edge2 = t.v[2]-t.v[0];

    float3 h = cross(ray.direction, edge2);
    float a = dot(edge1, h);

    if (std::abs(a) < 1e-6f)
        return false;

    float f = 1.0f / a;
    float3 s = ray.origin - t.v[0];
    float u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f)
        return false;

    float3 q = cross(s, edge1);
    float v = f * dot(ray.direction, q);

    if (v < 0.0f || u + v > 1.0f)
        return false;

    float t_value = f * dot(edge2, q);

    if (t_value > 1e-6f) {
        barCoord = make_float3(u, v, 1.0f - u - v);
        d = t_value;
        return true;
    }

    return false;
}

float3 sphereSampling(float3 seed) {
    // Compute radius with cube root scaling for uniform distribution
    double r = std::cbrt(seed.x);

    // Compute spherical coordinates
    double theta = 2.0 * 3.1415926535 * seed.y;  // Azimuthal angle (0 to 2π)
    double phi = std::acos(2.0 * seed.z - 1.0);  // Polar angle (0 to π)

    // Convert to Cartesian coordinates
    double x = r * std::sin(phi) * std::cos(theta);
    double y = r * std::sin(phi) * std::sin(theta);
    double z = r * std::cos(phi);

    return {x, y, z};
}

int vCount = 67;

void pathtrace(Ray& ray, Mesh& mesh, Framebuffer& f){
    float minD = 1e9;
    float3 minBarCoord;
    int minIdx = -1;
    for(int i = 0; i < mesh.size; i++){
        float3 barCoord;
        Triangle t = mesh.triangles[i];
        float d;
        if(rayTriangleIntersect(ray, t, barCoord, d)){
            if(d < minD){
                minD = d;
                minBarCoord = barCoord;
                minIdx = i;
            }

        }
    }
    if(minIdx != -1){
        Triangle t = mesh.triangles[minIdx];
        float2 uv = minBarCoord.x * t.uv[1] + minBarCoord.y * t.uv[2] + minBarCoord.z * t.uv[0];
        int2 px = {uv.x*f.size.x, uv.y*f.size.y};
        //srand(minIdx);
        //Pixel p = {rand(), rand(), rand()};
        float3 wPoint = ray.origin + minD * ray.direction;

        /*std::cout << "v " << ray.origin.x << " " << ray.origin.y << " " << ray.origin.z << std::endl;
        std::cout << "v " << wPoint.x << " " << wPoint.y << " " << wPoint.z << std::endl;
        std::cout << "l " << vCount << " " << vCount+1 << std::endl;*/
        vCount += 2;

        //wPoint = ray.direction;
        wPoint = 120.0f*(wPoint+2.0f);
        minD = 150.0f*(minD+2.0f);
        Pixel p = {255, 255, 255};
        //p = {minD, minD, minD};
        p = {wPoint.x, wPoint.y, wPoint.z};
        putPixel(f, px, p);
    }
}

float3 getBarCoord(Triangle2D& t, int2 p){
    float areaTotal = (t.v[1].y - t.v[2].y) * (t.v[0].x - t.v[2].x) + (t.v[2].x - t.v[1].x) * (t.v[0].y - t.v[2].y);
    float area1 = (t.v[1].y - t.v[2].y) * (p.x - t.v[2].x) + (t.v[2].x - t.v[1].x) * (p.y - t.v[2].y);
    float area2 = (t.v[2].y - t.v[0].y) * (p.x - t.v[2].x) + (t.v[0].x - t.v[2].x) * (p.y - t.v[2].y);
    float area3 = areaTotal - area1 - area2;

    return {
        area1 / areaTotal,
        area2 / areaTotal,
        area3 / areaTotal
    };
}
