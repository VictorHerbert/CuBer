#include "pathtracer.h"

#include <algorithm>
#include <cuda_runtime.h>

#include "primitives.h"
#include "operators.h"


bool rayTriangleIntersect(Ray& ray, Triangle& t, float3& barCoord) {
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
        return true;
    }

    return false;
}

void pathtrace(Ray& ray, Mesh& mesh, Framebuffer& f){
    for(int i = 0; i < mesh.size; i++){
        float3 barCoord;
        Triangle t = mesh.triangles[i];
        if(rayTriangleIntersect(ray, t, barCoord)){
            float2 uv = barCoord.x * t.uv[0] + barCoord.y * t.uv[1] + barCoord.z * t.uv[2];
            int2 px = {uv.x*f.size.x, uv.y*f.size.y};
            srand(i);
            Pixel p = {255, 255, 255};
            putPixel(f, px, p);
        }        
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
