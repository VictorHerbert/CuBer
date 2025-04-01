#include "test.h"
#include "pathtracer.h"

#include "operators.h"
#include "parser.h"
#include "render.h"

#include <time.h>

float rand1(){
    return static_cast<float>(rand())/RAND_MAX;
}

float3 rand3(){
    return {rand1(), rand1(), rand1()};
}

T_FUNC(test_pathtracer){
    auto triangles  = parse_obj("sample/cornell.obj");
    std::vector<Pixel> fdata(512*512);
    std::vector<float> fzbuffer(512*512, 1e9);

    Mesh mesh = {triangles.size(), triangles.data()};
    Framebuffer framebuffer = {512, 512, fdata.data(), fzbuffer.data()};
    Ray ray = {{0.471, 0.791, 0.519}, {1,0,0}}; //Should hit triangle 10

    for(int i = 0; i < mesh.size; i++){
        Triangle2D tri;
        for(int j = 0; j < 3; j++){
            float2 v = mesh.triangles[i].uv[j] * framebuffer.size;
            tri.v[j] =  {static_cast<int>(v.x), static_cast<int>(v.y)};
        }
        srand(i);
        Pixel p = {rand(), rand(), rand()};
        p = {255, 255, 255};

        //wireframeTriangle(tri, framebuffer, p);

        /*float step = 1/1e2;
        for(float u = 0; u < 1; u+= step)
            for(float v = 0; v < 1; v+= step){
                if(u+v > 1) break;
                float t = 1 - u - v;

                int2 pt = tri.applyUV({u,v});
                float3 wpt = u*mesh.triangles[i].v[0] + v*mesh.triangles[i].v[1] + t*mesh.triangles[i].v[2];
                wpt = 120.0f*(wpt + 2.0f);
                p = {wpt.x, wpt.y, wpt.z};
                putPixel(framebuffer, pt, p);
            }*/

    }    

    ray.direction = sphereSampling({0.471, 0.791, 0.519});
    pathtrace(ray, mesh, framebuffer);

    float step = 1/1e2;

    for(float i = 0; i < 1; i+= step)
        for(float j = 0; j < 1; j+= step)
            for(float k = 0; k < 1; k+= step){
                ray.direction = sphereSampling({i,j,k});
                pathtrace(ray, mesh, framebuffer);
            }


    framebuffer.save("test/output/path.png");
}