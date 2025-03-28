#include "test.h"
#include "pathtracer.h"

#include "operators.h"
#include "parser.h"

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
    Ray ray = {{0.471, 0.791, 0.519}, {1,0,0}}; //Should hit triangle 12


    for(int it = 0; it < 1e6; it++){
        srand(time(0));
        ray.direction = normalize(rand3());

        pathtrace(ray, mesh, framebuffer);
    }
            

    framebuffer.save("test/output/path.png");
}