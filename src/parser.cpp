#include <fstream>
#include <sstream>
#include <string>
#include <regex>

#include <cuda_runtime.h>

#include "parser.h"
#include "primitives.h"

#include <iostream>

std::vector<Triangle> parse_obj(std::string filename){
    std::ifstream file;
    file.open(filename);

    std::vector<Triangle> tri;

    if (!file)
        return tri;
    
    std::vector<float3> vertexes;
    std::vector<float2> uv;
    std::string cmd;
    //float3 currPoint;

    std::string line;

    while (std::getline(file, line)){
        std::replace(line.begin(), line.end(), '/', ' ');
        std::istringstream iss(line);
        iss >> cmd;
        
        if(cmd == "#")
            continue;
        else if(cmd == "mtllib")
            continue;
        else if(cmd == "o")
            continue;
        else if(cmd == "v"){
            //vertexes.push_back
            vertexes.push_back(float3());
            iss >>
                (vertexes.end()-1)->x >>
                (vertexes.end()-1)->y >>
                (vertexes.end()-1)->z;
        }
        else if(cmd == "vn")
            continue;
        else if(cmd == "vt"){
            uv.push_back(float2());
            iss >>
                (uv.end()-1)->x >>
                (uv.end()-1)->y;
        }
        else if(cmd == "s")
            continue;
        else if(cmd == "usemtl")
            continue;
        else if(cmd == "f"){
            Triangle t;

            for(int i = 0; i < 3; i++){
                int v, vt, vn;
                iss >> v >> vt >> vn;
                t.v[i] = vertexes[v-1];
                t.uv[i] = uv[vt-1];
            }
            tri.push_back(t);

        }
        else{
            std::cout << "Unrecognized instruction:" << cmd << std::endl;
            break;
        }
    }
    

    file.close();
    return tri;
}