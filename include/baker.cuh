#ifndef BAKER_H
#define BAKER_H

#include "mesh.cuh"
#include "framebuffer.cuh"

void computeLightmap(CpuMesh& mesh, Framebuffer& f);

#endif