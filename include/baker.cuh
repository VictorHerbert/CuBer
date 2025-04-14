#ifndef BAKER_H
#define BAKER_H

#include "mesh.cuh"
#include "framebuffer.cuh"

void computeGI(CpuMesh& mesh, Framebuffer& f);

#endif