#include "pathtracer.h"

#include <algorithm>
#include <cuda_runtime.h>

#include "primitives.h"
#include "operators.h"

void pathtraceTriangle(Triangle& t, Framebuffer& f) {    
    // Set random color for this triangle (you can later replace this with a texture lookup)
    Pixel tcol = {rand() % 256, rand() % 256, rand() % 256};  // Random RGB color for now

    // Convert UVs to framebuffer coordinates
    Triangle2D uv;
    for (int i = 0; i < 3; i++) {
        uv.v[i].x = static_cast<int>(t.uv[i].x * f.width);
        uv.v[i].y = static_cast<int>(t.uv[i].y * f.height);
    }

    // Sort vertices by Y-coordinate (bottom to top)
    if (uv.v[0].y > uv.v[1].y) std::swap(uv.v[0], uv.v[1]);
    if (uv.v[1].y > uv.v[2].y) std::swap(uv.v[1], uv.v[2]);
    if (uv.v[0].y > uv.v[1].y) std::swap(uv.v[0], uv.v[1]);

    // Step through the triangle row by row (scanline)
    for (int y = uv.v[0].y; y <= uv.v[2].y; ++y) {
        int xLeft, xRight;

        // Determine left and right boundaries of the current scanline
        if (y >= uv.v[1].y) {
            // Second half of the triangle: Interpolate between v[1] and v[2] for the left edge,
            // Interpolate between v[0] and v[2] for the right edge
            float alpha = float(y - uv.v[1].y) / float(uv.v[2].y - uv.v[1].y);
            xLeft = int((1 - alpha) * uv.v[1].x + alpha * uv.v[2].x);
            xRight = int((1 - alpha) * uv.v[0].x + alpha * uv.v[2].x);
        } else {
            // First half of the triangle: Interpolate between v[0] and v[1] for the left edge,
            // Interpolate between v[0] and v[2] for the right edge
            float alpha = float(y - uv.v[0].y) / float(uv.v[1].y - uv.v[0].y);
            xLeft = int((1 - alpha) * uv.v[0].x + alpha * uv.v[1].x);
            xRight = int((1 - alpha) * uv.v[0].x + alpha * uv.v[2].x);
        }

        // Iterate through each pixel between xLeft and xRight at this y level
        for (int x = std::min(xLeft, xRight); x <= std::max(xLeft, xRight); ++x) {
            // Calculate barycentric coordinates for the current pixel
            float areaTotal = (uv.v[1].y - uv.v[2].y) * (uv.v[0].x - uv.v[2].x) + (uv.v[2].x - uv.v[1].x) * (uv.v[0].y - uv.v[2].y);
            float area1 = (uv.v[1].y - uv.v[2].y) * (x - uv.v[2].x) + (uv.v[2].x - uv.v[1].x) * (y - uv.v[2].y);
            float area2 = (uv.v[2].y - uv.v[0].y) * (x - uv.v[2].x) + (uv.v[0].x - uv.v[2].x) * (y - uv.v[2].y);
            float area3 = areaTotal - area1 - area2;

            float lambda0 = area1 / areaTotal;
            float lambda1 = area2 / areaTotal;
            float lambda2 = area3 / areaTotal;

            // Interpolate the UV coordinates based on barycentric coordinates
            //float2 interpolatedUV = lambda0 * t.uv[0] + lambda1 * t.uv[1] + lambda2 * t.uv[2];

            // Optional: You could use these interpolated UVs for texture mapping or just for color interpolation
            // For now, just assign a random color to the pixel (replace this with actual texture lookup if needed)
            putPixel(f, {x, y}, tcol);
        }
    }
}

