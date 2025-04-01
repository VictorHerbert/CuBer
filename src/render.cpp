#include "render.h"

#include <cuda_runtime.h>

#include "primitives.h"


void drawLine(Framebuffer& f, int2 src, int2 dest, Pixel& col) {
    int x0 = src.x, y0 = src.y;
    int x1 = dest.x, y1 = dest.y;
    
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    while (true) {
        putPixel(f, {x0, y0}, col);
        
        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
    }
}

bool isInsideTriangle(int x, int y, const Triangle2D& t) {
    // Barycentric coordinates or edge function can be used to determine if the point is inside the triangle.
    int x0 = t.v[0].x, y0 = t.v[0].y;
    int x1 = t.v[1].x, y1 = t.v[1].y;
    int x2 = t.v[2].x, y2 = t.v[2].y;

    // Compute the area of the whole triangle
    int area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);

    // Compute sub-area of the triangle formed by point (x, y) and two triangle vertices
    int area1 = (x - x0) * (y1 - y0) - (x1 - x0) * (y - y0);
    int area2 = (x - x1) * (y2 - y1) - (x2 - x1) * (y - y1);
    int area3 = (x - x2) * (y0 - y2) - (x0 - x2) * (y - y2);

    // Check if the point is inside the triangle by comparing areas (signs of areas should be the same)
    return (area >= 0 && area1 >= 0 && area2 >= 0 && area3 >= 0) ||
           (area < 0 && area1 < 0 && area2 < 0 && area3 < 0);
}

void wireframeTriangle(Triangle2D& t, Framebuffer& f, Pixel& col) {
    // Draw the edges of the triangle using drawLine
    drawLine(f, t.v[0], t.v[1], col);
    drawLine(f, t.v[1], t.v[2], col);
    drawLine(f, t.v[2], t.v[0], col);
}

void drawTriangle(Triangle2D& t, Framebuffer& f){

    Pixel tcol = {rand(), rand(), rand()};

    // Sort vertices by Y-coordinate (bottom to top)
    if (t.v[0].y > t.v[1].y) std::swap(t.v[0], t.v[1]);
    if (t.v[1].y > t.v[2].y) std::swap(t.v[1], t.v[2]);
    if (t.v[0].y > t.v[1].y) std::swap(t.v[0], t.v[1]);

    // Step through the triangle row by row (scanline)
    for (int y = t.v[0].y; y <= t.v[2].y; ++y) {
        // Find the left and right x boundaries for the current scanline
        int xLeft = t.v[0].x, xRight = t.v[0].x;

        if (y >= t.v[1].y) {
            // Second half of triangle: interpolate between v[1] and v[2]
            float alpha = float(y - t.v[1].y) / float(t.v[2].y - t.v[1].y);
            xLeft = int((1 - alpha) * t.v[0].x + alpha * t.v[1].x);
            xRight = int((1 - alpha) * t.v[0].x + alpha * t.v[2].x);
        } else {
            // First half of triangle: interpolate between v[0] and v[1]
            float alpha = float(y - t.v[0].y) / float(t.v[1].y - t.v[0].y);
            xLeft = int((1 - alpha) * t.v[0].x + alpha * t.v[1].x);
            xRight = int((1 - alpha) * t.v[0].x + alpha * t.v[2].x);
        }

        // Iterate through each pixel between xLeft and xRight at this y level
        for (int x = std::min(xLeft, xRight); x <= std::max(xLeft, xRight); ++x) {
            // Calculate barycentric coordinates for the current pixel
            float areaTotal = (t.v[1].y - t.v[2].y) * (t.v[0].x - t.v[2].x) + (t.v[2].x - t.v[1].x) * (t.v[0].y - t.v[2].y);
            float area1 = (t.v[1].y - t.v[2].y) * (x - t.v[2].x) + (t.v[2].x - t.v[1].x) * (y - t.v[2].y);
            float area2 = (t.v[2].y - t.v[0].y) * (x - t.v[2].x) + (t.v[0].x - t.v[2].x) * (y - t.v[2].y);
            float area3 = areaTotal - area1 - area2;

            float lambda0 = area1 / areaTotal;
            float lambda1 = area2 / areaTotal;
            float lambda2 = area3 / areaTotal;

            // Interpolate using barycentric coordinates
            //float2 uv = lambda0 * t.uv[0] + lambda1 * t.uv[1] + lambda2 * t.uv[2];

            putPixel(f, {x, y}, tcol);

        }
    }
}