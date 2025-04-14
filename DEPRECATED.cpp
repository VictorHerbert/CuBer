float3 getBarCoord(Triangle2Di& t, int2 p){
    float areaTotal = (t.v[1].y - t.v[2].y) * (t.v[0].x - t.v[2].x) + (t.v[2].x - t.v[1].x) * (t.v[0].y - t.v[2].y);
    float area1 = (t.v[1].y - t.v[2].y) * (p.x - t.v[2].x) + (t.v[2].x - t.v[1].x) * (p.y - t.v[2].y);
    float area2 = (t.v[2].y - t.v[0].y) * (p.x - t.v[2].x) + (t.v[0].x - t.v[2].x) * (p.y - t.v[2].y);
    float area3 = areaTotal - area1 - area2;

    return {area1 / areaTotal, area2 / areaTotal, area3 / areaTotal};
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


uint2 projectUV(float2 coord, uint2 size){
    size.x = size.x*coord.x;
    size.y = size.y*coord.y;

    return size;
}

UVProj projectUV(UV& uv, uint2 size){
    UVProj UVProj;
    UVProj.v[0] = projectUV(uv.v[0], size);
    UVProj.v[1] = projectUV(uv.v[1], size);
    UVProj.v[2] = projectUV(uv.v[2], size);
    return UVProj;
}



void Framebuffer::drawTriangle(Triangle2Di& t, Color& p){

}
