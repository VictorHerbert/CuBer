#include "test.h"

#include "primitives.h"
#include "parser.h"

T_FUNC_OFF(test_parser){
    auto mesh  = parse_obj("wrong_path.obj");
    T_ASSERT(mesh.size(), 0);

    mesh  = parse_obj("sample/cornell.obj");
    T_ASSERT(mesh.size(), 32);
}