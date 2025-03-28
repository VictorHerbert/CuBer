#ifndef TEST_H
#define TEST_H

#include <string>
#include <vector>

using FuncPtr = void(*)();

namespace vtest{
    extern size_t test_count;
    extern size_t error_count;

    extern size_t success_count;
    extern size_t not_implemented_count;
    extern size_t fail_count;

    struct TestData{
        FuncPtr function;
        std::string name;
        size_t assert_count;
        size_t error_count;
    };

    extern std::vector<TestData> registered_functions;
    extern size_t current_test_idx;
}

/*
#define T_FUNC(func_name)  \
namespace vtest{ \
    void func_name##_register () __attribute__((constructor)); \
    void func_name(); \
    void func_name##_register(){ \
        vtest::registered_functions.push_back({ \
            .function = func_name, \
            .name = #func_name, \
            .assert_count = 0, \
            .error_count = 0, \
        }); \
    } \
}\
void vtest::func_name()
*/

#define T_FUNC(func_name) \
namespace vtest{ \
    void func_name##_dec(); \
    struct func_name##_fixture{ \
        func_name##_fixture(){ \
            vtest::registered_functions.push_back({ \
                .function = vtest::func_name##_dec, \
                .name = #func_name, \
                .assert_count = 0, \
                .error_count = 0, \
            }); \
        } \
    }; \
} \
static vtest::func_name##_fixture func_name##_fixture_inst; \
void vtest::func_name##_dec()

#define T_FUNC_OFF(func_name) \
namespace vtest{ \
    void func_name##_dec(); \
} \
void vtest::func_name##_dec()

#define T_ASSERT(value, expected) \
    vtest::registered_functions[vtest::current_test_idx].assert_count++; \
    if(value != expected) \
        vtest::registered_functions[vtest::current_test_idx].error_count++;
//std::cout << "ASSERT at line " << __LINE__ << ": expected " << expected << ", got " << value << std::endl;


#define T_DASSERT(value, expected) \
    vtest::registered_functions[vtest::current_test_idx].assert_count++; \
    if(value == expected) \
        vtest::registered_functions[vtest::current_test_idx].error_count++;

#define T_FAIL() \
    vtest::registered_functions[vtest::current_test_idx].assert_count++; \
    vtest::registered_functions[vtest::current_test_idx].error_count++;

#define T_SUCCESS() \
    vtest::registered_functions[vtest::current_test_idx].assert_count++;

#endif