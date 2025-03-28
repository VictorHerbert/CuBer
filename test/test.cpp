#include <iostream>

#include "test.h"

namespace vtest{
    size_t test_count = 0;
    size_t error_count = 0;

    size_t success_count = 0;
    size_t not_implemented_count = 0;
    size_t fail_count = 0;
    
    std::vector<TestData> registered_functions;
    size_t current_test_idx;
}

const char RED[] = "\033[1;31m";
const char GREEN[] = "\033[92m";
const char YELLOW[] = "\033[93m";
const char RESET[] = "\033[0m";

int main(){
    std::cout << "Testing starting...\n" << std::endl;

    for(vtest::current_test_idx = 0; vtest::current_test_idx < vtest::registered_functions.size(); vtest::current_test_idx++){
        vtest::TestData *data = &vtest::registered_functions[vtest::current_test_idx];

        std::cout << data->name << ": ";
        data->function();
        if(data->error_count != 0){
            vtest::fail_count++;
            std::cout << RED << "FAILED\n" << RESET;
        }
        /*else if(data->assert_count == 0){
            vtest::not_implemented_count++;
            std::cout <<  YELLOW << "NOT_IMPLEMENTED\n" << RESET;
        }*/
        else{
            vtest::success_count++;
            std::cout << GREEN << "PASSED\n" << RESET;
        }
        vtest::test_count++;
    }

    std::cout << "\n" << vtest::success_count << "/" << vtest::test_count << " tests passed" << std::endl;

    return fail_count;
}