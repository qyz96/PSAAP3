#include "Data.h"

extern "C" void TestSetValues(){
    gd.TestSetValues();
    printf("GlobalData in C++ memory is set.\n");
}

