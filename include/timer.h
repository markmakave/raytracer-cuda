#pragma once
#include <iostream>
#include <chrono>

namespace lm{

class Timer {

    std::string name;
    std::chrono::high_resolution_clock::time_point begin;

public:

    Timer(const std::string& name) : name(name) {
        begin = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto difference = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        std::cout << "Scope \"" << name << "\" took " << difference << " microseconds\n";
    }

};

}
