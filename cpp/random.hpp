#pragma once

#include <random>
#include <mutex>

class Random
{
public:
    static Random& get_instance();
    Random(const Random&) = delete;
    void operator=(const Random&) = delete;

    float get_gamma(float alpha, float beta);
    int get_int(int min, int max);

private:
    Random();

    std::mutex _mutex;
    std::mt19937 _gen;
};

