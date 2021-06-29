#pragma once

#include <vector>
#include <optional>

#include <torch/extension.h>

#include "mcts_constants.hpp"

class QueMap
{
private:
    const size_t nhash = 1000003;

    struct Record
    {
        size_t hvalue;
        Record *lque, *rque, *lmap, *rmap;
        unsigned char state[mc::nm];
        float value;
        float* prob;
    };

public:
    QueMap(size_t _maxn);
    ~QueMap();
    std::optional<std::pair<float, float*>> query(unsigned char[]);
    void insert(unsigned char[], float, torch::Tensor);

private:
    size_t cntn;
    size_t maxn;
    unsigned num_pushes;
    Record** h;
    Record* data1;
    float* data2;
    Record* lend_que;
    Record* rend_que;
};
