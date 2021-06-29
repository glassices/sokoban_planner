#pragma once

#include <torch/extension.h>
#include <torch/script.h>

#include <tuple>
#include <queue>
#include <vector>
#include <string>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>

class NN
{
public:
    using dtype = std::tuple<unsigned char*, unsigned, std::promise<std::pair<torch::Tensor, torch::Tensor>>>;
    /* <<buffer, tot, pms>, prio> top is the smallest */
    using qtype = std::pair<dtype, unsigned>;

    struct cmp
    {
        bool operator()(const qtype& lhs, const qtype& rhs) const
        {
            return lhs.second > rhs.second;
        }
    };

    using htype = std::priority_queue<qtype, std::vector<qtype>, cmp>;

    NN(const std::string&);
    ~NN();

    NN(const NN&) = delete;
    void operator=(const NN&) = delete;

    std::future<std::pair<torch::Tensor, torch::Tensor>> compute(
        unsigned char* buffer, unsigned tot, unsigned pri);

private:
    void gpu_worker(unsigned);
    void main_worker();

    bool is_running;
    unsigned num_threads;
    std::string module_path;
    std::vector<std::thread> threads;

    std::mutex mtx_heap;
    std::condition_variable cv_heap;
    htype heap;

    std::mutex mtx_main;
    std::condition_variable cv_main;
    std::vector<unsigned> to_serve;

    std::vector<std::mutex> mtx_nns;
    std::vector<std::condition_variable> cv_nns;
    std::vector<std::vector<dtype>> waits, works;
};
