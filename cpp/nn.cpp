#include "nn.hpp"
#include "mcts_constants.hpp"

NN::NN(const std::string& _module_path)
    : is_running(true),
      num_threads(torch::cuda::device_count()),
      module_path(_module_path),
      threads(num_threads + 1),
      mtx_nns(num_threads),
      cv_nns(num_threads),
      waits(num_threads),
      works(num_threads)
{
    for (unsigned i = 0; i < num_threads; i++)
        to_serve.push_back(i);
    for (unsigned i = 0; i < num_threads; i++) {
        waits[i].reserve(mc::maximum_batch);
        works[i].reserve(mc::maximum_batch);
    }
    for (unsigned i = 0; i < num_threads; i++)
        threads[i] = std::thread(&NN::gpu_worker, this, i);
    threads[num_threads] = std::thread(&NN::main_worker, this);
}

NN::~NN()
{
    /* notify without lock may cause dead lock
     * thd2: while (is_running ...)
     * thd1: is_running = false
     * thd1: notify
     * th2: cv.wait(lck)
     * deadlock
     * TODO: make is_running atomic (necessary???)
     */
    is_running = false;
    { std::scoped_lock lck(mtx_main); cv_main.notify_one(); }
    { std::scoped_lock lck(mtx_heap); cv_heap.notify_one(); }
    for (unsigned i = 0; i < num_threads; i++) {
        std::scoped_lock lck(mtx_nns[i]);
        cv_nns[i].notify_one();
    }
    for (auto& thd : threads) thd.join();
}

std::future<std::pair<torch::Tensor, torch::Tensor>> NN::compute(
    unsigned char* buffer, unsigned tot, unsigned pri)
{
    std::promise<std::pair<torch::Tensor, torch::Tensor>> pms;
    auto fut = pms.get_future();
    {
        std::scoped_lock lck(mtx_heap);
        heap.emplace(std::make_tuple(buffer, tot, std::move(pms)), pri);
        cv_heap.notify_one();
    }
    return fut;
}

void NN::gpu_worker(unsigned gpu)
{
    torch::NoGradGuard no_grad_guard;
    torch::Device device = torch::Device(torch::kCUDA, gpu);
    auto module = torch::jit::load(module_path, device);
    module.eval();

    auto& mtx_nn = mtx_nns[gpu];
    auto& cv_nn = cv_nns[gpu];
    auto& wait = waits[gpu];
    auto& work = works[gpu];

    auto tensor = torch::empty({mc::maximum_batch, mc::n, mc::m},
                               torch::TensorOptions().dtype(torch::kInt64)
                                                     .pinned_memory(true));
    for ( ; ; ) {
        {
            std::unique_lock lck(mtx_nn);
            while (is_running && wait.empty())
                cv_nn.wait(lck);
            if (!is_running) return;
            wait.swap(work);
        }
        {
            std::scoped_lock lck(mtx_main);
            to_serve.push_back(gpu);
            cv_main.notify_one();
        }

        size_t batch_size = 0;
        for (const auto& e : work)
            batch_size += std::get<1>(e);

        int64_t* head = tensor.data<int64_t>();
        for (const auto& e : work)
            head = std::copy(std::get<0>(e), std::get<0>(e) + mc::nm * std::get<1>(e), head);

        auto outs = module.forward({tensor.narrow(0, 0, batch_size).to(device)})
                          .toTuple()->elements();
            
        auto prob = outs[0].toTensor().to(torch::kCPU);
        auto value = outs[1].toTensor().to(torch::kCPU);
        size_t prev = 0;
        for (auto& e : work) {
            std::get<2>(e).set_value({prob.narrow(0, prev, std::get<1>(e)),
                                      value.narrow(0, prev, std::get<1>(e))});
            prev += std::get<1>(e);
        }
        work.clear();
    }
}

void NN::main_worker()
{
    unsigned gpu;
    for ( ; ; ) {
        {
            std::unique_lock lck(mtx_main);
            while (is_running && to_serve.empty())
                cv_main.wait(lck);
            if (!is_running) return;
            gpu = to_serve.back();
            to_serve.pop_back();
        }
        {
            std::unique_lock lck(mtx_heap);
            while (is_running && heap.empty())
                cv_heap.wait(lck);
            if (!is_running) return;
            unsigned n = (heap.size() - 1) / num_threads + 1;
            unsigned batch_size = 0;
            {
                std::scoped_lock lck(mtx_nns[gpu]);
                for (unsigned i = 0; i < n; i++) {
                    batch_size += std::get<1>(heap.top().first);
                    if (batch_size > mc::maximum_batch) break;
                    waits[gpu].push_back(std::move(const_cast<qtype&>(heap.top()).first));
                    heap.pop();
                }
                cv_nns[gpu].notify_one();
            }
        }
    }
}
