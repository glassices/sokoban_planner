#include <torch/extension.h>
#include <torch/script.h>

#include "mcts_constants.hpp"
#include "nn.hpp"
#include "board.hpp"
#include "random.hpp"
#include "map.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <ctime>
#include <chrono>
#include <cassert>
#include <set>
#include <sstream>
#include <ctime>

using namespace std;

/*
 * return the height, width, and number of valid moves
 * of the map, which is necessary when initializing a network
 */
tuple<unsigned, unsigned, unsigned, vector<unsigned>, unsigned> initialize()
{
    return {mc::n, mc::m, Map::get_instance().get_num_pushes(), Map::get_instance().get_mask(), mc::tot_boxes};
}

vector<bool> choose(unsigned n, unsigned m)
{
    vector<bool> ans;
    for (unsigned i = 0; i < n; i++) {
        if (Random::get_instance().get_int(0, n - i - 1) < m) {
            ans.push_back(true);
            m--;
        }
        else ans.push_back(false);
    }
    return ans;
}

mutex mtx_stats;
vector<unsigned> stats;

atomic<size_t> num_hash_hit;
atomic<size_t> num_hash_miss;

tuple<torch::Tensor, torch::Tensor, torch::Tensor> mcts(unsigned nbox, NN& nn)
{
    MCT_Helper mct_helper(nn);
    Board* board = mct_helper.memory_pool.alloc();

    Map::get_instance().get_init_state(choose(mc::tot_boxes, nbox),
                                       choose(mc::tot_boxes, nbox),
                                       board->state);
    auto end_state = Map::get_instance().end_state(board->state);
    if (end_state != 0) {
        mct_helper.memory_pool.free_without_destroy(board);
        return {torch::empty({0, mc::n, mc::m}, torch::kInt64),
                torch::empty({0, Map::get_instance().get_num_pushes()}),
                torch::empty({0})};
    }

    board->initialize(mct_helper);
    vector<Board*> boards;
    bool succeed = false;
    for (unsigned dep = 0; ; dep++) {
        board->simulate(dep, mct_helper);
        unsigned k;
        if (dep < mc::num_explore_moves) {
            auto rand_num = Random::get_instance().get_int(0, mc::num_mcts_sims - 1);
            for (unsigned i = 0; i < board->nvalids; i++)
                if (rand_num < board->nas[i]) {
                    k = i;
                    break;
                }
                else rand_num -= board->nas[i];
        }
        else {
            unsigned max_ns = 0;
            for (unsigned i = 0; i < board->nvalids; i++)
                if (board->nas[i] > max_ns) {
                    max_ns = board->nas[i];
                    k = i;
                }
        }
        assert(0 <= k && k < board->nvalids);
        boards.push_back(board);
        for (unsigned i = 0; i < board->nvalids; i++)
            if (i != k && board->succ[i]) {
                mct_helper.memory_pool.free(board->succ[i]);
                board->succ[i] = nullptr;
            }
        if (board->vas[k] == -2.0 || board->vas[k] == -1.0 || dep + 1 == mc::max_steps) {
            if (board->vas[k] == -1.0) {
                succeed = true;
                scoped_lock lck(mtx_stats);
                stats[nbox]++;
            }
            break;
        }
        if (!board->succ[k]) board->new_child(k, mct_helper);
        board = board->succ[k];
    }
    num_hash_hit += mct_helper.num_hash_hit;
    num_hash_miss += mct_helper.num_hash_miss;

    auto data = torch::empty({long(boards.size()), mc::n, mc::m}, torch::kInt64);
    auto prob = torch::zeros({long(boards.size()), Map::get_instance().get_num_pushes()});
    auto value = torch::empty({long(boards.size())});
    auto acs_prob = prob.accessor<float, 2>();
    auto acs_value = value.accessor<float, 1>();

    int64_t* data_ptr = data.data<int64_t>();
    for (unsigned k = 0; k < boards.size(); k++) {
        auto board = boards[k];
        data_ptr = copy(board->state, board->state + mc::nm, data_ptr);
        float best_val = 1.0f;
        unsigned max_ns = 0;
        for (unsigned i = 0; i < board->nvalids; i++) {
            acs_prob[k][board->valids[i]] = float(board->nas[i]) / mc::num_mcts_sims;
            if (board->vas[i] == -1.0) best_val = 0.0, max_ns = mc::num_mcts_sims;
            else if (board->vas[i] != -2.0 && board->nas[i] > max_ns)
                max_ns = board->nas[i], best_val = board->vas[i] / board->nas[i];
        }
        if (succeed)
            acs_value[k] = min(best_val + 1.0f / mc::max_steps, float(boards.size() - k) / mc::max_steps);
        else
            acs_value[k] = min(best_val + 1.0f / mc::max_steps, 1.0f);
    }
    mct_helper.memory_pool.free(boards[0]);
    return {data, prob, value};
}

tuple<torch::Tensor, torch::Tensor, torch::Tensor, float, string> generate_data(
    const string& module_path, unsigned lo, unsigned hi)
{
    lo = min(lo, mc::tot_boxes);
    hi = min(hi, mc::tot_boxes);
    stringstream ss;
    auto t1 = chrono::high_resolution_clock::now();

    unsigned num_episodes = mc::num_episodes_per_gpu * torch::cuda::device_count();
    num_episodes = num_episodes / (hi - lo + 1) * (hi - lo + 1);

    NN nn(module_path);

    vector<future<tuple<torch::Tensor, torch::Tensor, torch::Tensor>>> futs;
    futs.reserve(num_episodes);
    ss << "async mcts starts" << endl;
    stats.clear();
    for (unsigned i = 0; i < hi + 1; i++)
        stats.push_back(0);
    num_hash_hit = 0;
    num_hash_miss = 0;
    for (unsigned i = 0; i < num_episodes; i++)
        futs.push_back(async(launch::async, mcts, i % (hi - lo + 1) + lo, ref(nn)));

    vector<torch::Tensor> datas, probs, values;
    datas.reserve(num_episodes);
    probs.reserve(num_episodes);
    values.reserve(num_episodes);
    for (auto& fut : futs) {
        auto [data, prob, value] = fut.get();
        datas.push_back(move(data));
        probs.push_back(move(prob));
        values.push_back(move(value));
    }
    auto data = torch::cat(datas, 0);
    auto prob = torch::cat(probs, 0);
    auto value = torch::cat(values, 0);

    ss << "done mcts" << endl;
    auto t2 = chrono::high_resolution_clock::now();
    ss << "Execution time[mcts]: " << chrono::duration_cast<chrono::seconds>(t2 - t1).count() << 's' << endl;
    ss << "num_hash_hit: " << num_hash_hit << " num_hash_miss: " << num_hash_miss << endl;

    float acc = float(stats[hi]) / num_episodes * (hi - lo + 1);
    for (unsigned i = lo; i <= hi; i++) {
        ss << i << ": " << stats[i] << '/' << num_episodes / (hi - lo + 1) << '(' << float(stats[i]) / num_episodes * (hi - lo + 1) << ')';
        ss << (i == hi ? '\n' : ' ');
    }
    return {data, prob, value, acc, ss.str()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("initialize", &initialize, "initialize");
    m.def("generate_data", &generate_data, "generate_data");
}
