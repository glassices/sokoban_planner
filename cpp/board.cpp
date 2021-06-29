#include <vector>
#include <cmath>
#include <limits>
#include <iostream>

#include "board.hpp"
#include "map.hpp"

void push_box(unsigned char state[], unsigned mv)
{
    unsigned que[mc::nm];
    auto [cnt_box, nxt_box, cnt_player] = Map::get_instance().get_valid_pushes()[mv];
    assert(state[cnt_box] & 2);
    assert(!(state[nxt_box] & 2));
    state[cnt_box] &= 1;
    state[nxt_box] |= 2;
    unsigned lo = 0, hi = 1;
    state[que[0] = cnt_box] |= 4;
    while (lo < hi) {
        auto p = que[lo++];
        for (auto np : Map::get_instance().get_valid_moves()[p])
            if (state[np] <= 1) {
                state[np] |= 4;
                que[hi++] = np;
            }
    }
}

void Board::initialize(MCT_Helper& mct_helper)
{
    std::vector<unsigned> mvs = Map::get_instance().get_valids(state);
    nvalids = mvs.size();
    ns = 0;
    valids = new unsigned[nvalids];
    vas = new float[nvalids];
    nas = new unsigned[nvalids];
    pred_p = new float[nvalids];
    succ = new Board*[nvalids];
    memcpy(valids, mvs.data(), sizeof(unsigned) * nvalids);
    memset(nas, 0, sizeof(unsigned) * nvalids);
    std::fill_n(succ, nvalids, nullptr);

    /* calculate pred_p, vas and epuct */
    std::vector<unsigned> todo;
    auto ret = mct_helper.que_map.query(state);
    if (ret) {
        float sum = 0.0;
        for (unsigned i = 0; i < nvalids; i++)
            sum += ret->second[valids[i]];
        float mul = 1.0;
        for (unsigned i = 0; i < nvalids; i++) {
            pred_p[i] = ret->second[valids[i]] / sum;
            mul *= std::pow(pred_p[i], pred_p[i]);
        }
        epuct = mc::cpuct / mul;
    }
    else {
        mct_helper.num_hash_miss++;
        memcpy(mct_helper.buffer, state, sizeof(unsigned char) * mc::nm);
        todo.push_back(nvalids);
    }
    for (unsigned k = 0; k < nvalids; k++) {
        auto head = mct_helper.buffer + mc::nm * todo.size();
        memcpy(head, state, sizeof(unsigned char) * mc::nm);
        for (unsigned i = 0; i < mc::nm; i++)
            head[i] &= 3;
        push_box(head, valids[k]);
        auto end_state = Map::get_instance().end_state(head);
        if (end_state == 1) vas[k] = -1.0;
        else if (end_state == -1) vas[k] = -2.0;
        else {
            auto ret = mct_helper.que_map.query(head);
            if (ret) vas[k] = ret->first, mct_helper.num_hash_hit++;
            else todo.push_back(k), mct_helper.num_hash_miss++;
        }
    }
    if (!todo.empty()) {
        auto fut = mct_helper.nn.compute(mct_helper.buffer, todo.size(), mct_helper.pri);
        auto [prob, value] = fut.get();
        assert(prob.size(0) == todo.size() &&
               prob.size(1) == Map::get_instance().get_num_pushes() &&
               value.size(0) == todo.size());

        auto acs_prob = prob.accessor<float, 2>();
        auto acs_value = value.accessor<float, 1>();

        for (unsigned k = 0; k < todo.size(); k++) {
            if (todo[k] == nvalids) {
                float sum = 0.0;
                for (unsigned i = 0; i < nvalids; i++)
                    sum += acs_prob[k][valids[i]];
                float mul = 1.0;
                for (unsigned i = 0; i < nvalids; i++) {
                    pred_p[i] = acs_prob[k][valids[i]] / sum;
                    mul *= std::pow(pred_p[i], pred_p[i]);
                 }
                epuct = mc::cpuct / mul;
            }
            else vas[todo[k]] = acs_value[k];
            mct_helper.que_map.insert(mct_helper.buffer + mc::nm * k,
                                      acs_value[k],
                                      prob.select(0, k));
        }
    }
}

void Board::destroy(MemoryPoolType& memory_pool)
{
    delete[] valids;
    delete[] vas;
    delete[] nas;
    delete[] pred_p;
    for (unsigned i = 0; i < nvalids; i++)
        if (succ[i])
            memory_pool.free(succ[i]);
    delete[] succ;
}

void Board::simulate(unsigned dep, MCT_Helper& mct_helper)
{
    std::pair<Board*, unsigned> stacks[mc::num_mcts_sims + 10];
    while (ns < mc::num_mcts_sims) {
        mct_helper.pri = dep * mc::num_mcts_sims + ns;
        unsigned tot = 0;
        auto board = this;
        float leaf_v;
        for ( ; ; ) {
            unsigned best;
            float best_val = std::numeric_limits<float>::lowest();
            float tmp = board->epuct * std::sqrt(board->ns + 1);
            assert(board->nvalids);
            for (unsigned i = 0; i < board->nvalids; i++) {
                float v = board->vas[i] == -2.0 ? 1.0 : (
                          board->vas[i] == -1.0 ? 0.0 : (
                          board->nas[i] == 0 ? board->vas[i] :
                          board->vas[i] / board->nas[i]));
                float cnt_val = tmp * board->pred_p[i] / (1 + board->nas[i]) - v;
                if (cnt_val > best_val) {
                    best_val = cnt_val;
                    best = i;
                }
            }
            assert(0 <= best && best < board->nvalids);
            board->ns++;
            board->nas[best]++;
            if (board->vas[best] == -2.0) { leaf_v = 1.0; break; }
            if (board->vas[best] == -1.0) { leaf_v = 0.0; break; }
            if (board->nas[best] == 1) { leaf_v = board->vas[best]; break; }

            stacks[tot++] = {board, best};
            if (board->nas[best] == 2) board->new_child(best, mct_helper);
            board = board->succ[best];
        }
        for (unsigned i = 0; i < tot; i++) {
            auto [board, k] = stacks[i];
            board->vas[k] += std::min(float(tot - i) / mc::max_steps + leaf_v, 1.0f);
        }
    }
}

void Board::new_child(unsigned k, MCT_Helper& mct_helper)
{
    assert(!succ[k]);
    Board* child = mct_helper.memory_pool.alloc();
    memcpy(child->state, state, sizeof(unsigned char) * mc::nm);
    for (unsigned i = 0; i < mc::nm; i++)
        child->state[i] &= 3;
    push_box(child->state, valids[k]);
    child->initialize(mct_helper);
    succ[k] = child;
}

std::string Board::to_string() const
{
    std::string res;
    for (unsigned i = 0; i < mc::nm; i++) {
        if (!Map::get_instance().get_mask()[i]) res.push_back('#');
        else if (state[i] == 0) res.push_back(' ');
        else if (state[i] == 1) res.push_back('.');
        else if (state[i] == 2) res.push_back('$');
        else if (state[i] == 3) res.push_back('*');
        else if (state[i] == 4) res.push_back('@');
        else res.push_back('+');
        if ((i + 1) % mc::m == 0) res.push_back('\n');
    }
    return res;
}

MCT_Helper::MCT_Helper(NN& _nn)
    : pri(0),
      nn(_nn), que_map(mc::history_size),
      num_hash_hit(0), num_hash_miss(0)
{
    buffer = new unsigned char[(mc::tot_boxes * 4 + 1) * mc::nm];
}

MCT_Helper::~MCT_Helper()
{
    delete[] buffer;
}
