#pragma once

#include "mcts_constants.hpp"
#include "memory.hpp"
#include "nn.hpp"
#include "que_map.hpp"

class MCT_Helper;

struct Board
{
    using MemoryPoolType = MemoryPool<mc::num_mcts_sims + mc::max_steps + 10, Board>;
    /* initialize remaining fields after $state$ is given
     * buffer should be large enough to contain state information
     * of the current node and all its childrean
     */
    void initialize(MCT_Helper&);
    void destroy(MemoryPoolType&);
    void simulate(unsigned, MCT_Helper&);
    void new_child(unsigned k, MCT_Helper&);
    std::string to_string() const;

    /*
     * vas == -2 ==> unreachable
     * vas == -1 ==> goal
     * vas >= 0  ==> normal adding vas
     */
    float epuct;
    unsigned ns;
    unsigned nvalids;
    /* used for hash function of state */
    unsigned char state[mc::nm];
    unsigned* valids;
    float* vas;
    unsigned* nas;
    float* pred_p;
    Board** succ;
};

class MCT_Helper
{
public:
    MCT_Helper(NN&);
    ~MCT_Helper();

    /* priority */
    unsigned pri;
    unsigned char* buffer;
    NN& nn;
    Board::MemoryPoolType memory_pool;
    QueMap que_map;
    size_t num_hash_hit;
    size_t num_hash_miss;
};
