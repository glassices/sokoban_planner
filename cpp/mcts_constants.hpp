#pragma once

#include <string>
#include <cmath>

namespace mc
{
    const std::string map_path = "/home/fs01/df394/data/levels/sasquatch/sasquatch_29.txt";
    const unsigned n = 14;
    const unsigned m = 20;
    const unsigned nm = n * m;
    const unsigned tot_boxes = 18;
    const unsigned max_steps = 300;
    const unsigned num_episodes_per_gpu = 100;
    const unsigned history_size = 20000;
    const unsigned maximum_batch = 1000;

    /*
    const std::string map_path = "maps/xsokoban_29";
    const unsigned n = 11;
    const unsigned m = 17;
    const unsigned nm = n * m;
    const unsigned tot_boxes = 16;
    const unsigned max_steps = 250;
    const unsigned num_episodes_per_gpu = 100;
    const unsigned history_size = 20000;
    const unsigned maximum_batch = 1000;
    */

    /*
    const std::string map_path = "maps/sasquatch7_48.txt";
    const unsigned n = 21;
    const unsigned m = 21;
    const unsigned nm = n * m;
    const unsigned tot_boxes = 64;
    const unsigned max_steps = 1000;
    const unsigned num_episodes_per_gpu = 100;
    const unsigned history_size = 20000;
    const unsigned maximum_batch = 2000;
    */

    /*
    const std::string map_path = "maps/sasquatch7_50.txt";
    const unsigned n = 39;
    const unsigned m = 39;
    const unsigned nm = n * m;
    const unsigned tot_boxes = 256;
    const unsigned max_steps = 500;
    const unsigned num_episodes_per_gpu = 100;
    const unsigned history_size = 20000;
    const unsigned maximum_batch = 2000;
    */

    /*
    const std::string map_path = "maps/sven_1513.txt";
    const unsigned n = 15;
    const unsigned m = 17;
    const unsigned nm = n * m;
    const unsigned tot_boxes = 54;
    const unsigned max_steps = 2000;
    const unsigned num_episodes_per_gpu = 100;
    const unsigned history_size = 20000;
    const unsigned maximum_batch = 2000;
    */
    
    /*
    const std::string map_path = "maps/parking";
    const unsigned n = 8;
    const unsigned m = 14;
    const unsigned nm = n * m;
    const unsigned tot_boxes = 5;
    const unsigned max_steps = 200;
    const unsigned num_episodes_per_gpu = 100;
    const unsigned history_size = 20000;
    const unsigned maximum_batch = 1000;
    */

    const unsigned num_mcts_sims = 1600;
    const unsigned num_explore_moves = 30;
    /* even p1 = 0.0, p2 = 1.0, but v1 has one step short than v2,
     * this cpuct can prefer v1 after num_mcts_sims / 2 simulations
     */
    const float cpuct = 8.0 / max_steps;

}

