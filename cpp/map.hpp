#pragma once

#include <vector>
#include <tuple>

/*
 * state (total 6 different value):
 *   000: Floor ( )
 *   001: Goal square (.)
 *   010: Box ($)
 *   011: Box on goal square (*)
 *   100: Player (@)
 *   101: Player on goal square (+)
 */

class Map
{
public:
    static Map& get_instance();
    Map(const Map&) = delete;
    void operator=(const Map&) = delete;

    const std::vector<std::tuple<unsigned, unsigned, unsigned>>& get_valid_pushes() const;
    const std::vector<std::vector<unsigned>>& get_valid_moves() const;
    unsigned get_num_pushes() const;
    const std::vector<unsigned>& get_mask() const;
    void get_init_state(const std::vector<bool>& sub_boxes,
                        const std::vector<bool>& sub_goals,
                        unsigned char state[]) const;
    /* 1: goal, 0: unknown, -1: deadend */
    int end_state(unsigned char state[]) const;
    std::vector<unsigned> get_valids(unsigned char state[]) const;

private:
    Map();

    unsigned player;
    std::vector<unsigned> mask;
    std::vector<unsigned> boxes;
    std::vector<unsigned> goals;
    std::vector<std::vector<unsigned>> valid_moves;
    std::vector<std::tuple<unsigned, unsigned, unsigned>> valid_pushes;
};
