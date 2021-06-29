#include <fstream>
#include <string>
#include <cassert>
#include <cstring>

#include "map.hpp"
#include "mcts_constants.hpp"

Map& Map::get_instance()
{
    static Map instance;
    return instance;
}

const std::vector<std::tuple<unsigned, unsigned, unsigned>>& Map::get_valid_pushes() const
{
    return valid_pushes;
}

const std::vector<std::vector<unsigned>>& Map::get_valid_moves() const
{
    return valid_moves;
}

unsigned Map::get_num_pushes() const
{
    return valid_pushes.size();
}

const std::vector<unsigned>& Map::get_mask() const
{
    return mask;
}

void Map::get_init_state(const std::vector<bool>& sub_boxes,
                         const std::vector<bool>& sub_goals,
                         unsigned char state[]) const
{
    /*
     * state (total 6 different value):
     *   000: Floor ( )
     *   001: Goal square (.)
     *   010: Box ($)
     *   011: Box on goal square (*)
     *   100: Player (@)
     *   101: Player on goal square (+)
     */
    assert(sub_boxes.size() == mc::tot_boxes && sub_goals.size() == mc::tot_boxes);
    memset(state, 0, sizeof(unsigned char) * mc::nm);
    for (unsigned i = 0; i < mc::tot_boxes; i++) {
        if (sub_boxes[i]) state[boxes[i]] |= 2;
        if (sub_goals[i]) state[goals[i]] |= 1;
    }

    unsigned lo = 0, hi = 1;
    unsigned que[mc::nm];
    state[que[0] = player] |= 4;

    while (lo < hi) {
        auto p = que[lo++];
        for (auto np : valid_moves[p])
            if (state[np] <= 1) {
                state[np] |= 4;
                que[hi++] = np;
            }
    }
}

int Map::end_state(unsigned char state[]) const
{
    bool is_goal = true;
    for (unsigned i = 0; i < mc::nm; i++)
        if (state[i] == 1 || state[i] == 2 || state[i] == 5) {
            is_goal = false;
            break;
        }
    if (is_goal) return 1;
    
    for (auto [cnt_box, nxt_box, cnt_player] : valid_pushes)
        if ((state[cnt_box] & 2) && !(state[nxt_box] & 2) && (state[cnt_player] & 4))
            return 0;
    return -1;
}

std::vector<unsigned> Map::get_valids(unsigned char state[]) const
{
    std::vector<unsigned> mvs;
    mvs.reserve(valid_pushes.size());
    for (unsigned i = 0; i < valid_pushes.size(); i++) {
        auto [cnt_box, nxt_box, cnt_player] = valid_pushes[i];
        if ((state[cnt_box] & 2) && !(state[nxt_box] & 2) && (state[cnt_player] & 4))
            mvs.push_back(i);
    }
    return mvs;
}

Map::Map() : valid_moves(mc::nm)
{
    std::ifstream fin(mc::map_path);
    std::string line;
    std::vector<std::string> raw_init_map;
    unsigned m = 0;
    while (getline(fin, line)) {
        while (!line.empty() && line.back() != '#') line.pop_back();
        if (!line.empty()) {
            m = std::max(m, unsigned(line.size()));
            raw_init_map.push_back(move(line));
        }
    }
    unsigned n = raw_init_map.size();

    bool* vis = new bool[n * m];
    unsigned* que = new unsigned[n * m];
    unsigned lo = 0, hi = 1;
    memset(vis, 0, sizeof(bool) * n * m);

    for (unsigned x = 0; x < n; x++)
        for (unsigned y = 0; y < raw_init_map[x].size(); y++)
            if (raw_init_map[x][y] == '@' || raw_init_map[x][y] == '+')
                que[0] = x * m + y;
    vis[que[0]] = true;
    while (lo < hi) {
        unsigned p = que[lo++];
        int x = p / m, y = p % m;
        for (auto [dx, dy] : {std::pair{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
            int nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < n && ny >= 0 && ny < m &&
                ny < raw_init_map[nx].size() && raw_init_map[nx][ny] != '#' &&
                !vis[nx * m + ny]) {
                vis[nx * m + ny] = true;
                que[hi++] = nx * m + ny;
            }
        }
    }
    unsigned minx = n, miny = m, maxx = 0, maxy = 0;
    for (unsigned x = 0; x < n; x++)
        for (unsigned y = 0; y < m; y++)
            if (vis[x * m + y]) {
                minx = std::min(minx, x);
                miny = std::min(miny, y);
                maxx = std::max(maxx, x);
                maxy = std::max(maxy, y);
            }
    assert(maxx - minx + 1 == mc::n && maxy - miny + 1 == mc::m);

    for (unsigned x = minx; x <= maxx; x++)
        for (unsigned y = miny; y <= maxy; y++)
            mask.push_back(vis[x * m + y]);

    player = mc::nm;
    for (int x = minx; x <= maxx; x++)
        for (int y = miny; y <= maxy && y < raw_init_map[x].size(); y++) {
            char ch = raw_init_map[x][y];
            if (ch == '$' || ch == '*')
                boxes.push_back((x - minx) * mc::m + y - miny);
            if (ch == '+' || ch == '*' || ch == '.')
                goals.push_back((x - minx) * mc::m + y - miny);
            if (ch == '@' || ch == '+') {
                assert(player == mc::nm);
                player = (x - minx) * mc::m + y - miny;
            }
            if (vis[x * m + y]) {
                for (auto [dx, dy] : {std::pair{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
                    int nx = x + dx, ny = y + dy;
                    if (nx < 0 || nx >= n || ny < 0 || ny >= m || !vis[nx * m + ny]) continue;
                    valid_moves[(x - minx) * mc::m + y - miny].push_back((nx - minx) * mc::m + ny - miny);
                    int mx = x - dx, my = y - dy;
                    if (mx < 0 || mx >= n || my < 0 || my >= m || !vis[mx * m + my]) continue;
                    valid_pushes.emplace_back((x - minx) * mc::m + y - miny,
                                              (nx - minx) * mc::m + ny - miny,
                                              (mx - minx) * mc::m + my - miny);
                }
            }
        }
    assert(boxes.size() == mc::tot_boxes && goals.size() == mc::tot_boxes);
    assert(player != mc::nm);
    delete[] vis;
    delete[] que;
}

