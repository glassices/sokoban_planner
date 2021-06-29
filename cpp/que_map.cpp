#include <cassert>
#include <cstring>

#include "que_map.hpp"
#include "map.hpp"

QueMap::QueMap(size_t _maxn)
    : cntn(0), maxn(_maxn),
      num_pushes(Map::get_instance().get_num_pushes()),
      h(new Record*[nhash]),
      data1(new Record[maxn]),
      data2(new float[num_pushes * maxn]),
      lend_que(new Record),
      rend_que(new Record)
{
    std::fill_n(h, nhash, nullptr);
    for (unsigned i = 0; i < maxn; i++)
        (data1 + i)->prob = data2 + i * num_pushes;

    lend_que->rque = rend_que;
    rend_que->lque = lend_que;
}

QueMap::~QueMap()
{
    delete[] h;
    delete[] data1;
    delete[] data2;
    delete lend_que;
    delete rend_que;
}

size_t get_hash(unsigned char state[])
{
    size_t seed = 0;
    if (sizeof(size_t) == 4) {
        for (unsigned i = 0; i < mc::nm; i++)
            seed ^= state[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    else {
        for (unsigned i = 0; i < mc::nm; i++)
            seed ^= state[i] + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    }
    return seed;
}

std::optional<std::pair<float, float*>> QueMap::query(unsigned char state[])
{
    auto hvalue = get_hash(state);
    for (auto rec = h[hvalue % nhash]; rec; rec = rec->rmap)
        if (hvalue == rec->hvalue && memcmp(state, rec->state, sizeof(unsigned char) * mc::nm) == 0) {
            /* hit, move the element to the right most side of queue */
            rec->lque->rque = rec->rque;
            rec->rque->lque = rec->lque;
            rend_que->lque->rque = rec;
            rec->lque = rend_que->lque;
            rec->rque = rend_que;
            rend_que->lque = rec;
            return std::make_pair(rec->value, rec->prob);
        }
    return std::nullopt;
}

void QueMap::insert(unsigned char state[], float value, torch::Tensor prob)
{
    auto hvalue = get_hash(state);
    Record* cnt;
    if (cntn == maxn) {
        /* remove the left most element */
        cnt = lend_que->rque;
        cnt->lque->rque = cnt->rque;
        cnt->rque->lque = cnt->lque;
        auto hslot = cnt->hvalue % nhash;
        if (h[hslot] == cnt) h[hslot] = cnt->rmap;
        if (cnt->lmap) cnt->lmap->rmap = cnt->rmap;
        if (cnt->rmap) cnt->rmap->lmap = cnt->lmap;
    }
    else cnt = data1 + cntn++;

    cnt->value = value;
    auto acs_prob = prob.accessor<float, 1>();
    for (unsigned i = 0; i < num_pushes; i++)
        cnt->prob[i] = acs_prob[i];
    memcpy(cnt->state, state, sizeof(unsigned char) * mc::nm);
    cnt->hvalue = hvalue;

    /* insert to the right most */
    rend_que->lque->rque = cnt;
    cnt->lque = rend_que->lque;
    cnt->rque = rend_que;
    rend_que->lque = cnt;

    /* insert to map */
    auto hslot = hvalue % nhash;
    if (!h[hslot]) {
        h[hslot] = cnt;
        cnt->lmap = cnt->rmap = nullptr;
    }
    else {
        cnt->rmap = h[hslot];
        h[hslot]->lmap = cnt;
        cnt->lmap = nullptr;
        h[hslot] = cnt;
    }
}

