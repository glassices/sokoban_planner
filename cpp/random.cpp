#include "random.hpp"

Random::Random() : _gen(std::random_device()())
{}

Random& Random::get_instance()
{
    static Random instance;
    return instance;
}

float Random::get_gamma(float alpha, float beta)
{
    std::gamma_distribution<float> dist(alpha, beta);
    std::scoped_lock lck(_mutex);
    return dist(_gen);
}

int Random::get_int(int min, int max)
{
    std::uniform_int_distribution<> dist(min, max);
    std::scoped_lock lck(_mutex);
    return dist(_gen);
}

