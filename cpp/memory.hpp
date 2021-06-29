#pragma once

#include <cassert>
#include <utility>
#include <cstdlib>

template<size_t N, typename T>
class MemoryPool
{
    static_assert(sizeof(T) >= sizeof(void*));
    static_assert(alignof(T) % alignof(void*) == 0);
    static_assert(N > 0);

public:
    MemoryPool()
        : data(reinterpret_cast<char*>(aligned_alloc(alignof(T), N * sizeof(T))))
    {
        assert(data);
        head = data;
        for (size_t i = 0; i + 1 < N; i++)
            *reinterpret_cast<void**>(data + i * sizeof(T)) = data + (i + 1) * sizeof(T);
        *reinterpret_cast<void**>(data + (N - 1) * sizeof(T)) = nullptr;
    }

    ~MemoryPool()
    {
        size_t avail = 0;
        for (auto cnt = head; cnt; cnt = *reinterpret_cast<void**>(cnt)) avail++;
        assert(avail == N);
        std::free(data);
    }

    MemoryPool(const MemoryPool&) = delete;
    void operator=(const MemoryPool&) = delete;

    template<typename... Args>
    T* alloc(Args&&... args)
    {
        assert(head);
        T* result = reinterpret_cast<T*>(head);
        head = *reinterpret_cast<void**>(head);
        new (result) T(std::forward<Args>(args)...);
        return result;
    }
    
    void free(T* ptr)
    {
        ptr->destroy(*this);
        *reinterpret_cast<void**>(ptr) = head;
        head = ptr;
    }
    
    void free_without_destroy(T* ptr)
    {
        *reinterpret_cast<void**>(ptr) = head;
        head = ptr;
    }

private:
    char* data;
    void* head;
};

