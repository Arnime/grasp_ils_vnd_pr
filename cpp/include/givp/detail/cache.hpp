// SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <vector>

namespace givp::detail {

/// LRU evaluation cache to avoid redundant objective-function calls.
class EvaluationCache {
    std::size_t maxsize_;
    std::unordered_map<std::uint64_t, double> cache_;
    std::vector<std::uint64_t> insertion_order_;
    std::size_t hits_   = 0;
    std::size_t misses_ = 0;

    static std::uint64_t hash_solution(const std::vector<double>& solution,
                                        std::size_t half) {
        // FNV-1a over the integer-rounded representation
        std::uint64_t h = 14695981039346656037ULL;
        for (std::size_t i = 0; i < solution.size(); ++i) {
            std::int64_t rounded =
                (i < half)
                    ? static_cast<std::int64_t>(std::round(solution[i] * 1000.0))
                    : static_cast<std::int64_t>(std::round(solution[i]));
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(&rounded);
            for (std::size_t b = 0; b < sizeof(std::int64_t); ++b) {
                h ^= bytes[b];
                h *= 1099511628211ULL;
            }
        }
        return h;
    }

public:
    explicit EvaluationCache(std::size_t maxsize) : maxsize_(maxsize) {
        cache_.reserve(maxsize);
        insertion_order_.reserve(maxsize);
    }

    std::optional<double> get(const std::vector<double>& solution,
                               std::size_t half) {
        auto key = hash_solution(solution, half);
        auto it  = cache_.find(key);
        if (it != cache_.end()) {
            ++hits_;
            return it->second;
        }
        ++misses_;
        return std::nullopt;
    }

    void put(const std::vector<double>& solution, std::size_t half, double cost) {
        auto key = hash_solution(solution, half);
        if (cache_.count(key)) return;
        if (cache_.size() >= maxsize_ && !insertion_order_.empty()) {
            cache_.erase(insertion_order_.front());
            insertion_order_.erase(insertion_order_.begin());
        }
        cache_[key] = cost;
        insertion_order_.push_back(key);
    }

    void clear() {
        cache_.clear();
        insertion_order_.clear();
    }

    struct Stats {
        std::size_t hits, misses;
        double      hit_rate;
        std::size_t size;
    };

    Stats stats() const {
        std::size_t total = hits_ + misses_;
        double rate = (total > 0) ? static_cast<double>(hits_) / total : 0.0;
        return {hits_, misses_, rate, cache_.size()};
    }
};

} // namespace givp::detail
