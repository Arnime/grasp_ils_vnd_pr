# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

@testset "EvaluationCache" begin
    @testset "basic operations" begin
        GIVPOptimizer.set_integer_split!(4)  # all continuous
        cache = GIVPOptimizer.EvaluationCache(; maxsize = 3)

        sol1 = [1.0, 2.0, 3.0, 4.0]
        @test GIVPOptimizer.cache_get(cache, sol1) === nothing
        @test cache.misses == 1

        GIVPOptimizer.cache_put!(cache, sol1, 10.0)
        @test GIVPOptimizer.cache_get(cache, sol1) == 10.0
        @test cache.hits == 1

        stats = GIVPOptimizer.cache_stats(cache)
        @test stats["hits"] == 1
        @test stats["misses"] == 1
        @test stats["size"] == 1
        @test stats["hit_rate"] == 50.0
    end

    @testset "LRU eviction" begin
        GIVPOptimizer.set_integer_split!(2)
        cache = GIVPOptimizer.EvaluationCache(; maxsize = 2)

        GIVPOptimizer.cache_put!(cache, [1.0, 2.0], 10.0)
        GIVPOptimizer.cache_put!(cache, [3.0, 4.0], 20.0)
        @test GIVPOptimizer.cache_stats(cache)["size"] == 2

        # Adding third entry evicts the oldest (first)
        GIVPOptimizer.cache_put!(cache, [5.0, 6.0], 30.0)
        @test GIVPOptimizer.cache_stats(cache)["size"] == 2
        @test GIVPOptimizer.cache_get(cache, [1.0, 2.0]) === nothing  # evicted
        @test GIVPOptimizer.cache_get(cache, [3.0, 4.0]) == 20.0  # still there
        @test GIVPOptimizer.cache_get(cache, [5.0, 6.0]) == 30.0  # still there
    end

    @testset "overwrite existing key" begin
        GIVPOptimizer.set_integer_split!(2)
        cache = GIVPOptimizer.EvaluationCache(; maxsize = 5)
        GIVPOptimizer.cache_put!(cache, [1.0, 2.0], 10.0)
        GIVPOptimizer.cache_put!(cache, [1.0, 2.0], 99.0)  # same key, different value
        @test GIVPOptimizer.cache_get(cache, [1.0, 2.0]) == 99.0
        @test GIVPOptimizer.cache_stats(cache)["size"] == 1  # no duplicate entries
    end

    @testset "clear" begin
        GIVPOptimizer.set_integer_split!(2)
        cache = GIVPOptimizer.EvaluationCache(; maxsize = 10)
        GIVPOptimizer.cache_put!(cache, [1.0, 2.0], 10.0)
        GIVPOptimizer.cache_clear!(cache)
        @test cache.hits == 0
        @test cache.misses == 0
        # After clear, previously cached entry is gone
        @test GIVPOptimizer.cache_get(cache, [1.0, 2.0]) === nothing
        @test cache.misses == 1  # incremented by cache_get miss
    end

    @testset "hit_rate when empty" begin
        cache = GIVPOptimizer.EvaluationCache()
        stats = GIVPOptimizer.cache_stats(cache)
        @test stats["hit_rate"] == 0.0
    end

    @testset "integer rounding in hash" begin
        GIVPOptimizer.set_integer_split!(1)  # first var continuous, rest integer
        cache = GIVPOptimizer.EvaluationCache()
        # Two solutions that round to the same thing
        GIVPOptimizer.cache_put!(cache, [1.001, 2.4], 10.0)
        @test GIVPOptimizer.cache_get(cache, [1.001, 2.6]) === nothing  # different integer part
        @test GIVPOptimizer.cache_get(cache, [1.001, 2.3]) == 10.0  # rounds to same
    end

    GIVPOptimizer.set_integer_split!(nothing)
end
