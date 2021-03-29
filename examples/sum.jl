using BenchmarkTools
using Catwalk
using Folds
using FoldsCatwalk
using Transducers

type_instability(x) = Val(round(Int, 2x))

# This version emphasizes the possible gain better:
#vals=[Val(i) for i= 1:100]
#type_instability(x) = vals[abs(round(Int, 4x)) % length(vals) + 1]

asint(::Val{x}) where {x} = x::Int

function sum_baseline(xs)
    itr = xs |> Map(type_instability) |> Map(asint)
    Folds.sum(itr, SequentialEx())
end

function sum_catwalk(xs)
    itr = xs |> Map(type_instability) |> OptimizeInner() |> Map(asint)
    Folds.sum(itr, CatwalkEx())
end

function sum_catwalk_tuned(xs)
    itr = xs |> Map(type_instability) |> OptimizeInner() do 
        boost = Catwalk.CallBoost(:next;
                    optimizer = Catwalk.TopNOptimizer(15),
                    profilestrategy = Catwalk.SparseProfile(0))
        jit = Catwalk.JIT(boost; explorertype = Catwalk.NoExplorer)
    end |> Map(asint)
    Folds.sum(itr, CatwalkEx())
end

function demo_sum(n=20_000_000)
    xs = randn(n)

    println("Baseline:")
    baseline_result = @btime sum_baseline($xs)

    println("Catwalk defaults:")
    catwalk_result = @btime sum_catwalk($xs)

    println("Catwalk tuned:")
    catwalk_tuned_result = @btime sum_catwalk_tuned($xs)

    @assert baseline_result == catwalk_result
    @assert baseline_result == catwalk_tuned_result
end