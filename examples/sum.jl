using BenchmarkTools
using Folds
using FoldsCatwalk
using Transducers

type_instability(x) = Val(round(Int, 2x))
asint(::Val{x}) where {x} = x::Int

function sum_baseline(xs)
    itr = xs |> Map(type_instability) |> Map(asint)
    Folds.sum(itr, SequentialEx())
end

function sum_catwalk(xs)
    itr = xs |> Map(type_instability) |> OptimizeInner() |> Map(asint)
    Folds.sum(itr, CatwalkEx())
end

function demo_sum()
    xs = randn(1000_000)
    @assert sum_baseline(xs) == sum_catwalk(xs)
    @btime sum_baseline($xs)
    @btime sum_catwalk($xs)
end
