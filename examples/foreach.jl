using BenchmarkTools
using FoldsCatwalk
using Transducers

returnnothing(_) = nothing
returnnothing(_, _) = nothing

foreach1(f, xs, exc) = transduce(Map(f), returnnothing, nothing, xs, exc)

type_instability(x) = Val(round(Int, 2x))

trydispatch(_) = nothing
trydispatch(::Val{typemax(Int)}) = error("unexpected!") # so that Julia has to try dispatch

function foreach_baseline(xs)
    itr = xs |> Map(type_instability)
    foreach1(trydispatch, itr, SequentialEx())
end

function foreach_catwalk(xs)
    itr = xs |> Map(type_instability) |> OptimizeInner()
    foreach1(trydispatch, itr, CatwalkEx())
end

function demo_foreach()
    xs = randn(1000_000)
    @btime foreach_baseline($xs)
    @btime foreach_catwalk($xs)
end
