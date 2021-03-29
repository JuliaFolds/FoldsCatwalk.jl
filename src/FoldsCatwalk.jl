module FoldsCatwalk

export CatwalkEx, OptimizeInner

using Catwalk
using Transducers
using Transducers:
    Executor,
    R_,
    Transducer,
    complete,
    extract_transducer,
    inner,
    next,
    start,
    xform

"""
    CatwalkEx(batchsize)

JIT compile a fold for every `batchsize` iteration. See [`OptimizeInner`](@ref)
for how to specify what to JIT.
"""
struct CatwalkEx <: Executor
    batchsize::Int
end
CatwalkEx() = CatwalkEx(1_000_000)

Transducers.maybe_set_simd(exc::CatwalkEx, _) = exc

"""
    OptimizeInner([factory = Catwalk.JIT])

A marker for instructing `CatwalkEx` to optimize transducers and reducing
function inner to this position. Optionally, it takes a callable to create a
Catwalk JIT object.

```
xs |> Map(f) |> OptimizeInner() |> Map(g)
                                   ------
                                  JIT target
```

It supports the type instability caused by the outer transducers (such as
`Map(f)`), collection (`xf`), and/or the reducing function output.
"""
struct OptimizeInner{Factory}
    factory::Factory
end

OptimizeInner(::Type{T}) where {T} = OptimizeInner{Type{T}}(T)
OptimizeInner() = OptimizeInner(Catwalk.JIT)

(f::OptimizeInner{Factory})(outer::Outer) where {Factory,Outer} =
    OptimizeInnerFoldable{Factory,Outer}(f.factory, outer)

struct OptimizeInnerFoldable{Factory,Outer}
    factory::Factory
    outer::Outer
end

function Transducers.transduce(xf, rf, init, coll, exc::CatwalkEx)
    xfi, opt = extract_transducer(xf(coll))
    opt::OptimizeInnerFoldable
    xfo, itr = extract_transducer(opt.outer)
    rf0 = reducingfunction(xfi ∘ xfo, rf)
    acc = start(rf0, init)
    y = iterate(itr)
    y === nothing && return complete(rf0, acc)
    acc = next(rf0, acc, first(y))
    state = last(y)
    batchsize = exc.batchsize
    @assert batchsize > 0
    jit = opt.factory()
    while true
        Catwalk.step!(jit)
        rf1 = reducingfunction(xfi ∘ OptimizeXF(Catwalk.ctx(jit)) ∘ xfo, rf)
        acc, state, done = onebatch(rf1, acc, itr, state, batchsize)
        done && return complete(rf1, acc)
    end
end

function onebatch(rf::RF, acc, itr, state, counter) where {RF}
    jitctx = xform(inner(rf)).jitctx
    while counter != 0
        y = iterate(itr, state)
        y === nothing && return acc, state, true
        state = last(y)
        val = next(rf, acc, first(y), jitctx)
        val isa Reduced && return val, state, true
        acc = val
        counter -= 1
    end
    return acc, state, false
end

struct OptimizeXF{C} <: Transducer
    jitctx::C
end

@inline @jit next mapped function Transducers.next(rf::R_{Map}, acc, input, jitctx)
    mapped = xform(rf).f(input)
    next(inner(rf), acc, mapped)
end

function Transducers.next(rf::R_{OptimizeXF}, acc, input)
    next(inner(rf), acc, input)
end

end # module FoldsCatwalk
