using Test
using Flux
using LinearAlgebra

include("./CompressedSensing.jl")

model = Chain(Dense(16 => 16))
true_signal = model(randn(16) / 4)

@testset begin
    recov = recoveryerror(true_signal, model, I, 16, tolerance=1e-8)
    @info recov
    @test recov ≈ 0 atol = 1e-4
end




