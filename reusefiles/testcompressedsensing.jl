using Test
using Flux
using LinearAlgebra

include("./CompressedSensing.jl")

@testset "main_test" begin
    model = Chain(Dense(16 => 16))
    true_signal = model(randn(16) / 4)

    recov = recoveryerror(true_signal, model, I, 16, tolerance=1e-8)
    @test recov ≈ 0 atol = 1e-3

    loss(x) = x[1]^2 + x[2]^2 / 5
    z = [-1.0, 3.0]
    optimise!(loss, z, tolerance=1e-12)
    atol = 1e-3
    @test z ≈ [0, 0] atol = 1e-3
    @test recoveryerror(true_signal, model, samplefourierwithoutreplacement(16, 16), 16, tolerance=1e-9) ≈ 0 atol = 1e-2
end


#samplefourierwithoutreplacement(4, 10)