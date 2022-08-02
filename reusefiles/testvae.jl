using Test
using TensorBoardLogger
using Logging

#Base.global_logger(TBLogger("./reusefiles/logs/"))
Logging.global_logger(Logging.ConsoleLogger())

@info("name", test=0.1, other=0.2)
@info other = 0.4

@testset "VAE" begin
    include("./vaemodel.jl")
    vaemodel = makevae()

    #check forward pass works (no errors)
    input_length = size(vaemodel.encoder.encoderbody[1].weight)[2]
    x = randn(input_length) / sqrt(input_length)
    vaemodel(x)
    vaeloss(vaemodel, 0.5, 0.5)(x)

    #test with batch_size ==3 
    x = randn(input_length, 3) / sqrt(input_length)
    vaemodel(x)
    vaeloss(vaemodel, 0.5, 0.5)(x)

    x = [randn(input_length, 3) for i in 1:5]
    train!(vaeloss(vaemodel, 0.5, 0.5), params(vaemodel), x, Flux.Optimise.ADAM(0.001))
end