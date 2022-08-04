using Test
using TensorBoardLogger
using Logging
using MLDatasets
using Flux: @epochs, train!, params, DataLoader

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

    data = reshape(MNIST(Float32,:train).features, 28^2, :)
    loss = vaeloss(vaemodel, 0.5, 0.5)
end

vaemodel = makevae()

loss = vaeloss(vaemodel, 0.5f0, 0.5)
data = reshape(MNIST(Float32,:train).features[:,:,1:64], 28^2, :)
loader = DataLoader(data, batchsize=32, shuffle=true)
vaemodel(data)
with_logger(TBLogger("./reusefiles/logs/")) do
    @epochs 2 train!(loss, params(vaemodel), loader, Flux.Optimise.ADAM(0.001))
end

#include("./trainloops.jl")
#trainlognsave(loss,)

#does not work
#train!(vaeloss(vaemodel, 0.5, 0.5) ,params(vaemodel), loader, Flux.Optimise.ADAM(0.001))