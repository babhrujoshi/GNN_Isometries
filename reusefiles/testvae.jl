using Test
using TensorBoardLogger
using Logging
using MLDatasets
using Flux
using Flux: @epochs, train!, params, DataLoader
using CUDA

#Base.global_logger(TBLogger("./reusefiles/logs/"))
#Logging.global_logger(Logging.ConsoleLogger())

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

    loss = vaeloss(vaemodel, 0.5, 0.5)
    data = reshape(MNIST(Float32,:train).features[:,:,1:4], 28^2, :)
    vaemodel(data[:,1:2])
    
    loader = DataLoader(data, batchsize=2, shuffle=true)

    #@epochs 2 train!(loss, params(vaemodel), loader, Flux.Optimise.ADAM(0.001))

    include("trainloops.jl")

    #trainlognsave(loss, vaemodel, params(vaemodel), loader, Flux.Optimise.ADAM(0.001), 20, "./reusefiles/models/","./reusefiles/logs/",save_interval = 4)

    trainlognsave(loss, vaemodel, params(vaemodel), loader, Flux.Optimise.ADAM(0.001), 20, "./reusefiles/models/","./reusefiles/logs/",save_interval = 4)
end

function trainvaefromMNIST()
    include("./vaemodel.jl")
    include("./trainloops.jl")
    vaemodel = makevae()
    loss = vaeloss(vaemodel, 0.5, 0.5)
    traindata = reshape(MNIST(Float32,:train).features[:,:,1:4] |> gpu, 28^2, :)
    testdata = reshape(MNIST(Float32,:test).features[:,:,1:4] |> gpu, 28^2, :)
    trainloader = DataLoader(traindata, batchsize=2, shuffle=true)
    testloader = DataLoader(testdata, batchsize=2, shuffle=true)
    trainvalidatelognsave(loss, vaemodel, params(vaemodel), trainloader, testloader, Flux.Optimise.ADAM(0.001), 4, "./reusefiles/models/","./reusefiles/logs/",saveinterval = 2, validateinterval=2)
end

trainvaefromMNIST()

#include("./trainloops.jl")
#trainlognsave(loss,)

#does not work
#train!(vaeloss(vaemodel, 0.5, 0.5) ,params(vaemodel), loader, Flux.Optimise.ADAM(0.001))