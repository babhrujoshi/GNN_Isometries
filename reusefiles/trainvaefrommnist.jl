using Flux
using Flux: params, DataLoader
using MLDatasets

include("./vaemodel.jl")
include("./trainloops.jl")

function trainvaefromMNIST()
    vaemodel = makevae()
    loss = vaeloss(vaemodel, 1f0, 0.01f0)
    traindata = reshape(MNIST(Float32,:train).features[:,:,1:end] |> gpu, 28^2, :)
    testdata = reshape(MNIST(Float32,:test).features[:,:,1:end] |> gpu, 28^2, :)
    trainloader = DataLoader(traindata, batchsize=32, shuffle=true)
    testloader = DataLoader(testdata, batchsize=32, shuffle=true)
    trainvalidatelognsave(loss, vaemodel, params(vaemodel), trainloader, testloader, Flux.Optimise.ADAM(0.001), 100, "./reusefiles/models/","./reusefiles/logs/",saveinterval = 10, validateinterval=5)
end

trainvaefromMNIST()