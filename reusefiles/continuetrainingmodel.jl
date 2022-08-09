using Flux
using Flux: params, DataLoader
using MLDatasets
using BSON: @load

include("./vaemodel.jl")

@load "./reusefiles/savedmodels/vae100epoch" model

include("./trainloops.jl")


loss = vaeloss(model, 1.0f0, 0.01f0)
traindata = reshape(MNIST(Float32, :train).features[:, :, 1:end] |> gpu, 28^2, :)
testdata = reshape(MNIST(Float32, :test).features[:, :, 1:end] |> gpu, 28^2, :)
trainloader = DataLoader(traindata, batchsize=32, shuffle=true)
testloader = DataLoader(testdata, batchsize=32, shuffle=true)


trainvalidatelognsave(loss, model, params(model), trainloader, testloader, Flux.Optimise.ADAM(0.0001), 100, "./reusefiles/models/", "./reusefiles/logs/", saveinterval=10, validateinterval=5)