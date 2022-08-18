using Flux: Chain, Dense, params, DataLoader
using MLDatasets
using BenchmarkTools

include("vaemodel.jl")
#include("trainloops.jl")

model = makeVAE(512, 512, 16)
batchsize = 64

traindata = reshape(MNIST(Float32, :train).features[:, :, 1:end], 28^2, :)
trainloader = DataLoader(traindata, batchsize=batchsize)

trainVAE(1.0f0, 0.01f0, model, params(model), trainloader, Flux.Optimise.ADAM(0.001), 40, "./reusefiles/models/", "./reusefiles/logs/", label="tinyv9", loginterval=100)

#@btime trainvalidatelognsavewithlog(loss, model, params(model), trainloader, validateloader, Flux.Optimise.ADAM(0.001), 1, "./reusefiles/models/", "./reusefiles/logs/", label="tinyv5", loginterval=5, saveinterval=50)

#end