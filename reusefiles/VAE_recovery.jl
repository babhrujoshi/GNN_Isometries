using Plots, Images

include("vaemodels.jl")
include("compressedsensing.jl")

function plot_arrayoftestMNISTrecoveries(decoder, numimg, aimedm, k, n; plotwidth=800, kwargs...)
    MNISTtestdata = reshape(MNIST(Float32, :test).features, 28^2, :)

    function getimgpair()
        truesignal = MNISTtestdata[:, rand(1:size(MNISTtestdata)[2])]
        F = samplefourierwithoutreplacement(aimedm, n)
        measurements = F * truesignal
        recovery = recoversignal(measurements, F, decoder, k, tolerance=5e-4; kwargs...)
        return [reshape(truesignal, 28, 28)', reshape(recovery, 28, 28)']
    end

    plots = vcat([hcat((plot(colorview(Gray, 1.0f0 .- img), axis=([], false)) for img in getimgpair())...) for i in 1:numimg]...)
    scale = plotwidth / numimg
    plot(plots..., layout=(2, numimg), size=(numimg * scale, 2 * scale), background_color=:grey93)
end