### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ 1f7a5367-69ea-41a8-aa94-818f2d791c4e
using Pkg

# ╔═╡ 7e3750bc-cedd-4616-9763-4f09409597da
Pkg.activate("./")

# ╔═╡ 2b040426-5a6e-49e6-82cb-81a539568aba
begin
	using BSON: load
	using Flux
end

# ╔═╡ b8b174f7-dd75-4ece-b579-bcce0b7d601e
using .VaeModels

# ╔═╡ cdc6a668-a12e-4ca0-907f-dd60109a84e5
include("./reusefiles/vaemodels.jl")

# ╔═╡ 14826669-ecda-4177-a86a-a621c8473b17
VaeModels

# ╔═╡ afeb0670-2636-4413-9e25-aac0d416eecf
load("./reusefiles/savedmodels/incoherentepoch20", @module)

# ╔═╡ e474a4b7-9cba-4991-ae16-2b3fe900859f


# ╔═╡ 4ef1a5b0-4c40-455d-8bd7-829bd25a7782


# ╔═╡ 8504381e-7834-4a74-bf46-4cc33580c665


# ╔═╡ 0e9d53c8-6e5b-4f79-8df1-a966b8e137b5
VaeModels

# ╔═╡ 553a73c8-37c2-46ca-b12f-984b73f82a15


# ╔═╡ a4aae9c8-95f2-48a9-b729-e880169a3184


# ╔═╡ 9f333d7d-c41b-4f4d-888f-161097b83367


# ╔═╡ d1529d4e-0dce-483b-8ed3-d0e567cc41dd


# ╔═╡ 7b1ae841-0355-44d2-9320-e3cb62b04227


# ╔═╡ 511e94a8-448b-4991-aea7-1a0d60c336a8
returnobj = load("./reusefiles/savedmodels/incoherentepoch20", VaeModels)

# ╔═╡ 9cd0f179-da54-4c5d-930c-71819bd99479


# ╔═╡ 1d2418af-04a6-4db7-9297-59df7133307e
@__MODULE__

# ╔═╡ 3b1123ec-909b-4390-b0f6-59456d60636b
Main

# ╔═╡ 32352960-f607-4790-bc6e-30c4d8130400


# ╔═╡ 83305fb0-8f8d-4a21-bab5-b084f381f7bd


# ╔═╡ f0370d71-63cc-4456-acea-42c1032b0ab8


# ╔═╡ 0bba19b3-0d93-4434-b2cb-6aa08f4adf77


# ╔═╡ 0f2dda46-7965-4383-a8ae-6ee6c2a3d5d0


# ╔═╡ 244c06b1-f068-418d-a301-de5752f25048


# ╔═╡ d6f432d2-ab83-41af-9e51-93d084aebeff


# ╔═╡ 22848cc7-c74a-46bf-b5fe-f49ca833a168


# ╔═╡ Cell order:
# ╠═1f7a5367-69ea-41a8-aa94-818f2d791c4e
# ╠═7e3750bc-cedd-4616-9763-4f09409597da
# ╠═cdc6a668-a12e-4ca0-907f-dd60109a84e5
# ╠═2b040426-5a6e-49e6-82cb-81a539568aba
# ╠═14826669-ecda-4177-a86a-a621c8473b17
# ╠═afeb0670-2636-4413-9e25-aac0d416eecf
# ╠═b8b174f7-dd75-4ece-b579-bcce0b7d601e
# ╠═e474a4b7-9cba-4991-ae16-2b3fe900859f
# ╠═4ef1a5b0-4c40-455d-8bd7-829bd25a7782
# ╠═8504381e-7834-4a74-bf46-4cc33580c665
# ╠═0e9d53c8-6e5b-4f79-8df1-a966b8e137b5
# ╠═553a73c8-37c2-46ca-b12f-984b73f82a15
# ╠═a4aae9c8-95f2-48a9-b729-e880169a3184
# ╠═9f333d7d-c41b-4f4d-888f-161097b83367
# ╠═d1529d4e-0dce-483b-8ed3-d0e567cc41dd
# ╠═7b1ae841-0355-44d2-9320-e3cb62b04227
# ╠═511e94a8-448b-4991-aea7-1a0d60c336a8
# ╠═9cd0f179-da54-4c5d-930c-71819bd99479
# ╠═1d2418af-04a6-4db7-9297-59df7133307e
# ╠═3b1123ec-909b-4390-b0f6-59456d60636b
# ╠═32352960-f607-4790-bc6e-30c4d8130400
# ╠═83305fb0-8f8d-4a21-bab5-b084f381f7bd
# ╠═f0370d71-63cc-4456-acea-42c1032b0ab8
# ╠═0bba19b3-0d93-4434-b2cb-6aa08f4adf77
# ╠═0f2dda46-7965-4383-a8ae-6ee6c2a3d5d0
# ╠═244c06b1-f068-418d-a301-de5752f25048
# ╠═d6f432d2-ab83-41af-9e51-93d084aebeff
# ╠═22848cc7-c74a-46bf-b5fe-f49ca833a168
