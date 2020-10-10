
using PyPlot
T = 10
N = 10
δt = T/N
u = ones(N)
λ = -5
#f = y -> λ*y
for i = 1 : N - 1
    u[i+1] = u[i] / (1-δt * λ)
end
fg = plot(u)
savefig("test.png")