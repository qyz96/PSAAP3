using ADCME

m = 1
c_v = 1
### Constants: Chemical Kinetics Level
V = 1
N = 2
M = 4
ν1 = zeros(N,M)
ν2 = zeros(N,M)
ν = ν2 - ν1
ΔS = ones(M)
ΔH = ones(M)
pa = 1
### Constants: Combustion Chamber Level
min_dot = 1
mout_dot = 1
Yin = ones(N)
Yout = ones(N)
Qdot = 1
hin = 1
u = ones(N)
### Unknowns
T = 1
W = ones(N)
Y = ones(N)
###
ρ = m / V
R = 1
X = ρ * Y ./ W
p = sum(X) * T
Af = ones(M)
β = ones(M)
E = ones(M)
Q = ones(M)
Kf = ones(M)
Kr = ones(M)

##### Computing ω_dot 
Kf = Af .* (T .^ β) .* exp.(-E ./ (R * T))
Kr = Kf ./ (((pa/(R * T)) .^ sum(ν, dims=1)' .* exp.(ΔS ./ R - ΔH ./ (R * T))))
Q = Kf .* prod(X .^ ν1, dims=1)' .- Kr .* prod(X .^ ν2, dims=1)'
ω_dot = W .* sum(ν .* Q', dims=2)
###### Species Conservation
mgen_dot = V .* ω_dot .* W
Y_dot = (1 / m) .* min_dot .* (Yin .- Y) - mout_dot .* Y .+ mgen_dot 
###### Energy Conservation
T_dot = 1 / (m * c_v) * (Qdot + min_dot * (hin - sum(u .* Yin)) - p * V / m * mout_dot - sum(mgen_dot .* u))


