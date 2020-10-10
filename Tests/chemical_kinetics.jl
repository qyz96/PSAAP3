using ADCME
m = 1 # Total mass
### Constants: Chemical Kinetics Level
V = 1  # Volume
N = 2  # Number of Species
M = 4  # Number of Reactions
W = ones(N) # Molar weight
ν1 = zeros(N,M) # Forward molar stoichiometric coefficients
ν2 = zeros(N,M) # Backward model stoichiometric coefficients
ν = ν2 - ν1  
ΔS = ones(M)  # Enthalpy changes for reaction j
ΔH = ones(M)  # Entahlpy for reaction j
pa = 1 # 1 bar
R = 8.31446261815324 # Gas constant
### Constants: Combustion Chamber Level
min_dot = 1 # Rate at which mass enters the chamber
mout_dot = 1 # Rate at which mass leaves the chamber
Yin = ones(N) # Mass fraction of species entering the chamber
Yout = ones(N) # Mass fraction of species leaving the chamber
Qdot = 1 # Heating source
hin = 1 # Enthalpy of input
### Unknowns
T = 1 # Temperature
Y = ones(N) # Mass fractions
### Quantities that depend on the Unknowns
u = 2.5 * R ./ W .* T   # Internal energy for species
c_v = sum(2.5 * R .* Y ./ W) # Mass heat capacities
ρ = m / V # density
X = ρ * Y ./ W # Concentration
p = sum(X) * T # pressure
Af = ones(M) # preexponential constant Afj
β = ones(M) # Temperature exponent
E = ones(M) # Activation energy for the reactions
Q = ones(M) # Individual progress rates
Kf = ones(M) # Forward reaction coefficients
Kr = ones(M) # Reverse reaction coefficients
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


