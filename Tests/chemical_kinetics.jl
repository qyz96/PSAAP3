using ADCME
### Test Reaction CH4 + 2 O2 => CO2 + 2 H20
### Constants: Chemical Kinetics Level
V = 0.7693166084401741  # Volume
m = 1 # Total mass
N = 4  # Number of Species
M = 1  # Number of Reactions
W = [16.04276; 31.9988; 44.0098; 18.01528] # Molar weight
cvk = [1710.8, 658.57, 656.75, 1403.4]
ν1 = [1; 0.5; 0; 0] # Forward molar stoichiometric coefficients
ν2 = [0; 0; 1; 2] # Backward model stoichiometric coefficients
forward_order = [1; 2; 0; 0] # Forward reaction order
reverse_order = [0; 0; 1; 2] # Reverse reaction order
ν = ν2 - ν1  
S0 = [186.3; 205.1; 213.7; 188.8]
H0 = [-74.8; 0; -393.5; -241.8]
ΔS = [-5163.4343755]  # Entropy change for reaction j
ΔH = [-8.0253912e+08] # Entahlpy change for reaction j
pa = 100000 # 1 bar
R = 8314.46261815324 # Gas constant in kmol
### Constants: Combustion Chamber Level
min_dot = 0 # Rate at which mass enters the chamber
mout_dot = 0 # Rate at which mass leaves the chamber
Yin = ones(N) # Mass fraction of species entering the chamber
Yout = ones(N) # Mass fraction of species leaving the chamber
Qdot = 0 # Heating source
m_dot = min_dot - mout_dot
hin = 1 # Enthalpy of input
### Unknowns
T = 300 # Temperature
Y = [0.25; 0.25; 0.25; 0.25] # Mass fractions
### Quantities that depend on the Unknowns
ρ = m / V # density
Af = [3.47850e+08] # preexponential constant Afj
β = [0] # Temperature exponent
E = [83680000.0] # Activation energy for the reactions in kJ (divide the value from Cantera by 1000)
Q = ones(M) # Individual progress rates
Kf = ones(M) # Forward reaction coefficients
Kr = zeros(M) # Reverse reaction coefficients

##### Computing ω_dot 
c_v = sum(cvk .* Y) # Mass heat capacities
X = ρ * Y ./ W # Concentration
u = X .* cvk .* W .* T   # Internal energy for species
p = sum(X) * R * T # pressure
Kf = Af .* (T .^ β) .* exp.(-E ./ (R * T))
Kr = Kf ./ (((pa/(R * T)) .^ sum(ν, dims=1)' .* exp.(ΔS ./ R - ΔH ./ (R * T))))
Q = Kf .* prod(X .^ ν1, dims=1)' .- Kr .* prod(X .^ ν2, dims=1)'
ω_dot = W .* sum(ν .* Q', dims=2)
###### Species Conservation
mgen_dot = V .* ω_dot .* W
Y_dot = (1 / m) .* min_dot .* (Yin .- Y) - mout_dot .* Y .+ mgen_dot 
###### Energy Conservation
T_dot = 1 / (m * c_v) * (-Qdot + min_dot * (hin - sum(u .* Yin)) - p * V / m * mout_dot - sum(mgen_dot .* u))


function F(Y, T)
    ##### Computing ω_dot 
    c_v = sum(2.5 * R .* Y ./ W) # Mass heat capacities
    X = ρ * Y ./ W # Concentration
    p = sum(X) * T # pressure
    Kf = Af .* (T .^ β) .* exp.(-E ./ (R * T))
    Kr = Kf ./ (((pa/(R * T)) .^ sum(ν, dims=1)' .* exp.(ΔS ./ R - ΔH ./ (R * T))))
    Q = Kf .* prod(X .^ ν1, dims=1)' .- Kr .* prod(X .^ ν2, dims=1)'
    ω_dot = W .* sum(ν .* Q', dims=2)
    ###### Species Conservation
    mgen_dot = V .* ω_dot .* W
    Y_dot = (1 / m) .* min_dot .* (Yin .- Y) - mout_dot .* Y .+ mgen_dot 
    ###### Energy Conservation
    T_dot = 1 / (m * c_v) * (-Qdot + min_dot * (hin - sum(u .* Yin)) - p * V / m * mout_dot - sum(mgen_dot .* u))
    J = gradients(Y_dot, Y)
    return Y_dot, T_dot, J
end