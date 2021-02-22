import Pkg
using Plots
using PyCall
# using PlotThemes
# theme(PlotThemes:dark)
# PlotThemes.dark_bg
# using DifferentialEquations

# Plots.theme(:juno)
Plots.theme(:solarized_light)

# our modules
#using ModThermo: compute_cpHS, compute_falloff
include("RKF.jl")
using .RKF: coeff_RKF

# using PyCall
#### Read data
py"""
import numpy as np
"""

const MIN_SCALE_FACTOR = 0.125;
# const MIN_SCALE_FACTOR = 0.9;
const MAX_SCALE_FACTOR = 4.0;
const MAX_ATTEMPTS = 15
# const ADAPTIVE_TIMESTEP = true
const ADAPTIVE_TIMESTEP = false

function compute_cpHS(a, R, T)
    m = size(a)[1]
    cpHS = zeros(m, 3)
    for i = 1:m
        if (T > a[i,1])
            cpHS[i,1] = (a[i,2] + a[i,3] * T + a[i,4] * T^2 + a[i,5] * T^3 + a[i,6] * T^4) * R
            cpHS[i,2] = (a[i,2] + a[i,3] * T / 2 + a[i,4] * T^2 /3 + a[i,5] * T^3 /4 + a[i,6] * T^4 /5 + a[i,7]/T) * R * T
            cpHS[i,3] = (a[i,2] * log(T) + a[i,3] * T + a[i,4] * T^2 /2 + a[i,5] * T^3 /3 + a[i,6] * T^4 / 4 + a[i,8]) * R
        else
            cpHS[i,1] = (a[i,9] + a[i,10] * T + a[i,11] * T^2 + a[i,12] * T^3 + a[i,13] * T^4) * R
            cpHS[i,2] = (a[i,9] + a[i,10] * T / 2 + a[i,11] * T^2 /3 + a[i,12] * T^3 /4 + a[i,13] * T^4 /5 + a[i,14]/T) * R * T
            cpHS[i,3] = (a[i,9] * log(T) + a[i,10] * T + a[i,11] * T^2 /2 + a[i,12] * T^3 /3 + a[i,13] * T^4 / 4 + a[i,15]) * R        
        end
    end
    return cpHS
end

function compute_falloff(T, pr, a)
    fcent = (1 - a[1]) * exp(-T/a[2]) + a[1] * exp(-T/a[3]) + exp(-a[4]/T)
    c = -0.4 - 0.67 * log(10, fcent)
    n = 0.75 - 1.27 * log(10, fcent)
    f1 = (log(10, pr) + c) / (n - 0.14 *(log(10, pr) + c))
    return  10 ^ (log(10, fcent) / (1 + f1 ^ 2))
end

function compute_m_dot(m_dot_in, m_dot_out)
    return m_dot_in - m_dot_in
end

function get_W_bar(W_k, Y_k)
    inv_W_bar = sum(W_k .* Y_k)
    return 1. / inv_W_bar 
end

function get_Cp_bar(T, Y_k)
    cpHS = compute_cpHS(NASA_coeffs, R, T)
    cp_k = cpHS[:,1]
    return sum(cp_k .* Y_k)
end

function get_Cv_bar(T, Y_k)
    return get_Cp_bar(T, Y_k) - R
end

function get_ρ_from_PT(P, T, Y_k, W_k)
    W_bar = get_W_bar(W_k, Y_k)
    ρ = (P * W_bar) / (R * T)
    return ρ
end

function compute_enthaply(NASA_coeffs, R, T, Y)
    cpHS = compute_cpHS(NASA_coeffs, R, T)
    h = cpHS[:,2]
    return sum(h .* Y_in)
end

function compute_internal_energy(NASA_coeffs, R, T, Y)
    h = compute_enthaply(NASA_coeffs, R, T, Y)
    u = h ./ W_k - R ./ W_k .* T   # Internal energy for species
    return u
end

function compute_ω_dot(ρ, T, Y)
    X = ρ * Y ./ W_k 
    W_bar = get_W_bar(W_k, Y)
    Q = ones(M) # Individual progress rates
    cpHS = compute_cpHS(NASA_coeffs, R, T)
    cp_k = cpHS[:,1]
    cp_bar = cp_k'*Y
    h = cpHS[:,2]
    s = cpHS[:,3]
    cv_k = cp_k .- R
    ΔS = ν' * s  # Entropy change for reaction j
    ΔH = ν' * h # Entahlpy change for reaction j
    ####
    M_t = efficiency_t' * X
    Kf_t = Af_t .* (T .^ β_t) .* exp.(-E_t ./ (R * T)) .* M_t
    Kr_t = Kf_t ./ (((pa/(R * T)) .^ sum(ν[:,tbd], dims=1)' .* exp.(ΔS[tbd] ./ R - ΔH[tbd] ./ (R * T))))
    Q[tbd] = Kf_t .* prod(X .^ order_t, dims=1)' .- Kr_t .* prod(X .^ ν2[:,tbd], dims=1)' .* reversible[tbd]
    ####
    M_f = efficiency_f' * X
    Kf_lo = Af_lo .* (T .^ β_lo) .* exp.(-E_lo ./ (R * T)) .* M_f
    Kf_hi = Af_hi .* (T .^ β_hi) .* exp.(-E_hi ./ (R * T)) 
    Pr = Kf_lo ./ Kf_hi
    Fac = ones(size(falofr)[1])
    Fac[troefall] = [compute_falloff(T, Pr[s], troefall_coeff[:,i]) for (i,s) in enumerate(troefall)]
    Kf_f = Kf_lo ./ (1 .+ (Kf_lo ./ Kf_hi)) .* Fac
    Kr_f = Kf_f ./ (((pa/(R * T)) .^ sum(ν[:,falofr], dims=1)' .* exp.(ΔS[falofr] ./ R - ΔH[falofr] ./ (R * T))))
    Q[falofr] = Kf_f .* prod(X .^ order_f, dims=1)' .- Kr_f .* prod(X .^ ν2[:,falofr], dims=1)' .* reversible[falofr]
    ####
    Kf = Af .* (T .^ β) .* exp.(-E ./ (R * T))
    Kr = Kf ./ (((pa/(R * T)) .^ sum(ν[:,elmr], dims=1)' .* exp.(ΔS[elmr] ./ R - ΔH[elmr] ./ (R * T))))
    Q[elmr] = Kf .* prod(X .^ order, dims=1)' .- Kr .* prod(X .^ ν2[:,elmr], dims=1)' .* reversible[elmr]
    ##### Computing ω_dot 
    cv_bar = sum(cv_k ./ W_k .* Y) # Mass heat capacities, dot product with mass fractions Y to compute total cv_bar
    u =  compute_internal_energy(NASA_coeffs, R, T, Y)
    # p = sum(X) * R * T # pressure
    p = ρ * R * T / W_bar # pressure
    ω_dot = W_k .* sum(ν .* Q', dims=2)
    ###### Species Conservation
    return ω_dot
end

function get_Q_dot(t)
    # Careful, Q_dot is the heat flux LEAVING the domain
    # The source term should be negative
    # Center time [s]
    t0 = 1e-2
    # Amplitude 
    # amplitude = 1e9
    amplitude = 1e10
    # full width at half maximum
    fwhm = 1e-3
    return -amplitude * exp(-(t - t0)^2 * 4 * log(2) / fwhm^2)
end

# --------------------------------------------
# Input values

# --- geom ---
# [m]
length_chamber = 20e-2
# [m]
radius_chamber = 4e-2
# Volume chamber [m^3]
const V_ch = 2 * π * radius_chamber  

# [m]
radius_nozzle = 0.025 
# Nozzle throat nozzle area [m^2]
const A_nozzle = 2 * π * radius_nozzle  

# --- thermo ---
const R = 8314.4621 # Gas constant in kmol
# NASA_coeffs = [1000.0 3.66096065 0.000656365811 -1.41149627e-7 2.05797935e-11 -1.29913436e-15 -1215.97718 3.41536279 3.78245636 -0.00299673416 9.84730201e-6 -9.68129509e-9 3.24372837e-12 -1063.94356 3.65767573; 1000.0 2.6770389 0.0029731816 -7.7376889e-7 9.4433514e-11 -4.2689991e-15 -29885.894 6.88255 4.1986352 -0.0020364017 6.5203416e-6 -5.4879269e-9 1.771968e-12 -30293.726 -0.84900901; 1000.0 1.911786 0.0096026796 -3.38387841e-6 5.3879724e-10 -3.19306807e-14 -10099.2136 8.48241861 5.14825732 -0.013700241 4.93749414e-5 -4.91952339e-8 1.70097299e-11 -10245.3222 -4.63322726; 1000.0 3.0484859 0.0013517281 -4.8579405e-7 7.8853644e-11 -4.6980746e-15 -14266.117 6.0170977 3.5795335 -0.00061035369 1.0168143e-6 9.0700586e-10 -9.0442449e-13 -14344.086 3.5084093; 1000.0 4.6365111 0.0027414569 -9.9589759e-7 1.6038666e-10 -9.1619857e-15 -49024.904 -1.9348955 2.356813 0.0089841299 -7.1220632e-6 2.4573008e-9 -1.4288548e-13 -48371.971 9.9009035; 1000.0 2.95257637 0.0013969004 -4.92631603e-7 7.86010195e-11 -4.60755204e-15 -923.948688 5.87188762 3.53100528 -0.000123660988 -5.02999433e-7 2.43530612e-9 -1.40881235e-12 -1046.97628 2.96747038]
# W_k = [31.9988, 18.01528, 16.04276, 28.010399999999997, 44.0098, 28.01348]

const NASA_coeffs = py"np.load"(".npy files/NASA_coeffs.npy")
const W_k = py"np.load"(".npy files/molecular_weights.npy") # Molar weight
const ν1 = py"np.load"(".npy files/reactants_stoich_coeffs.npy") # Forward molar stoichiometric coefficients
const ν2 = py"np.load"(".npy files/product_stoich_coeffs.npy") # Backward model stoichiometric coefficients
const reversible = py"np.load"(".npy files/reversible.npy")
const N = size(ν1)[1]  # Number of Species
const N_species =  N
const M = size(ν1)[2]  # Number of Reactions
ν1_order = zeros(N,M)
const ν = ν2 - ν1  
const pa = 100000 # 1 bar

### Quantities that depend on the Unknowns
const tbd = py"np.load"(".npy files/tbd.npy") .+ 1
const falofr = py"np.load"(".npy files/falofr.npy") .+ 1
const elmr = py"np.load"(".npy files/elmr.npy") .+ 1

const order = py"np.load"(".npy files/reaction_orders.npy")
const Af = py"np.load"(".npy files/pre_exponential_factor.npy") # preexponential constant Afj
const β = py"np.load"(".npy files/temperature_exponent.npy") # Temperature exponent
const E = py"np.load"(".npy files/activation_energy.npy") # Activation energy for the reactions in kJ

const order_t = py"np.load"(".npy files/reaction_orders_t.npy")
const efficiency_t = py"np.load"(".npy files/efficiency_t.npy")
const Af_t = py"np.load"(".npy files/pre_exponential_factor_t.npy") # preexponential constant Afj
const β_t = py"np.load"(".npy files/temperature_exponent_t.npy") # Temperature exponent
const E_t = py"np.load"(".npy files/activation_energy_t.npy") # Activation energy for the reactions in kJ

const order_f = py"np.load"(".npy files/reaction_orders_f.npy")
const troefall = py"np.load"(".npy files/troefall.npy") .+ 1
const troefall_coeff = py"np.load"(".npy files/troefall_coeff.npy")
const efficiency_f = py"np.load"(".npy files/efficiency_f.npy")
const Af_hi = py"np.load"(".npy files/pre_exponential_factor_hi.npy") # preexponential constant Afj
const β_hi = py"np.load"(".npy files/temperature_exponent_hi.npy") # Temperature exponent
const E_hi = py"np.load"(".npy files/activation_energy_hi.npy") 

const Af_lo = py"np.load"(".npy files/pre_exponential_factor_lo.npy") # preexponential constant Afj
const β_lo = py"np.load"(".npy files/temperature_exponent_lo.npy") # Temperature exponent
const E_lo = py"np.load"(".npy files/activation_energy_lo.npy") 

ν1_order[:,elmr] = order
ν1_order[:,tbd] = order_t
ν1_order[:,falofr] = order_f

println("Start")
# --- Inlet parameters ---
# [Kg/s],  Rate at which mass enters the chamber
const m_dot_in = 5.0

Y_in = zeros(N_species) # Mass fraction of species entering the chamber
# Y_O2
Y_in[1] = 0.21 
# Y_CH4
Y_in[3] = 0.07 
# Y_N2
Y_in[6] = 1 - Y_in[1] - Y_in[3]
println("Sum yk in = ", sum(Y_in))

# Enthalpy init
const T_in = 1000.0 # [K]
h_in = compute_enthaply(NASA_coeffs, R, T_in, Y_in)

# --- Inlet parameters ---
const P_out = 101325.0 # [Pa]

# --- Init ---
#species = """ O2 H2O CH4 CO CO2 N2 """,
Y_init = zeros(N_species) # Mass fraction of species entering the chamber
# Y_O2
Y_init[1] = 0.21 
# Y_N2
Y_init[6] = 1.0 - Y_init[1] 
println("Sum yk init = ", sum(Y_init))

const T_init = 300 # [K]
P_init = P_out # [Pa]
ρ_init = get_ρ_from_PT(P_init, T_init, Y_init, W_k)
m_init = ρ_init * V_ch


function compute_m_dot_in()
    # Defined to be a constant...for now
    return m_dot_in
end

function compute_m_dot_out(ρ_ch, P_ch, T_ch, A_nozzle, cp_bar, cv_bar)
    # Compute the efflux of the combustor via simple isentropic relations.
    # The efficiency of the nozzle is modeled using a discharge coefficient
    # :param t: time
    # :return: mass flux at the outlet (efflux) of the combustion chamber.
    # Avoid a unity pressure ratio at startup

    # Arbitrary small value to avoid numerical issue when P_ch = P_out
    ϵ = 1.0
    γ_s = cp_bar / cv_bar
    r_gas_specific = cp_bar - cv_bar
    P_ratio = P_out / (P_ch + ϵ)

    power_1 = 2.0 / γ_s
    power_2 = (γ_s + 1.0) / γ_s
    pressure_term = P_ratio^power_1 - P_ratio^power_2
    _sqrt_term = 2. * γ_s * r_gas_specific * T_ch / (γ_s - 1.0)
    _sqrt_term *= pressure_term
    _sqrt_term = sqrt(_sqrt_term)

    _mdot_unchoked = ρ_ch * A_nozzle * _sqrt_term

    power = (γ_s + 1.0) / (γ_s - 1.0)
    _gamma_term = (2. / (γ_s + 1.0))^power
    _sqrt = sqrt(γ_s * r_gas_specific * T_ch * _gamma_term)
    _mdot_choked = ρ_ch * A_nozzle * _sqrt

    _p_crit_downstream = P_ch * (2. / (γ_s + 1.0))^(γ_s / (γ_s - 1.0))

    if _p_crit_downstream < P_out
        _mdot = _mdot_unchoked
    else
        _mdot = _mdot_choked
    end

    return _mdot
end

function rk_substep(time, state_vec)
    m = state_vec[1]
    T = state_vec[2]
    Y = state_vec[3:end]

    Q_dot = get_Q_dot(time)

    ρ = m / V_ch 
    W_bar = get_W_bar(W_k, Y)
    P = ρ * R / W_bar * T

    cp_bar = get_Cp_bar(T, Y)
    cv_bar = get_Cv_bar(T, Y)
    u =  compute_internal_energy(NASA_coeffs, R, T, Y)

    # Mass source term (kinetics)
    ω_dot = compute_ω_dot(ρ, T, Y)
    # ω_dot = 0. * Y
    mgen_dot = V_ch .* ω_dot
    # println("mgen_dot", mgen_dot)

    # Mass flow rates 
    m_dot_in = compute_m_dot_in()
    m_dot_out = compute_m_dot_out(ρ, P, T, A_nozzle, cp_bar, cv_bar)

    # Mass conservation
    m_dot = m_dot_in - m_dot_out

    # Species conservation
    # Careful! Yizhou used the wrong expression here
    Y_dot = (1 / m) .* (m_dot_in .* (Y_in .- Y) .+ mgen_dot) 

    # Energy conservation
    T_dot = 1 / (m * cv_bar) * (-Q_dot + m_dot_in * (h_in - sum(u .* Y_in)) - P * V_ch / m * m_dot_out - sum(mgen_dot .* u))
    
    return [m_dot; T_dot; Y_dot]
end

function store_state(state)
    m = state[1]
    T = state[2]
    Y = state[3:end]
    W_bar = get_W_bar(W_k, Y)
    P = (m / V_ch) * R * T / W_bar
    return m, T, Y, P
end

function RK_loop()
    ## RK4
    global scale
    global dt
    dt = 1e-5
    n_t = 500000
    # n_t = 25
    state_vec = [m_init; T_init; Y_init]
    m_t = zeros(n_t)
    T_t = zeros(n_t)
    time_t = zeros(n_t)
    P_t = zeros(n_t)
    Q_dot_t = zeros(n_t)
    Y_t = zeros(N_species, n_t)
    time = 0.0
    time_t[1] = time
    m_t[1], T_t[1], Y_t[:,1], P_t[1] = store_state(state_vec)
    Q_dot_t[1] = 0.0
    dt_t = zeros(n_t)
    scale_t = zeros(n_t)
    dt_t[1] = dt
    scale_t[1] = 1

    # Getting RKF coefficients
    A_rk, B_rk, Ck_rk, CH_rk, CT_rk = RKF.coeff_RKF()
    tolerance = 1e-1 #* ones(size(state_vec))

    for idx = 2:n_t
        n_attemps = 1
        if mod(idx, 1000) == 0
            println("timestep ", idx)
        end
        Q_dot = get_Q_dot(time)
        
        if ADAPTIVE_TIMESTEP
            while n_attemps < MAX_ATTEMPTS
                # println("> n_attemps: ", n_attemps)
                n_attemps += 1
                state_vec_0 = copy(state_vec)

                # RKF45
                k1 = rk_substep(time + A_rk[1] * dt,
                               state_vec
                               )
                k2 = rk_substep(time + A_rk[2] * dt,
                               state_vec 
                               + B_rk[2, 1] * k1 * dt
                               )
                k3 = rk_substep(time + A_rk[3] * dt,
                               state_vec 
                               + B_rk[3, 1] * k1 * dt
                               + B_rk[3, 2] * k2 * dt
                               )
                k4 = rk_substep(time + A_rk[4] * dt,
                               state_vec 
                               + B_rk[4, 1] * k1 * dt
                               + B_rk[4, 2] * k2 * dt
                               + B_rk[4, 3] * k3 * dt
                               )
                k5 = rk_substep(time + A_rk[5] * dt,
                               state_vec 
                               + B_rk[5, 1] * k1 * dt
                               + B_rk[5, 2] * k2 * dt
                               + B_rk[5, 3] * k3 * dt
                               + B_rk[5, 4] * k4 * dt
                               )

                k6 = rk_substep(time + A_rk[6] * dt,
                                    state_vec 
                                    + B_rk[6, 1] * k1 * dt
                                    + B_rk[6, 2] * k2 * dt
                                    + B_rk[6, 3] * k3 * dt
                                    + B_rk[6, 4] * k4 * dt
                                    + B_rk[6, 5] * k5 * dt
                                    )

                state_vec_out = state_vec_0 + dt .* (CH_rk[1] * k1 + CH_rk[2] * k2 + CH_rk[3] * k3 + CH_rk[4] * k4 + CH_rk[5] * k5 + CH_rk[6] * k6)

                step_rk4 = dt * (Ck_rk[1] * k1 + Ck_rk[2] * k2 + Ck_rk[3] * k3 + Ck_rk[4] * k4 + Ck_rk[5] * k5 + Ck_rk[6] * k6)
                # truncation error
                step_rk5 = dt * (CT_rk[1] * k1 + CT_rk[2] * k2 + CT_rk[3] * k3 + CT_rk[4] * k4 + CT_rk[5] * k5 + CT_rk[6] * k6)

                err = sqrt(sum((step_rk4 .- step_rk5).^2)/8)

                # Both are used, depends if we want to be on the safe or efficient side of the force...
                scale = 0.8 .* (tolerance/err).^(1/7)
                # scale = 0.9 .* (tolerance/err).^(1/5)

                dt *= scale
                if err < tolerance
                    state_vec = copy(state_vec_out)
                    # println("> error is sufficiently low, update state and break")
                    break
                end

                if dt < 1e-11
                    println("\t>Error: timestep too small")
                    break
                end
            end
        else
            # RK4
            k1 = rk_substep(time, state_vec)
            k2 = rk_substep(time + 0.5 * dt, state_vec + 0.5 * dt * k1)
            k3 = rk_substep(time + 0.5 * dt, state_vec + 0.5 * dt * k2)
            k4 = rk_substep(time + dt, state_vec + dt * k3)
            state_vec += 1/6 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
        end
        
        time += dt
        time_t[idx] = time
        dt_t[idx] = dt
        Q_dot_t[idx] = Q_dot
        scale_t[idx] = scale
        m_t[idx], T_t[idx], Y_t[:,idx], P_t[idx] = store_state(state_vec)
    end

    println("return data from RK loop")
    return time_t, m_t, T_t, Y_t, P_t, Q_dot_t, dt_t, scale_t
end

println("Do RK4 loop")
time_t, m_t, T_t, Y_t, P_t, Q_dot_t, dt_t, scale_t = RK_loop()
ρ_t = m_t / V_ch

plotly()
# gr()
println("Plot PTMY")
p1 = plot(time_t, P_t,
        #   marker=:none, markevery=10, 
          ylabel = "Pressure [Pa]", legend = false) # Make a line plot
p2 = plot(time_t, T_t,
        #   marker=:none, markevery=10, 
          ylabel = "Temperature [K]", legend = false) # Make a line plot
p3 = plot(time_t, ρ_t,
        #   marker=:none, markevery=10, 
          ylabel = "Density [kg/m^3]", xlabel="Time [s]", legend = false) # Make a line plot

          # " O2  H2O CH4 CO CO2 N2 "
p4 = plot(time_t, Y_t[1,:], ylabel = "Y [-]", xlabel="Time [s]", label="O2") # Make a line plot
plot!(time_t, Y_t[2,:], label="H2O")
plot!(time_t, Y_t[3,:], label="CH4")
plot!(time_t, Y_t[4,:], label="CO")
plot!(time_t, Y_t[5,:], label="CO2")
plot!(time_t, Y_t[6,:], legend = true, label="N2")

sum_yk = zeros(size(T_t))
for i=1:6
    sum_yk[:] += Y_t[i, :]
end
plot!(time_t, sum_yk, legend = true, label="sum")
plot_size = (2*400, 2*300)
plot_PTMY = plot(p1, p2, p3, p4, layout = (2, 2), size=plot_size)
display(plot_PTMY)

plot_dt_t = plot(time_t, dt_t, xlabel='t', ylabel="dt", marker=:o, yaxis=:log, size=plot_size)
display(plot_dt_t)

plot_scale_t = plot(time_t, scale_t, xlabel='t', ylabel="scale factor of dt", marker=:o, size =plot_size)
display(plot_scale_t)