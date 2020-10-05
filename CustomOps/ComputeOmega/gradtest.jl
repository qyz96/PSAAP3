using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

libspaap3 = tf.load_op_library("../build/libpsaap3.so")

function set_global_data()
    lib = "../build/libpsaap3.so"
    @eval ccall((:TestSetValues, $lib), Cvoid, ())
end


function run_this_function(Y, T)
    lib = "../build/libpsaap3.so"
    ω = zeros(1)
    dω = zeros(2)
    @eval ccall((:forward_ComputeOmega_Julia, $lib), Cvoid, (Ptr{Cdouble}, 
                Ptr{Cdouble}, Ptr{Cdouble}, Cdouble), $ω, $dω, $Y, $T)
    return ω, Array(reshape(dω, 2, 1)')
end


function compute_omega(temp,y)
    compute_omega_ = load_op_and_grad(libspaap3,"compute_omega", multiple=true)
    temp,y = convert_to_tensor(Any[temp,y], [Float64,Float64])
    ω, dω = compute_omega_(temp,y)
    set_shape(ω, 1), set_shape(dω, (1, 2))
end

set_global_data()

temp = 1.0
y = rand(1)
# TODO: specify your input parameters
u = compute_omega(temp,y)
sess = Session(); init(sess)
@show run(sess, u)

# uncomment it for testing gradients
# error() 


# TODO: change your test parameter to `m`
#       in the case of `multiple=true`, you also need to specify which component you are testings
# gradient check -- v
function scalar_function(x)
    return sum(compute_omega(temp,x)[1]^2)
end

# TODO: change `m_` and `v_` to appropriate values
m_ = constant(rand(1))
v_ = rand(1)
y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(v_.*dy_)
end

sess = Session(); init(sess)
sval_ = run(sess, s_)
wval_ = run(sess, w_)
close("all")
loglog(gs_, abs.(sval_), "*-", label="finite difference")
loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
savefig("gradtest.png")