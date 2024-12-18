#=
using Oceananigans
using Oceananigans.Units
using GLMakie

ϵ = 0.1
λ = 60 # meters
g = 9.81

const k = 2π / λ

c = sqrt(g / k)
const δ = 500
const cᵍ = c / 2
const Uˢ = ϵ^2 * c

@inline A(ξ) = exp(- ξ^2 / 2δ^2)
@inline A′(ξ) = - ξ / δ^2 * A(ξ)
@inline A′′(ξ) = (ξ^2 / δ^2 - 1) * A(ξ) / δ^2

# Write the Stokes drift as
#
# uˢ(x, z, t) = A(x - cᵍ * t) * ûˢ(z)
#
# which describes a wave packet propogating with speed cᵍ. This implies

@inline    ûˢ(z)       = Uˢ * exp(2k * z)
@inline    uˢ(x, z, t) =         A(x - cᵍ * t) * ûˢ(z)
@inline ∂z_uˢ(x, z, t) =   2k *  A(x - cᵍ * t) * ûˢ(z)
@inline ∂t_uˢ(x, z, t) = - cᵍ * A′(x - cᵍ * t) * ûˢ(z)

# Note that if uˢ represents the solenoidal component of the Stokes drift,
# then
#
# ```math
# ∂z_wˢ = - ∂x_uˢ = - A′ * ûˢ .
# ```
#
# Thus, after integrating from bottom up to ``z`` and assuming that ``w`` at
# the bottom vanishes, we find that
#
# ```math
# wˢ = - A′ / 2k * ûˢ
# ```
#
# and

@inline ∂x_wˢ(x, z, t) = -  1 / 2k * A′′(x - cᵍ * t) * ûˢ(z)
@inline ∂t_wˢ(x, z, t) = + cᵍ / 2k * A′′(x - cᵍ * t) * ûˢ(z)
@inline    wˢ(x, z, t) = -  1 / 2k *  A′(x - cᵍ * t)  * ûˢ(z)

stokes_drift = StokesDrift(; uˢ, wˢ, ∂z_uˢ, ∂x_wˢ) #∂t_uˢ, ∂t_wˢ, ∂x_wˢ)
coriolis = FPlane(f=0.1)

grid = RectilinearGrid(size = (512, 1024),
                       x = (-2kilometers, 7kilometers),
                       z = (-4096, 0),
                       topology = (Periodic, Flat, Bounded))

#model = NonhydrostaticModel(; grid, stokes_drift, coriolis, tracers=:b, buoyancy=BuoyancyTracer())
model = NonhydrostaticModel(; grid, stokes_drift, tracers=:b) #, coriolis, buoyancy=BuoyancyTracer())

# Set Lagrangian-mean flow equal to uˢ,
uᵢ(x, z) = uˢ(x, z, 0)

# And put in a stable stratification,
N² = 0
bᵢ(x, z) = N² * z
set!(model, u=uᵢ, b=bᵢ)

Δx = minimum_xspacing(grid)
Δt = 0.2 * Δx / cᵍ
simulation = Simulation(model; Δt, stop_iteration = 1800)

progress(sim) = @info string("Iter: ", iteration(sim), ", time: ", prettytime(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

filename = "surface_wave_induced_flow.jld2"
outputs = model.velocities
simulation.output_writers[:jld2] = JLD2OutputWriter(model, outputs; filename,
                                                    schedule = IterationInterval(10),
                                                    overwrite_existing = true)

run!(simulation)
=#

ut = FieldTimeSeries(filename, "u")
wt = FieldTimeSeries(filename, "w")

times = ut.times
Nt = length(times)

fig = Figure(size=(800, 800))
axU = Axis(fig[1, 1], xlabel="x (m)", ylabel="U = H⁻¹ ∫ u dz (m s⁻¹)")
axu = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)")
axw = Axis(fig[3, 1], xlabel="x (m)", ylabel="z (m)")

#n = Observable(1)
slider = Slider(fig[4, 1], range=1:Nt, startvalue=1)
n = slider.value

rowsize!(fig.layout, 1, Relative(0.2))

u = XFaceField(grid)
U = Field(Average(u, dims=3))

Un = @lift begin
    parent(u) .= parent(ut[$n])
    compute!(U)
    U
end
    
un = @lift interior(ut[$n], :, 1, :)
wn = @lift interior(wt[$n], :, 1, :)

xu, yu, zu = nodes(ut)
xw, yw, zw = nodes(wt)

lines!(axU, xu, U)
heatmap!(axu, xu, zu, un, colormap=:balance, colorrange=(-5e-6, 5e-6))
heatmap!(axw, xw, zw, wn, colormap=:balance, colorrange=(-5e-6, 5e-6))

display(fig)

record(fig, "surface_wave_induced_flow.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end

