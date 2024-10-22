using Oceananigans
using Oceananigans.Units
using Printf

ϵ = 0.3
λ = 100 # meters
g = 9.81
f = 0 #e-4
N² = 1e-3
prefix = @sprintf("surface_wave_induced_flow_f%04d_Nsq%04d", 1e6 * f, 1e6 * N²) 

const k = 2π / λ

c = sqrt(g / k)
const δ = 400kilometer
const cᵍ = c / 2
const Uˢ = ϵ^2 * c

@show Ro = Uˢ / (f * δ)

@inline       A(ξ, η) = exp(- (ξ^2 + η^2) / 2δ^2)
@inline    ∂ξ_A(ξ, η) = - ξ / δ^2 * A(ξ, η)
@inline    ∂η_A(ξ, η) = - η / δ^2 * A(ξ, η)
@inline ∂η_∂ξ_A(ξ, η) = η * ξ / δ^4 * A(ξ, η)
@inline   ∂²ξ_A(ξ, η) = (ξ^2 / δ^2 - 1) * A(ξ, η) / δ^2

# Write the Stokes drift as
#
# uˢ(x, y, z, t) = A(x, y, t) * ûˢ(z)
#
# which implies

@inline    ûˢ(z)          = Uˢ * exp(2k * z)
@inline    uˢ(x, y, z, t) =           A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂z_uˢ(x, y, z, t) =     2k *  A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂y_uˢ(x, y, z, t) =        ∂η_A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂t_uˢ(x, y, z, t) = - cᵍ * ∂ξ_A(x - cᵍ * t, y) * ûˢ(z)

# where we have noted that η = y, and ξ = x - cᵍ t,
# such that ∂ξ/∂x = 1 and ∂ξ/∂t = - cᵍ.
#
# Note that if uˢ represents the solenoidal component of the Stokes drift,
# then
#
# ```math
# ∂z_wˢ = - ∂x_uˢ = - ∂ξ_A * ∂ξ/∂x * ûˢ .
#                 = - ∂ξ_A * ûˢ .
# ```
#
# We therefore find that
#
# ```math
# wˢ = - ∂ξ_A / 2k * ûˢ .
# ```
#
# and

@inline ∂x_wˢ(x, y, z, t) = -  1 / 2k *   ∂²ξ_A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂y_wˢ(x, y, z, t) = -  1 / 2k * ∂η_∂ξ_A(x - cᵍ * t, y) * ûˢ(z)
@inline ∂t_wˢ(x, y, z, t) = + cᵍ / 2k *   ∂²ξ_A(x - cᵍ * t, y) * ûˢ(z)

stokes_drift = StokesDrift(; ∂z_uˢ, ∂t_uˢ, ∂y_uˢ, ∂t_wˢ, ∂x_wˢ, ∂y_wˢ)

grid = RectilinearGrid(size = (128, 64, 16),
                       x = (-10δ, 30δ),
                       y = (-10δ, 10δ),
                       z = (-512, 0),
                       topology = (Periodic, Periodic, Bounded))

model = NonhydrostaticModel(; grid, stokes_drift,
                            tracers = :b,
                            coriolis = FPlane(; f),
                            buoyancy = BuoyancyTracer(),
                            timestepper = :RungeKutta3)

# Set Lagrangian-mean flow equal to uˢ,
uᵢ(x, y, z) = uˢ(x, y, z, 0)

# And put in a stable stratification,
bᵢ(x, y, z) = N² * z
set!(model, u=uᵢ, b=bᵢ)

Δx = minimum_xspacing(grid)
Δt = 0.2 * Δx / cᵍ
simulation = Simulation(model; Δt, stop_iteration = 200)

progress(sim) = @info string("Iter: ", iteration(sim), ", time: ", prettytime(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

outputs = model.velocities

u, v, w = model.velocities
e = @at (Center, Center, Center) (u^2 + v^2 + w^2) / 2
outputs = merge(outputs, (; e))
Nx, Ny, Nz = size(grid)

simulation.output_writers[:xy] = JLD2OutputWriter(model, outputs;
                                                  filename = prefix * "_xy",
                                                  schedule = IterationInterval(3),
                                                  indices = (:, :, Nz),
                                                  overwrite_existing = true)

j = round(Int, Ny/2)
simulation.output_writers[:xz] = JLD2OutputWriter(model, outputs;
                                                  filename = prefix * "_xz",
                                                  schedule = IterationInterval(3),
                                                  indices = (:, j, :),
                                                  overwrite_existing = true)

run!(simulation)

#####
##### Visualization
#####

using GLMakie

uxyt = FieldTimeSeries(prefix * "_xy.jld2", "u")
wxyt = FieldTimeSeries(prefix * "_xy.jld2", "w")
uxzt = FieldTimeSeries(prefix * "_xz.jld2", "u")
wxzt = FieldTimeSeries(prefix * "_xz.jld2", "w")

grid = uxyt.grid
times = uxyt.times
Nt = length(times)
Nz = size(grid, 3)

n = Observable(1)

uxyn = @lift uxyt[$n]
wxyn = @lift wxyt[$n]
uxzn = @lift uxzt[$n]
wxzn = @lift wxzt[$n]

xa, ya, za = nodes(grid, Center(), Center(), Center())
xa = reshape(xa, Nx, 1)
ya = reshape(ya, 1, Ny)

An = @lift begin
    t = times[$n]
    ξ = @. xa - cᵍ * t
    A.(ξ, ya)
end

fig = Figure(size=(1600, 800))

axuxy = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)", aspect=2, title="u")
axwxy = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)", aspect=2, title="w")

axuxz = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)", aspect=2, title="u")
axwxz = Axis(fig[2, 2], xlabel="x (m)", ylabel="z (m)", aspect=2, title="w")

ulim = 1e-2 * Uˢ
elim = ulim^2
wlim = 1e-1 * Uˢ / (δ * 2k)

heatmap!(axuxy, uxyn, colormap=:balance, colorrange=(-ulim, ulim))
contour!(axuxy, xa, ya, An, color=:gray, levels=5)

heatmap!(axwxy, wxyn, colormap=:balance, colorrange=(-wlim, wlim))
contour!(axwxy, xa, ya, An, color=:gray, levels=5)

heatmap!(axuxz, uxzn, colormap=:balance, colorrange=(-ulim, ulim))
heatmap!(axwxz, wxzn, colormap=:balance, colorrange=(-wlim, wlim))

record(fig, prefix * ".mp4", 1:Nt, framerate=8) do nn
    n[] = nn
end

