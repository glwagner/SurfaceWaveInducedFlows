using Oceananigans
using GLMakie
using Printf

f = 0
N² = 0

for f = [0] #1e-6, 1e-5, 1e-4]
    for N² = [1e-4, 1e-3]

        prefix = @sprintf("surface_wave_induced_flow_f%04d_Nsq%04d", 1e6 * f, 1e6 * N²) 

        uxyt = FieldTimeSeries(prefix * "_xy.jld2", "u")
        wxyt = FieldTimeSeries(prefix * "_xy.jld2", "w")
        uxzt = FieldTimeSeries(prefix * "_xz.jld2", "u")
        wxzt = FieldTimeSeries(prefix * "_xz.jld2", "w")

        grid = uxyt.grid
        times = uxyt.times
        Nt = length(times)
        Nx, Ny, Nz = size(grid)

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

        fig = Figure(size=(1500, 800))

        axuxy = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)", aspect=2, title="u")
        axwxy = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)", aspect=2, title="w")

        axuxz = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)", aspect=2, title="u")
        axwxz = Axis(fig[2, 2], xlabel="x (m)", ylabel="z (m)", aspect=2, title="w")

        ulim = maximum(abs, uxyt)
        wlim = maximum(abs, wxyt)

        heatmap!(axuxy, uxyn, colormap=:balance, colorrange=(-ulim, ulim))
        contour!(axuxy, xa, ya, An, color=:gray, levels=5)

        heatmap!(axwxy, wxyn, colormap=:balance, colorrange=(-wlim, wlim))
        contour!(axwxy, xa, ya, An, color=:gray, levels=5)

        heatmap!(axuxz, uxzn, colormap=:balance, colorrange=(-ulim, ulim))
        heatmap!(axwxz, wxzn, colormap=:balance, colorrange=(-2wlim, 2wlim))

        #display(fig)

        record(fig, prefix * ".mp4", 1:Nt, framerate=24) do nn
            @info "Drawing frame $nn of $Nt..."
            n[] = nn
        end
        
    end
end
