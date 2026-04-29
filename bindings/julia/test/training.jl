using MarsRuntime
using Test

fitted_spec = fit_model([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0]; max_terms = 5, max_degree = 1, penalty = 3.0)
@test haskey(fitted_spec, "basis_terms")
@test !isempty(fitted_spec["basis_terms"])
replay = predict_model(fitted_spec, [[0.0], [1.0], [2.0]])
@test length(replay) == 3
@test isapprox(replay[1], 1.0; atol = 1e-12, rtol = 1e-12)
@test isapprox(replay[2], 3.0; atol = 1e-12, rtol = 1e-12)
@test isapprox(replay[3], 5.0; atol = 1e-12, rtol = 1e-12)
