using MarsRuntime
using JSON
using Test

fixtures_dir = joinpath(@__DIR__, "..", "..", "..", "tests", "fixtures")
model_specs = sort(filter(name -> startswith(name, "model_spec_") && endswith(name, ".json"), readdir(fixtures_dir)))

@test !isempty(model_specs)

for file in model_specs
    suffix = replace(replace(file, "model_spec_" => ""), ".json" => "")
    spec = load_model_spec(joinpath(fixtures_dir, file))
    fixture = JSON.parsefile(joinpath(fixtures_dir, "runtime_portability_fixture_$suffix.json"))
    probe = [[value === nothing ? NaN : Float64(value) for value in row] for row in fixture["probe"]]
    actual_matrix = design_matrix(spec, probe)
    expected_matrix = [[value === nothing ? NaN : Float64(value) for value in row] for row in fixture["design_matrix"]]
    for row_idx in eachindex(expected_matrix)
        for col_idx in eachindex(expected_matrix[row_idx])
            if isnan(expected_matrix[row_idx][col_idx])
                @test isnan(actual_matrix[row_idx][col_idx])
            else
                @test isapprox(actual_matrix[row_idx][col_idx], expected_matrix[row_idx][col_idx]; atol = 1e-12, rtol = 1e-12)
            end
        end
    end
    actual = predict_model(spec, probe)
    expected = [value === nothing ? NaN : Float64(value) for value in fixture["predict"]]
    for idx in eachindex(expected)
        if isnan(expected[idx])
            @test isnan(actual[idx])
        else
            @test isapprox(actual[idx], expected[idx]; atol = 1e-12, rtol = 1e-12)
        end
    end
end

fitted_spec = fit_model([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0]; max_terms = 5, max_degree = 1, penalty = 3.0)
@test haskey(fitted_spec, "basis_terms")
@test !isempty(fitted_spec["basis_terms"])
replay = predict_model(fitted_spec, [[0.0], [1.0], [2.0]])
@test length(replay) == 3
@test isapprox(replay[1], 1.0; atol = 1e-12, rtol = 1e-12)
@test isapprox(replay[2], 3.0; atol = 1e-12, rtol = 1e-12)
@test isapprox(replay[3], 5.0; atol = 1e-12, rtol = 1e-12)
