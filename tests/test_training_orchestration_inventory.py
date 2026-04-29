"""Tests for Phase 0 Task 1: Inventory of Python training orchestration and Rust primitives."""

from pathlib import Path


def test_inventory_document_exists():
    """Test that the training orchestration inventory document exists."""
    inventory_path = Path("docs/training_orchestration_inventory.md")
    assert inventory_path.exists(), (
        "Inventory document not found at docs/training_orchestration_inventory.md"
    )


def test_inventory_has_python_forward_pass_mapping():
    """Test that Python forward-pass is mapped to Rust equivalents."""
    inventory_path = Path("docs/training_orchestration_inventory.md")
    content = inventory_path.read_text()

    required_sections = [
        "ForwardPasser",
        "generate_candidates",
        "find_best_candidate_addition",
        "calculate_rss_and_coeffs",
        "build_basis_matrix",
        "calculate_gcv_for_basis_set",
    ]
    for section in required_sections:
        assert section in content, f"Missing Python forward-pass mapping for: {section}"


def test_inventory_has_python_pruning_mapping():
    """Test that Python pruning logic is mapped to Rust equivalents."""
    inventory_path = Path("docs/training_orchestration_inventory.md")
    content = inventory_path.read_text()

    required_sections = [
        "PruningPasser",
        "compute_gcv_for_subset",
        "pruning RSS",
        "coefficient refit",
    ]
    for section in required_sections:
        assert section in content, f"Missing Python pruning mapping for: {section}"


def test_inventory_has_rust_primitive_documentation():
    """Test that Rust primitives are documented with API boundaries."""
    inventory_path = Path("docs/training_orchestration_inventory.md")
    content = inventory_path.read_text()

    required_sections = [
        "fit_least_squares",
        "score_candidate",
        "score_pruning_subset",
        "calculate_gcv",
        "effective_parameters",
    ]
    for section in required_sections:
        assert section in content, (
            f"Missing Rust primitive documentation for: {section}"
        )


def test_inventory_identifies_unsupported_edge_cases():
    """Test that unsupported edge cases are documented."""
    inventory_path = Path("docs/training_orchestration_inventory.md")
    content = inventory_path.read_text()

    assert "Unsupported" in content or "Edge Cases" in content, (
        "Missing documentation of unsupported edge cases"
    )
    assert "missing" in content.lower() or "NaN" in content, (
        "Missing documentation of missing value handling differences"
    )


def test_inventory_documents_rust_training_api_boundary():
    """Test that the selected Rust training API boundary is documented."""
    inventory_path = Path("docs/training_orchestration_inventory.md")
    content = inventory_path.read_text()

    assert "API Boundary" in content or "Training API" in content, (
        "Missing Rust training API boundary documentation"
    )
    assert "pub fn" in content or "pub struct" in content, (
        "Missing Rust public API documentation in inventory"
    )
