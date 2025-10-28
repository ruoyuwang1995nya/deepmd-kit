#!/usr/bin/env python3
"""
Test script demonstrating gradient computation for property models in DeePMD-kit.

This script shows how to:
1. Use the new PropertyFittingNetWithGrad class
2. Use the utility function to compute gradients for existing PropertyFittingNet
3. Compare results and verify gradient computation
"""

import torch
import numpy as np
from deepmd.pt.model.task.property import PropertyFittingNet
from deepmd.pt.model.task.property_grad import (
    PropertyFittingNetWithGrad, 
    compute_property_gradients
)


def create_test_data(nframes=2, natoms=4, dim_descrpt=128):
    """Create synthetic test data for testing."""
    # Generate random test data
    torch.manual_seed(42)
    
    descriptor = torch.randn(nframes, natoms, dim_descrpt, requires_grad=True)
    atype = torch.randint(0, 2, (nframes, natoms))  # 2 atom types
    coord = torch.randn(nframes, natoms, 3, requires_grad=True)
    
    return descriptor, atype, coord


def test_property_gradient_computation():
    """Test gradient computation for property models."""
    print("Testing Property Model Gradient Computation")
    print("=" * 50)
    
    # Create test data
    nframes, natoms, dim_descrpt = 2, 4, 128
    descriptor, atype, coord = create_test_data(nframes, natoms, dim_descrpt)
    
    # Test parameters
    ntypes = 2
    property_name = "test_property"
    task_dim = 2  # Multi-dimensional property for testing
    
    print(f"Test setup:")
    print(f"  - Frames: {nframes}")
    print(f"  - Atoms per frame: {natoms}")
    print(f"  - Descriptor dimension: {dim_descrpt}")
    print(f"  - Property dimension: {task_dim}")
    print(f"  - Atom types: {ntypes}")
    print()
    
    # Test 1: Standard PropertyFittingNet (no gradients)
    print("1. Testing standard PropertyFittingNet:")
    standard_fitting = PropertyFittingNet(
        ntypes=ntypes,
        dim_descrpt=dim_descrpt,
        property_name=property_name,
        task_dim=task_dim
    )
    
    # Forward pass without gradients
    result_standard = standard_fitting.forward(descriptor, atype)
    property_values = result_standard[property_name]
    print(f"   Property shape: {property_values.shape}")
    print(f"   Property values (sample): {property_values[0, 0, :].detach().numpy()}")
    print()
    
    # Test 2: PropertyFittingNetWithGrad (gradient-enabled)
    print("2. Testing PropertyFittingNetWithGrad:")
    grad_fitting = PropertyFittingNetWithGrad(
        ntypes=ntypes,
        dim_descrpt=dim_descrpt,
        property_name=property_name,
        task_dim=task_dim,
        enable_grad=True
    )
    
    # Copy weights from standard fitting to ensure same predictions
    grad_fitting.load_state_dict(standard_fitting.state_dict())
    
    # Forward pass with gradients
    result_grad = grad_fitting.forward_with_grad(descriptor, atype, coord)
    property_values_grad = result_grad[property_name]
    property_gradients = result_grad[f"{property_name}_grad"]
    
    print(f"   Property shape: {property_values_grad.shape}")
    print(f"   Gradient shape: {property_gradients.shape}")
    print(f"   Property values (sample): {property_values_grad[0, 0, :].detach().numpy()}")
    print(f"   Gradient values (sample): {property_gradients[0, 0, 0, :].detach().numpy()}")
    print()
    
    # Test 3: Utility function for existing PropertyFittingNet
    print("3. Testing utility function with existing PropertyFittingNet:")
    result_utility = compute_property_gradients(
        standard_fitting, descriptor, atype, coord
    )
    property_values_util = result_utility[property_name]
    property_gradients_util = result_utility[f"{property_name}_grad"]
    
    print(f"   Property shape: {property_values_util.shape}")
    print(f"   Gradient shape: {property_gradients_util.shape}")
    print(f"   Property values (sample): {property_values_util[0, 0, :].detach().numpy()}")
    print(f"   Gradient values (sample): {property_gradients_util[0, 0, 0, :].detach().numpy()}")
    print()
    
    # Test 4: Verify consistency
    print("4. Verifying consistency between methods:")
    
    # Check if property values match
    prop_diff = torch.abs(property_values_grad - property_values_util).max()
    print(f"   Max property difference: {prop_diff.item():.2e}")
    
    # Check if gradients match
    grad_diff = torch.abs(property_gradients - property_gradients_util).max()
    print(f"   Max gradient difference: {grad_diff.item():.2e}")
    
    # Verify gradients are reasonable (non-zero and finite)
    grad_norm = torch.norm(property_gradients, dim=-1).mean()
    print(f"   Average gradient norm: {grad_norm.item():.6f}")
    
    all_finite = torch.all(torch.isfinite(property_gradients))
    print(f"   All gradients finite: {all_finite}")
    print()
    
    # Test 5: Gradient check using finite differences
    print("5. Finite difference gradient check:")
    eps = 1e-5
    
    # Perturb coordinates slightly
    coord_plus = coord.clone()
    coord_plus[0, 0, 0] += eps  # Perturb x-coordinate of first atom
    
    coord_minus = coord.clone()
    coord_minus[0, 0, 0] -= eps
    
    # Compute property at perturbed coordinates
    with torch.no_grad():
        result_plus = compute_property_gradients(standard_fitting, descriptor, atype, coord_plus)
        result_minus = compute_property_gradients(standard_fitting, descriptor, atype, coord_minus)
        
        prop_plus = torch.sum(result_plus[property_name][0, :, 0])  # Sum over atoms, first property component
        prop_minus = torch.sum(result_minus[property_name][0, :, 0])
        
        # Finite difference gradient
        fd_grad = (prop_plus - prop_minus) / (2 * eps)
        
        # Analytical gradient
        analytical_grad = property_gradients[0, 0, 0, 0]  # First atom, first property, x-component
        
        print(f"   Finite difference gradient: {fd_grad.item():.6f}")
        print(f"   Analytical gradient: {analytical_grad.item():.6f}")
        print(f"   Relative error: {abs(fd_grad - analytical_grad) / (abs(fd_grad) + 1e-10):.2e}")
    
    print()
    print("Test completed successfully!")


def demonstrate_use_cases():
    """Demonstrate practical use cases for property gradients."""
    print("\nPractical Use Cases for Property Gradients")
    print("=" * 50)
    
    # Create test data
    descriptor, atype, coord = create_test_data()
    
    # Example 1: HOMO/LUMO orbital energies and their gradients
    print("Example 1: HOMO energy gradients (for geometry optimization)")
    homo_fitting = PropertyFittingNetWithGrad(
        ntypes=2,
        dim_descrpt=128,
        property_name="homo",
        task_dim=1,
        enable_grad=True
    )
    
    result = homo_fitting.forward_with_grad(descriptor, atype, coord)
    homo_energy = result["homo"]
    homo_grad = result["homo_grad"]
    
    print(f"   HOMO energies: {homo_energy.squeeze().detach().numpy()}")
    print(f"   Gradient norm per atom: {torch.norm(homo_grad, dim=-1).squeeze().detach().numpy()}")
    print()
    
    # Example 2: Dipole moment gradients
    print("Example 2: Dipole moment gradients (for polarizability analysis)")
    dipole_fitting = PropertyFittingNetWithGrad(
        ntypes=2,
        dim_descrpt=128,
        property_name="dipole",
        task_dim=3,  # x, y, z components
        enable_grad=True
    )
    
    result = dipole_fitting.forward_with_grad(descriptor, atype, coord)
    dipole = result["dipole"]
    dipole_grad = result["dipole_grad"]
    
    print(f"   Dipole components shape: {dipole.shape}")
    print(f"   Dipole gradients shape: {dipole_grad.shape}")
    print(f"   Total dipole magnitude: {torch.norm(torch.sum(dipole, dim=1), dim=-1).detach().numpy()}")
    print()
    
    # Example 3: Using gradients for optimization
    print("Example 3: Using gradients for property-based optimization")
    print("   (Pseudocode for minimizing HOMO-LUMO gap)")
    print("""
    # Pseudo-optimization loop
    optimizer = torch.optim.Adam([coord], lr=0.01)
    
    for step in range(num_steps):
        homo_result = homo_fitting.forward_with_grad(descriptor, atype, coord)
        lumo_result = lumo_fitting.forward_with_grad(descriptor, atype, coord)
        
        gap = lumo_result["lumo"] - homo_result["homo"]
        loss = torch.sum(gap)  # Minimize HOMO-LUMO gap
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    """)


if __name__ == "__main__":
    # Run tests
    test_property_gradient_computation()
    
    # Show use cases
    demonstrate_use_cases()
