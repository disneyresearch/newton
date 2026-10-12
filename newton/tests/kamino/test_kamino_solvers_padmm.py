# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Proximal-ADMM Solver."""

import unittest
from typing import Any

import numpy as np
import warp as wp

from newton import ModelBuilder
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.dynamics.dual import DualProblem
from newton._src.solvers.kamino._src.kinematics.constraints import unpack_constraint_solutions
from newton._src.solvers.kamino._src.linalg import ConjugateResidualSolver, LLTBlockedSolver
from newton._src.solvers.kamino._src.linalg.utils.matrix import SquareSymmetricMatrixProperties
from newton._src.solvers.kamino._src.linalg.utils.range import in_range_via_gaussian_elimination
from newton._src.solvers.kamino._src.solvers.padmm import PADMMSolver, PADMMWarmStartMode
from newton._src.solvers.kamino._src.solvers.padmm.kernels import _update_spectral_penalty
from newton._src.solvers.kamino._src.solvers.padmm.math import (
    project_to_coulomb_cone,
    project_to_coulomb_dual_cone,
)
from newton._src.solvers.kamino._src.solvers.padmm.types import (
    PADMMConfigStruct,
    PADMMPenalty,
    PADMMStatus,
    convert_config_to_struct,
)
from newton._src.solvers.kamino._src.utils import logger as msg
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.extract import (
    extract_delassus,
    extract_info_vectors,
    extract_problem_vector,
)
from newton.tests.kamino.utils.make import make_containers, update_containers
from newton.tests.utils import basics

###
# Helper functions
###


class TestSetup:
    def __init__(
        self,
        builder_fn,
        max_world_contacts: int = 32,
        perturb: bool = True,
        gravity: bool = True,
        device: wp.DeviceLike = None,
        sparse: bool = False,
        **kwargs,
    ):
        # Cache the max contacts allocated for the test problem
        self.max_world_contacts = max_world_contacts

        # Construct the model description using model builders for different systems
        self.builder: ModelBuilder = builder_fn(**kwargs)

        # Set ad-hoc configurations
        if not gravity:
            for i in range(len(self.builder.world_gravity)):
                self.builder.world_gravity[i] = wp.vec3(0.0, 0.0, 0.0)
        if perturb:
            u_0 = wp.spatial_vectorf(10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            for i in range(len(self.builder.body_qd)):
                self.builder.body_qd[i] = u_0

        # Create the model and containers from the builder
        model = ModelKamino.from_newton(self.builder.finalize(device=device))
        self.model, self.data, self.state, self.limits, self.detector, self.jacobians = make_containers(
            model=model, max_world_contacts=max_world_contacts, sparse=sparse
        )
        self.contacts = self.detector.contacts
        self.state_p = self.model.state()

        # Create the DualProblem to be solved
        self.problem = DualProblem(
            model=self.model,
            data=self.data,
            limits=self.limits,
            contacts=self.contacts,
            jacobians=self.jacobians,
            solver=ConjugateResidualSolver if sparse else LLTBlockedSolver,
            sparse=sparse,
        )

        # Update the sim data containers
        update_containers(
            model=self.model,
            data=self.data,
            state=self.state,
            limits=self.limits,
            detector=self.detector,
            jacobians=self.jacobians,
        )

    def build(self):
        # Build the dual problem
        self.problem.build(
            model=self.model,
            data=self.data,
            limits=self.limits,
            contacts=self.contacts,
            jacobians=self.jacobians,
        )

    def cache(self, solver: PADMMSolver):
        # Unpack the computed constraint multipliers to the respective joint-limit
        # and contact data for post-processing and optional solver warm-starting
        unpack_constraint_solutions(
            lambdas=solver.data.solution.lambdas,
            v_plus=solver.data.solution.v_plus,
            model=self.model,
            data=self.data,
            limits=self.limits,
            contacts=self.contacts,
        )


def print_dual_problem_summary(D: np.ndarray, v_f: np.ndarray, notes: str = ""):
    D_props = SquareSymmetricMatrixProperties(D)
    v_f_is_in_range, *_ = in_range_via_gaussian_elimination(D, v_f)
    msg.info("Delassus Properties %s:\n%s\n", notes, D_props)
    msg.info("v_f is in range of D %s: %s\n", notes, v_f_is_in_range)


def check_padmm_solution(
    test: unittest.TestCase, model: ModelKamino, problem: DualProblem, solver: PADMMSolver, verbose: bool = False
):
    # Extract numpy arrays from the solver state and solution
    only_active_dims = True
    D_wp_np = extract_delassus(problem.delassus, only_active_dims=only_active_dims)
    v_f_wp_np = extract_problem_vector(problem.delassus, problem.data.v_f.numpy(), only_active_dims=only_active_dims)
    P_wp_np = extract_problem_vector(problem.delassus, problem.data.P.numpy(), only_active_dims=only_active_dims)
    v_plus_wp_np = extract_problem_vector(
        problem.delassus, solver.data.solution.v_plus.numpy(), only_active_dims=only_active_dims
    )
    lambdas_wp_np = extract_problem_vector(
        problem.delassus, solver.data.solution.lambdas.numpy(), only_active_dims=only_active_dims
    )

    # Optional verbose output
    status = solver.data.status.numpy()
    for w in range(model.size.num_worlds):
        # Recover the original (preconditioned) Delassua matrix from the in-place regularized storage
        dtype = D_wp_np[w].dtype
        ncts = D_wp_np[w].shape[0]
        I_np = dtype.type(solver.config[0].eta + solver.config[0].rho_0) * np.eye(D_wp_np[w].shape[0], dtype=dtype)
        D = D_wp_np[w] - I_np

        # Recover original Delassus matrix and v_f from preconditioned versions
        D_true = np.diag(np.reciprocal(P_wp_np[w])) @ D @ np.diag(np.reciprocal(P_wp_np[w]))
        v_f_true = np.diag(np.reciprocal(P_wp_np[w])) @ v_f_wp_np[w]

        # Compute the true dual solution and error
        v_plus_true = np.matmul(D_true, lambdas_wp_np[w]) + v_f_true
        error_dual_abs_l2 = np.linalg.norm(v_plus_true - v_plus_wp_np[w]) / float(ncts)
        error_dual_abs_inf = np.linalg.norm(v_plus_true - v_plus_wp_np[w], ord=np.inf)

        # Extract solver status
        converged = bool(status[w]["converged"])
        iterations = status[w]["iterations"]
        r_p = status[w]["r_p"]
        r_d = status[w]["r_d"]
        r_c = status[w]["r_c"]

        # Optionally print relevant solver data
        if verbose:
            print_dual_problem_summary(D, v_f_wp_np[w], "(preconditioned)")
            print_dual_problem_summary(D_true, v_f_true)
            msg.notif(
                "\n---------"
                f"\nconverged: {converged}"
                f"\niterations: {iterations}"
                "\n---------"
                f"\nr_p: {r_p}"
                f"\nr_d: {r_d}"
                f"\nr_c: {r_c}"
                "\n---------"
                f"\nerror_dual_abs_l2: {error_dual_abs_l2}"
                f"\nerror_dual_abs_inf: {error_dual_abs_inf}"
                "\n---------"
                f"\nsolution: lambda: {lambdas_wp_np[w]}"
                f"\nsolution: v_plus: {v_plus_wp_np[w]}"
                "\n---------\n"
            )

        # Check results
        test.assertTrue(converged)
        test.assertLessEqual(iterations, solver.config[w].max_iterations)
        test.assertLessEqual(r_p, solver.config[w].primal_tolerance)
        test.assertLessEqual(r_d, solver.config[w].dual_tolerance)
        test.assertLessEqual(r_c, solver.config[w].compl_tolerance)
        # Using expanded tolerance for true dual error due to potential numerical inaccuracies
        # between in-solver residual and true residual.
        test.assertLessEqual(error_dual_abs_l2, solver.config[w].dual_tolerance * 4.0)
        test.assertLessEqual(error_dual_abs_inf, solver.config[w].dual_tolerance * 4.0)


def solve_apadmm_bilateral_reference(
    D: np.ndarray,
    v_f: np.ndarray,
    P: np.ndarray,
    config: PADMMSolver.Config,
) -> dict[str, Any]:
    """Solve a bilateral APADMM problem with a NumPy iteration reference.

    This mirrors the accelerated PADMM state transitions for one dense,
    preconditioned bilateral problem. The bilateral feasible set is all of
    :math:`R^n`, so the production solver's projection is the identity:
    ``y = x - z_hat / rho``. It consequently does not model unilateral limit
    or contact projection, De Saxce corrections, or complementarity residuals.

    The reference also omits production features that are irrelevant to this
    fixed-iteration regression: multiple worlds, sparse/iterative linear
    solves, adaptive penalty updates, and convergence-based early termination.
    It instead solves the fixed dense system with NumPy and executes exactly
    ``max_iterations`` iterations, recording the state-derived quantities
    exposed through ``PADMMInfo`` for comparison.
    """
    dtype = D.dtype
    eta = dtype.type(config.eta)
    rho = dtype.type(config.rho_0)
    restart_tolerance = dtype.type(config.restart_tolerance)
    a_0 = dtype.type(config.a_0)
    lhs = D + (eta + rho) * np.eye(D.shape[0], dtype=dtype)

    x = np.zeros_like(v_f)
    y = np.zeros_like(v_f)
    z = np.zeros_like(v_f)
    x_p = np.zeros_like(v_f)
    y_p = np.zeros_like(v_f)
    z_p = np.zeros_like(v_f)
    y_hat = np.zeros_like(v_f)
    z_hat = np.zeros_like(v_f)
    a_p = a_0
    r_a_p = dtype.type(np.finfo(dtype).max)
    restart = 0
    num_restarts = 0
    history = {
        "a": [],
        "norm_x": [],
        "norm_y": [],
        "norm_z": [],
        "r_dx": [],
        "r_dy": [],
        "r_dz": [],
        "r_comb": [],
        "r_comb_ratio": [],
        "num_restarts": [],
    }

    for _ in range(config.max_iterations):
        x_candidate = np.linalg.solve(lhs, -v_f + eta * x_p + rho * y_hat + z_hat)
        y_candidate = x_candidate - z_hat / rho
        z_candidate = z_hat + rho * (y_candidate - x_candidate)

        dx = P * (x_candidate - x_p)
        dy = P * (y_candidate - y_hat)
        dz = (z_candidate - z_hat) / P
        r_a = rho * np.dot(dy, dy) + np.dot(dz, dz) / rho

        if r_a < restart_tolerance * r_a_p:
            restart = 0
            a = dtype.type((1.0 + np.sqrt(1.0 + 4.0 * a_p * a_p)) / 2.0)
            factor = dtype.type((a_p - 1.0) / a)
            x = x_candidate
            y = y_candidate
            z = z_candidate
            y_hat = y + factor * (y - y_p)
            z_hat = z + factor * (z - z_p)
            x_p = x.copy()
            y_p = y.copy()
            z_p = z.copy()
        else:
            # A restart drops the momentum but keeps the current iterate.
            restart = 1
            num_restarts += 1
            r_a = r_a_p / restart_tolerance
            a = a_0
            y_hat = y_p.copy()
            z_hat = z_p.copy()
            x = x_candidate
            y = y_candidate
            z = z_candidate
            x_p = x.copy()
            y_p = y.copy()
            z_p = z.copy()

        a_p = a
        r_a_pp = r_a_p
        r_a_p = dtype.type(r_a)

        history["a"].append(a)
        history["norm_x"].append(np.linalg.norm(x))
        history["norm_y"].append(np.linalg.norm(y))
        history["norm_z"].append(np.linalg.norm(z))
        history["r_dx"].append(np.linalg.norm(dx))
        history["r_dy"].append(np.linalg.norm(dy))
        history["r_dz"].append(np.linalg.norm(dz))
        history["r_comb"].append(r_a)
        history["r_comb_ratio"].append(r_a / r_a_pp)
        history["num_restarts"].append(num_restarts)

    return {
        "x": x,
        "y": y,
        "z": z,
        "x_p": x_p,
        "y_p": y_p,
        "z_p": z_p,
        "y_hat": y_hat,
        "z_hat": z_hat,
        "restart": restart,
        "num_restarts": num_restarts,
        "history": {key: np.asarray(values, dtype=dtype) for key, values in history.items()},
    }


def save_solver_info(solver: PADMMSolver, path: str | None = None, verbose: bool = False):
    # Attempt to import matplotlib for plotting
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib is not available so we skip plotting

    solver_has_acceleration = solver._use_acceleration

    nw = solver.size.num_worlds
    status = solver.data.status.numpy()
    iterations = [status[w]["iterations"] for w in range(nw)]
    offsets_np = solver.data.info.offsets.numpy()
    num_rho_updates_np = extract_info_vectors(offsets_np, solver.data.info.num_rho_updates.numpy(), iterations)
    norm_s_np = extract_info_vectors(offsets_np, solver.data.info.norm_s.numpy(), iterations)
    norm_x_np = extract_info_vectors(offsets_np, solver.data.info.norm_x.numpy(), iterations)
    norm_y_np = extract_info_vectors(offsets_np, solver.data.info.norm_y.numpy(), iterations)
    norm_z_np = extract_info_vectors(offsets_np, solver.data.info.norm_z.numpy(), iterations)
    f_ccp_np = extract_info_vectors(offsets_np, solver.data.info.f_ccp.numpy(), iterations)
    f_ncp_np = extract_info_vectors(offsets_np, solver.data.info.f_ncp.numpy(), iterations)
    r_dx_np = extract_info_vectors(offsets_np, solver.data.info.r_dx.numpy(), iterations)
    r_dy_np = extract_info_vectors(offsets_np, solver.data.info.r_dy.numpy(), iterations)
    r_dz_np = extract_info_vectors(offsets_np, solver.data.info.r_dz.numpy(), iterations)
    r_primal_np = extract_info_vectors(offsets_np, solver.data.info.r_primal.numpy(), iterations)
    r_dual_np = extract_info_vectors(offsets_np, solver.data.info.r_dual.numpy(), iterations)
    r_compl_np = extract_info_vectors(offsets_np, solver.data.info.r_compl.numpy(), iterations)
    r_pd_np = extract_info_vectors(offsets_np, solver.data.info.r_pd.numpy(), iterations)
    r_dp_np = extract_info_vectors(offsets_np, solver.data.info.r_dp.numpy(), iterations)
    r_ncp_primal_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_primal.numpy(), iterations)
    r_ncp_dual_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_dual.numpy(), iterations)
    r_ncp_compl_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_compl.numpy(), iterations)
    r_ncp_natmap_np = extract_info_vectors(offsets_np, solver.data.info.r_ncp_natmap.numpy(), iterations)

    if solver_has_acceleration:
        num_restarts_np = extract_info_vectors(offsets_np, solver.data.info.num_restarts.numpy(), iterations)
        a_np = extract_info_vectors(offsets_np, solver.data.info.a.numpy(), iterations)
        r_comb_np = extract_info_vectors(offsets_np, solver.data.info.r_comb.numpy(), iterations)
        r_comb_ratio_np = extract_info_vectors(offsets_np, solver.data.info.r_comb_ratio.numpy(), iterations)

    if verbose:
        for w in range(nw):
            print(f"[World {w}] =======================================================================")
            print(f"solver.info.num_rho_updates: {num_rho_updates_np[w]}")
            print(f"solver.info.norm_s: {norm_s_np[w]}")
            print(f"solver.info.norm_x: {norm_x_np[w]}")
            print(f"solver.info.norm_y: {norm_y_np[w]}")
            print(f"solver.info.norm_z: {norm_z_np[w]}")
            print(f"solver.info.f_ccp: {f_ccp_np[w]}")
            print(f"solver.info.f_ncp: {f_ncp_np[w]}")
            print(f"solver.info.r_dx: {r_dx_np[w]}")
            print(f"solver.info.r_dy: {r_dy_np[w]}")
            print(f"solver.info.r_dz: {r_dz_np[w]}")
            print(f"solver.info.r_primal: {r_primal_np[w]}")
            print(f"solver.info.r_dual: {r_dual_np[w]}")
            print(f"solver.info.r_compl: {r_compl_np[w]}")
            print(f"solver.info.r_pd: {r_pd_np[w]}")
            print(f"solver.info.r_dp: {r_dp_np[w]}")
            print(f"solver.info.r_ncp_primal: {r_ncp_primal_np[w]}")
            print(f"solver.info.r_ncp_dual: {r_ncp_dual_np[w]}")
            print(f"solver.info.r_ncp_compl: {r_ncp_compl_np[w]}")
            print(f"solver.info.r_ncp_natmap: {r_ncp_natmap_np[w]}")
            if solver_has_acceleration:
                print(f"solver.info.num_restarts: {num_restarts_np[w]}")
                print(f"solver.info.a: {a_np[w]}")
                print(f"solver.info.r_comb: {r_comb_np[w]}")
                print(f"solver.info.r_comb_ratio: {r_comb_ratio_np[w]}")

    # List of (label, data) for plotting
    info_list = [
        ("num_rho_updates", num_rho_updates_np),
        ("norm_s", norm_s_np),
        ("norm_x", norm_x_np),
        ("norm_y", norm_y_np),
        ("norm_z", norm_z_np),
        ("f_ccp", f_ccp_np),
        ("f_ncp", f_ncp_np),
        ("r_dx", r_dx_np),
        ("r_dy", r_dy_np),
        ("r_dz", r_dz_np),
        ("r_primal", r_primal_np),
        ("r_dual", r_dual_np),
        ("r_compl", r_compl_np),
        ("r_pd", r_pd_np),
        ("r_dp", r_dp_np),
        ("r_ncp_primal", r_ncp_primal_np),
        ("r_ncp_dual", r_ncp_dual_np),
        ("r_ncp_compl", r_ncp_compl_np),
        ("r_ncp_natmap", r_ncp_natmap_np),
    ]
    if solver_has_acceleration:
        info_list.extend(
            [
                ("num_restarts", num_restarts_np),
                ("a", a_np),
                ("r_comb", r_comb_np),
                ("r_comb_ratio", r_comb_ratio_np),
            ]
        )

    # Plot all info as subplots: rows=info_list, cols=worlds
    n_rows = len(info_list)
    n_cols = nw
    _fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False)
    for row, (label, arr) in enumerate(info_list):
        for col in range(nw):
            ax = axes[row, col]
            ax.plot(arr[col], label=f"{label}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f"World {col}")
            if col == 0:
                ax.set_ylabel(label)
            else:
                ax.set_ylabel("")
            ax.grid(True)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()


###
# Tests
###


class TestPADMMSolver(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.verbose = test_context.verbose  # Set to True for detailed output
        self.savefig = test_context.verbose  # Set to True to generate solver info plots
        self.default_device = wp.get_device(test_context.device)
        self.output_path = test_context.output_path / "test_solvers_padmm"

        # Create output directory if saving figures
        if self.savefig:
            self.output_path.mkdir(parents=True, exist_ok=True)

        # Set debug-level logging to print verbose test output to console
        if self.verbose:
            print("\n")  # Add newline before test output for better readability
            msg.set_log_level(msg.LogLevel.INFO)
        else:
            msg.reset_log_level()

    def tearDown(self):
        self.default_device = None
        if self.verbose:
            msg.reset_log_level()

    def test_00_make_padmm_default(self):
        """
        Test creating a PADMMSolver with default initialization.
        """
        # Creating a default PADMMSolver without any model or config
        # should result in a solver without any memory allocation.
        solver = PADMMSolver()
        self.assertIsNone(solver._size)
        self.assertIsNone(solver._data)
        self.assertIsNone(solver._device)
        self.assertEqual(solver.config, [])

        # Requesting the solver data container when the
        # solver has not been finalized should raise an
        # error since no allocations have been made.
        self.assertRaises(RuntimeError, lambda: solver.data)

    def test_01_finalize_padmm_default(self):
        """
        Test creating a PADMMSolver with default initialization and then finalizing all memory allocations.
        """
        # Create a test setup
        test = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Creating a default PADMMSolver without any model or config
        # should result in a solver without any memory allocation.
        solver = PADMMSolver()

        # Finalize the solver with a model
        solver.finalize(test.model)

        # Check that the solver has been properly allocated
        self.assertIsNotNone(solver._size)
        self.assertIsNotNone(solver._data)
        self.assertIsNotNone(solver._device)
        self.assertEqual(len(solver.config), test.model.size.num_worlds)
        self.assertIs(solver._device, test.model.device)
        self.assertIs(solver.size, test.model.size)

    def test_02_padmm_solve(self):
        """
        Tests the Proximal-ADMM (PADMM) solver with default config on the reference problem.
        """
        # Create the test problem
        test = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver config
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        config = PADMMSolver.Config()
        config.primal_tolerance = 1e-6
        config.dual_tolerance = 1e-6
        config.compl_tolerance = 1e-6
        config.eta = 1e-5
        config.rho_0 = 1.0
        config.max_iterations = 200

        # Create the PADMM solver
        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.NONE,
            use_acceleration=False,
            collect_info=self.savefig,
        )

        # Solve the test problem
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        # check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Extract solver info
        if self.savefig:
            msg.notif("Generating solver info plots...")
            path = self.output_path / "test_02_padmm_solve.pdf"
            save_solver_info(solver=solver, path=str(path))

        # Check solution
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

    def test_03_padmm_solve_with_acceleration(self):
        """
        Tests the Accelerated Proximal-ADMM (APADMM) solver on the reference problem with Nesterov acceleration.
        """
        # Create the test problem
        test = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver config
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        config = PADMMSolver.Config()
        config.primal_tolerance = 1e-6
        config.dual_tolerance = 1e-6
        config.compl_tolerance = 1e-6
        config.restart_tolerance = 0.999
        config.eta = 1e-5
        config.rho_0 = 1.0
        config.max_iterations = 200

        # Create the PADMM solver
        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.NONE,
            use_acceleration=True,
            collect_info=self.savefig,
        )

        # Solve the test problem
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        # check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Extract solver info
        if self.savefig:
            msg.notif("Generating solver info plots...")
            path = self.output_path / "test_03_padmm_solve_with_acceleration.pdf"
            save_solver_info(solver=solver, path=str(path))

        # Check solution
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

    def test_04_padmm_solve_with_internal_warmstart(self):
        """
        Tests the Proximal-ADMM (PADMM) solver on the reference problem with internal warmstarting.
        """
        # Create the test problem
        test = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver config
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        config = PADMMSolver.Config()
        config.primal_tolerance = 1e-6
        config.dual_tolerance = 1e-6
        config.compl_tolerance = 1e-6
        config.eta = 1e-5
        config.rho_0 = 1.0
        config.max_iterations = 200

        # Create the ADMM solver
        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.INTERNAL,
            use_acceleration=False,
            collect_info=self.savefig,
        )

        # Initial cold-started solve
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Second solve with warm-starting from previous solution
        test.build()
        solver.warmstart(test.problem, test.model, test.data)
        expected_forces = solver.data.solution.lambdas.numpy() * config.warmstart_scale
        np.testing.assert_allclose(solver.data.state.x_p.numpy(), expected_forces)
        np.testing.assert_allclose(solver.data.state.y_p.numpy(), expected_forces)
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Extract solver info
        if self.savefig:
            msg.notif("Generating solver info plots...")
            path = self.output_path / "test_04_padmm_solve_with_internal_warmstart.pdf"
            save_solver_info(solver=solver, path=str(path))

    def test_05_padmm_solve_with_container_warmstart(self):
        """
        Tests the Proximal-ADMM (PADMM) solver on the reference problem with container-based warmstarting.
        """
        # Create the test problem
        test = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver config
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        config = PADMMSolver.Config()
        config.primal_tolerance = 1e-6
        config.dual_tolerance = 1e-6
        config.compl_tolerance = 1e-6
        config.eta = 1e-5
        config.rho_0 = 1.0
        config.max_iterations = 200

        # Create the ADMM solver
        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.CONTAINERS,
            use_acceleration=False,
            collect_info=self.savefig,
        )

        # Initial cold-started solve
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Second solve with warm-starting from previous solution
        test.cache(solver=solver)
        test.build()
        solver.warmstart(test.problem, test.model, test.data, test.limits, test.contacts)
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Extract solver info
        if self.savefig:
            msg.notif("Generating solver info plots...")
            path = self.output_path / "test_05_padmm_solve_with_container_warmstart.pdf"
            save_solver_info(solver=solver, path=str(path))

    def test_06_padmm_solve_with_acceleration_and_internal_warmstart(self):
        """
        Tests the Proximal-ADMM (PADMM) solver on the reference problem with container-based warmstarting.
        """
        # Create the test problem
        test = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver config
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        config = PADMMSolver.Config()
        config.primal_tolerance = 1e-5
        config.dual_tolerance = 1e-5
        config.compl_tolerance = 1e-5
        config.restart_tolerance = 0.999
        config.eta = 1e-5
        config.rho_0 = 1.0
        config.max_iterations = 200

        # Create the ADMM solver
        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.INTERNAL,
            use_acceleration=True,
            collect_info=self.savefig,
        )

        # Initial cold-started solve
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Second solve with warm-starting from previous solution
        test.build()
        solver.warmstart(test.problem, test.model, test.data)
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Extract solver info
        if self.savefig:
            msg.notif("Generating solver info plots...")
            path = self.output_path / "test_06_padmm_solve_with_acceleration_and_internal_warmstart.pdf"
            save_solver_info(solver=solver, path=str(path))

    def test_07_padmm_solve_with_acceleration_and_container_warmstart(self):
        """
        Tests the Proximal-ADMM (PADMM) solver on the reference problem with container-based warmstarting.
        """
        # Create the test problem
        test = TestSetup(builder_fn=basics.build_box_on_plane, max_world_contacts=8, device=self.default_device)

        # Define solver config
        # NOTE: These are all equal to their default values
        # but are defined here explicitly for the purposes
        # of experimentation and testing.
        config = PADMMSolver.Config()
        config.primal_tolerance = 1e-5
        config.dual_tolerance = 1e-5
        config.compl_tolerance = 1e-5
        config.restart_tolerance = 0.999
        config.eta = 1e-5
        config.rho_0 = 1.0
        config.max_iterations = 200

        # Create the ADMM solver
        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.CONTAINERS,
            use_acceleration=True,
            collect_info=self.savefig,
        )

        # Initial cold-started solve
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Second solve with warm-starting from previous solution
        test.cache(solver=solver)
        test.build()
        solver.warmstart(test.problem, test.model, test.data, test.limits, test.contacts)
        solver.solve(problem=test.problem)
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

        # Extract solver info
        if self.savefig:
            msg.notif("Generating solver info plots...")
            path = self.output_path / "test_07_padmm_solve_with_acceleration_and_container_warmstart.pdf"
            save_solver_info(solver=solver, path=str(path))

    def test_08_padmm_solve_single_contact(self):
        """
        Tests the Proximal-ADMM (PADMM) solver with default config on the sphere-on-plane problem
        (no constraints and limits) with a single contact.
        """
        # Create the test problem
        test = TestSetup(builder_fn=basics.build_sphere_on_plane, max_world_contacts=1, device=self.default_device)

        # Create the PADMM solver
        solver = PADMMSolver(model=test.model, collect_info=self.savefig)

        # Solve the test problem
        test.build()
        solver.reset()
        solver.coldstart()
        solver.solve(problem=test.problem)

        # Extract solver info
        if self.savefig and solver.data.info is not None:
            msg.notif("Generating solver info plots...")
            path = self.output_path / "test_08_padmm_solve.pdf"
            save_solver_info(solver=solver, path=str(path))

        # Check solution
        check_padmm_solution(self, test.model, test.problem, solver, verbose=self.verbose)

    def test_08a_padmm_collect_info_matches_solution(self):
        """Record consistent diagnostics for dense and sparse PADMM problems."""
        info_vectors = {}
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                test = TestSetup(
                    builder_fn=basics.build_box_on_plane,
                    device=self.default_device,
                    sparse=sparse,
                )
                solver = PADMMSolver(
                    model=test.model,
                    config=PADMMSolver.Config(
                        primal_tolerance=1.0e3,
                        dual_tolerance=1.0e3,
                        compl_tolerance=1.0e3,
                        max_iterations=8,
                    ),
                    warmstart=PADMMWarmStartMode.NONE,
                    use_acceleration=False,
                    use_graph_conditionals=False,
                    collect_info=True,
                )

                test.build()
                solver.reset()
                solver.coldstart()
                solver.solve(problem=test.problem)

                info = solver.data.info
                status = solver.data.status
                solution = solver.data.solution
                iterations = int(status.numpy()[0]["iterations"])

                # History was written
                self.assertTrue((info.norm_x.numpy()[:iterations] != 0.0).all())
                np.testing.assert_array_equal(info.norm_x.numpy()[iterations:], 0.0)

                # Solution is consistent with info
                np.testing.assert_allclose(info.lambdas.numpy(), solution.lambdas.numpy())
                np.testing.assert_allclose(info.v_aug.numpy(), info.v_plus.numpy() + info.s.numpy())

                # Store info for dense-sparse comparison
                info_vectors[sparse] = (info.v_plus.numpy(), info.v_aug.numpy(), info.s.numpy())

        # Compare dense-sparse info
        for dense_values, sparse_values in zip(info_vectors[False], info_vectors[True], strict=True):
            np.testing.assert_allclose(dense_values, sparse_values, rtol=1e-5, atol=1e-6)

    def test_09_apadmm_restart_matches_bilateral_reference(self):
        """Match a NumPy reference across an APADMM solve whose final iteration restarts.

        The restart tolerance is tight enough to reject every step, so this pins the
        combined-residual formula, the restart bookkeeping, and the rule that a restart
        rewinds only the extrapolation while keeping the current iterate.
        """
        test = TestSetup(
            builder_fn=basics.build_box_pendulum,
            max_world_contacts=1,
            device=self.default_device,
            ground=False,
        )
        config = PADMMSolver.Config(
            primal_tolerance=0.0,
            dual_tolerance=0.0,
            compl_tolerance=0.0,
            restart_tolerance=1.0e-6,
            max_iterations=6,
        )

        test.build()
        D = extract_delassus(test.problem.delassus, only_active_dims=True)[0]
        v_f = extract_problem_vector(
            test.problem.delassus,
            test.problem.data.v_f.numpy(),
            only_active_dims=True,
        )[0]
        P = extract_problem_vector(
            test.problem.delassus,
            test.problem.data.P.numpy(),
            only_active_dims=True,
        )[0]
        reference = solve_apadmm_bilateral_reference(D, v_f, P, config)

        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.NONE,
            use_acceleration=True,
            use_graph_conditionals=False,
            collect_info=True,
        )
        solver.coldstart()
        solver.solve(problem=test.problem)

        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status[11]), reference["restart"])
        self.assertEqual(int(status[12]), reference["num_restarts"])

        # The tight restart tolerance makes the solve alternate between accepted and
        # restarted steps, so both writeback branches run and the run ends on a restart.
        self.assertEqual(reference["restart"], 1)
        self.assertEqual(reference["num_restarts"], 3)

        iterations = int(status[1])
        reference_history = reference["history"]
        solver_history = {
            "a": solver.data.info.a.numpy()[:iterations],
            "norm_x": solver.data.info.norm_x.numpy()[:iterations],
            "norm_y": solver.data.info.norm_y.numpy()[:iterations],
            "norm_z": solver.data.info.norm_z.numpy()[:iterations],
            "r_dx": solver.data.info.r_dx.numpy()[:iterations],
            "r_dy": solver.data.info.r_dy.numpy()[:iterations],
            "r_dz": solver.data.info.r_dz.numpy()[:iterations],
            "r_comb": solver.data.info.r_comb.numpy()[:iterations],
            "r_comb_ratio": solver.data.info.r_comb_ratio.numpy()[:iterations],
            "num_restarts": solver.data.info.num_restarts.numpy()[:iterations],
        }
        for name, values in solver_history.items():
            np.testing.assert_allclose(values, reference_history[name], rtol=1e-5, atol=1e-6, err_msg=name)

        ncts = D.shape[0]
        np.testing.assert_allclose(solver.data.state.x_p.numpy()[:ncts], reference["x_p"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(solver.data.state.y_p.numpy()[:ncts], reference["y_p"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(solver.data.state.z_p.numpy()[:ncts], reference["z_p"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(solver.data.state.y_hat.numpy()[:ncts], reference["y_hat"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(solver.data.state.z_hat.numpy()[:ncts], reference["z_hat"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            solver.data.solution.lambdas.numpy()[:ncts],
            P * reference["y"],
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            solver.data.solution.v_plus.numpy()[:ncts],
            reference["z"] / P,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_10_apadmm_converged_iteration_does_not_restart(self):
        """Suppress the acceleration restart on the iteration that converges.

        The restart tolerance is tight enough that every iteration past the first fails the
        combined-residual test, while the convergence tolerances are loose enough that the
        solve converges on that same iteration. The restart must then be suppressed, so that
        a solve reported as converged never also reports a restart it never acted on.
        """
        test = TestSetup(
            builder_fn=basics.build_box_pendulum,
            max_world_contacts=1,
            device=self.default_device,
            ground=False,
        )
        config = PADMMSolver.Config(
            primal_tolerance=1.0e3,
            dual_tolerance=1.0e3,
            compl_tolerance=1.0e3,
            restart_tolerance=1.0e-6,
            max_iterations=8,
        )

        test.build()
        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.NONE,
            use_acceleration=True,
            use_graph_conditionals=False,
        )
        solver.coldstart()
        solver.solve(problem=test.problem)

        status = solver.data.status.numpy()[0]
        self.assertEqual(int(status[0]), 1)
        # Convergence is gated on having completed more than one iteration.
        self.assertEqual(int(status[1]), 2)
        self.assertEqual(int(status[11]), 0)
        self.assertEqual(int(status[12]), 0)

    def test_11_apadmm_honors_per_world_iteration_budgets(self):
        """Give each world exactly its own iteration budget when the budgets differ.

        A world that exhausts its budget must stop advancing rather than keep stepping on
        the shared loop, and must release the shared completion counter only once, so that
        a world with a larger budget is not terminated early alongside it.
        """
        budgets = [3, 8]
        modes = [False, True] if self.default_device.is_cuda else [False]

        for use_graph_conditionals in modes:
            with self.subTest(use_graph_conditionals=use_graph_conditionals):

                def builder_fn():
                    builder = ModelBuilder()
                    builder.replicate(basics.build_box_pendulum(ground=False), world_count=len(budgets))
                    return builder

                test = TestSetup(
                    builder_fn=builder_fn,
                    max_world_contacts=1,
                    device=self.default_device,
                )
                configs = [
                    PADMMSolver.Config(
                        primal_tolerance=0.0,
                        dual_tolerance=0.0,
                        compl_tolerance=0.0,
                        max_iterations=budget,
                    )
                    for budget in budgets
                ]

                test.build()
                solver = PADMMSolver(
                    model=test.model,
                    config=configs,
                    warmstart=PADMMWarmStartMode.NONE,
                    use_acceleration=True,
                    use_graph_conditionals=use_graph_conditionals,
                )
                solver.coldstart()
                solver.solve(problem=test.problem)

                status = solver.data.status.numpy()
                self.assertEqual([int(status[w][1]) for w in range(len(budgets))], budgets)

    def assert_coulomb_cone_projection(self, cone: str, case: str, mu: float, x: list[float], expected: list[float]):
        """Assert one primal or dual Coulomb cone projection."""
        is_primal = cone == "primal"

        @wp.kernel
        def project(x: wp.array[wp.vec3f], mu: wp.array[wp.float32], y: wp.array[wp.vec3f]):
            if wp.static(is_primal):
                y[0] = project_to_coulomb_cone(x[0], mu[0])
            else:
                y[0] = project_to_coulomb_dual_cone(x[0], mu[0])

        x_arg = wp.array(np.array([x], dtype=np.float32), dtype=wp.vec3f, device=self.default_device)
        mu_arg = wp.array(np.array([mu], dtype=np.float32), dtype=wp.float32, device=self.default_device)
        y = wp.zeros(1, dtype=wp.vec3f, device=self.default_device)
        wp.launch(kernel=project, dim=1, inputs=[x_arg, mu_arg], outputs=[y], device=self.default_device)
        with self.subTest(mu=mu, cone=cone, case=case):
            np.testing.assert_allclose(y.numpy()[0], expected, rtol=1.0e-6, atol=1.0e-6)

    def test_12_coulomb_cone_projections_handle_zero_friction(self):
        """Handle the ray and half-space projections at zero friction."""
        mu = 0.0

        cone = "primal"
        self.assert_coulomb_cone_projection(cone, "positive ray", mu, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
        self.assert_coulomb_cone_projection(cone, "polar", mu, [0.0, 0.0, -1.0], [0.0, 0.0, 0.0])
        self.assert_coulomb_cone_projection(cone, "projection", mu, [1.0, 0.0, 1.0], [0.0, 0.0, 1.0])

        cone = "dual"
        self.assert_coulomb_cone_projection(cone, "pos. half-space", mu, [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
        self.assert_coulomb_cone_projection(cone, "polar", mu, [0.0, 0.0, -1.0], [0.0, 0.0, 0.0])
        self.assert_coulomb_cone_projection(cone, "projection", mu, [1.0, 0.0, -1.0], [1.0, 0.0, 0.0])

    def test_13_coulomb_cone_projections_handle_all_branches(self):
        """Handle interior, polar, and general projections at positive friction."""
        for mu in (1.0e-6, 1.0, 1e6):
            den = 1.0 + mu * mu

            cone = "primal"
            self.assert_coulomb_cone_projection(cone, "interior", mu, [0.5 * mu, 0.0, 1.0], [0.5 * mu, 0.0, 1.0])
            self.assert_coulomb_cone_projection(cone, "polar", mu, [1.0, 0.0, -2.0 * mu], [0.0, 0.0, 0.0])
            self.assert_coulomb_cone_projection(cone, "projection", mu, [1.0, 0.0, 0.0], [mu * mu / den, 0.0, mu / den])

            cone = "dual"
            self.assert_coulomb_cone_projection(cone, "interior", mu, [1.0, 0.0, 2.0 * mu], [1.0, 0.0, 2.0 * mu])
            self.assert_coulomb_cone_projection(cone, "polar", mu, [1.0, 0.0, -2.0 / mu], [0.0, 0.0, 0.0])
            self.assert_coulomb_cone_projection(cone, "projection", mu, [1.0, 0.0, 0.0], [1.0 / den, 0.0, mu / den])


class TestPADMMSpectralPenalty(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    @staticmethod
    def _spectral_rho(config: PADMMSolver.Config, m: float, L: float, p: float) -> float:
        rho = np.sqrt(m * L) * (L / m) ** p
        return float(np.clip(rho, config.rho_min, config.rho_max))

    def test_01_initial_penalty_matches_spectral_rule(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                test = TestSetup(
                    builder_fn=basics.build_box_on_plane,
                    max_world_contacts=8,
                    device=self.default_device,
                    sparse=sparse,
                )
                test.build()
                # NOTE: The dense Delassus matrix is copied since it is regularized in-place by the solve
                D_np = [D.copy() for D in extract_delassus(test.problem.delassus, only_active_dims=True)]
                config = PADMMSolver.Config(
                    penalty_update_method="spectral",
                    penalty_update_freq=0,
                    max_iterations=1,
                    spectral_max_iterations=500,
                    spectral_tolerance=1e-7,
                )
                solver = PADMMSolver(model=test.model, config=config, use_graph_conditionals=False)
                solver.coldstart()
                solver.solve(problem=test.problem)

                # Without compliance, the smallest eigenvalue is bounded from below by the proximal parameter
                penalty = solver.data.penalty.numpy()
                for w, D in enumerate(D_np):
                    L = np.linalg.eigvalsh(D.astype(np.float64))[-1] + config.eta
                    np.testing.assert_allclose(penalty[w]["m"], config.eta, rtol=1e-6)
                    np.testing.assert_allclose(penalty[w]["L"], L, rtol=1e-3)
                    np.testing.assert_allclose(penalty[w]["p"], config.spectral_power_0, rtol=1e-6)
                    expected_rho = self._spectral_rho(config, config.eta, L, config.spectral_power_0)
                    np.testing.assert_allclose(penalty[w]["rho"], expected_rho, rtol=2e-3)
                    np.testing.assert_allclose(solver.data.state.sigma.numpy()[w][0], config.eta + penalty[w]["rho"])

    def test_02_spectral_exponent_update_rule(self):
        """Checks the adaptation of the spectral exponent from the ratio of the primal and dual residuals."""
        config = PADMMSolver.Config(
            penalty_update_method="spectral",
            alpha=10.0,
            spectral_power_inc=0.05,
            spectral_power_dec=0.1,
            max_iterations=100,
        )
        cases = (
            # (r_p, r_d, converged, expected change of p)
            (1.0, 1e-2, 0, 0.05),
            (1e-2, 1.0, 0, -0.1),
            (1.0, 0.5, 0, 0.0),
            (1.0, 1e-2, 1, 0.0),
        )
        device = self.default_device
        num_worlds = len(cases)
        configs = wp.array([convert_config_to_struct(config)] * num_worlds, dtype=PADMMConfigStruct, device=device)
        statuses, penalties = [], []
        for r_p, r_d, converged, _ in cases:
            status = PADMMStatus()
            status.r_p, status.r_d, status.converged, status.iterations = r_p, r_d, converged, 10
            statuses.append(status)
            penalty = PADMMPenalty()
            penalty.p, penalty.m, penalty.L = 0.2, 1e-5, 4.0
            penalty.rho = self._spectral_rho(config, 1e-5, 4.0, 0.2)
            penalties.append(penalty)
        status = wp.array(statuses, dtype=PADMMStatus, device=device)
        penalty = wp.array(penalties, dtype=PADMMPenalty, device=device)
        sigma = wp.zeros(num_worlds, dtype=wp.vec2f, device=device)
        wp.launch(_update_spectral_penalty, dim=num_worlds, inputs=[configs, status], outputs=[penalty, sigma])
        penalty_np = penalty.numpy()
        for w, (*_, dp) in enumerate(cases):
            p = 0.2 + dp
            np.testing.assert_allclose(penalty_np[w]["p"], p, rtol=1e-6)
            np.testing.assert_allclose(penalty_np[w]["rho"], self._spectral_rho(config, 1e-5, 4.0, p), rtol=1e-5)
            self.assertEqual(penalty_np[w]["num_updates"], int(dp != 0.0))

    def test_03_penalty_updates_follow_block_frequency(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                test = TestSetup(
                    builder_fn=basics.build_box_on_plane,
                    max_world_contacts=8,
                    device=self.default_device,
                    sparse=sparse,
                )
                test.build()
                config = PADMMSolver.Config(
                    penalty_update_method="spectral",
                    penalty_update_freq=5,
                    primal_tolerance=0.0,
                    dual_tolerance=0.0,
                    compl_tolerance=0.0,
                    max_iterations=23,
                )
                solver = PADMMSolver(model=test.model, config=config, use_graph_conditionals=False)
                solver.coldstart()
                solver.solve(problem=test.problem)

                # The penalty is updated at most once per block of iterations
                status = solver.data.status.numpy()[0]
                penalty = solver.data.penalty.numpy()[0]
                num_blocks = config.max_iterations // config.penalty_update_freq
                self.assertEqual(status["iterations"], config.max_iterations)
                self.assertLessEqual(penalty["num_updates"], num_blocks)
                steps = (penalty["p"] - config.spectral_power_0) / config.spectral_power_inc
                self.assertLessEqual(abs(steps), num_blocks + 1e-3)
                expected_rho = self._spectral_rho(config, penalty["m"], penalty["L"], penalty["p"])
                np.testing.assert_allclose(penalty["rho"], expected_rho, rtol=1e-5)

                # The regularization of the problem holds the final penalty
                eta = test.problem.data.eta.numpy()[: test.data.info.num_total_cts.numpy()[0]]
                np.testing.assert_allclose(eta, config.eta + penalty["rho"], rtol=1e-6)

    def test_04_penalty_warmstart(self):
        """Checks that adaptive penalties are warm-started across solves only if enabled, until reset."""
        common = {"primal_tolerance": 0.0, "dual_tolerance": 0.0, "compl_tolerance": 0.0, "max_iterations": 20}
        cases = (
            ("spectral", False, {"penalty_update_method": "spectral", "penalty_update_freq": 5}),
            ("balanced", True, {"penalty_update_method": "balanced", "penalty_update_freq": 1, "rho_0": 1e-3}),
        )
        for name, sparse, kwargs in cases:
            for penalty_warmstart in (False, True):
                with self.subTest(method=name, penalty_warmstart=penalty_warmstart):
                    test = TestSetup(
                        builder_fn=basics.build_box_on_plane,
                        max_world_contacts=8,
                        device=self.default_device,
                        sparse=sparse,
                    )
                    test.build()
                    config = PADMMSolver.Config(penalty_warmstart=penalty_warmstart, **common, **kwargs)
                    solver = PADMMSolver(model=test.model, config=config, use_graph_conditionals=False)
                    solver.coldstart()
                    solver.solve(problem=test.problem)
                    key = "p" if name == "spectral" else "rho"
                    initial = config.spectral_power_0 if name == "spectral" else config.rho_0
                    final = solver.data.penalty.numpy()[0][key]
                    self.assertNotAlmostEqual(final, initial, places=4)

                    # The next solve starts from the last penalty only if warm-starting is enabled
                    solver._initialize()
                    expected = final if penalty_warmstart else initial
                    np.testing.assert_allclose(solver.data.penalty.numpy()[0][key], expected, rtol=1e-6)
                    self.assertEqual(solver.data.penalty.numpy()[0]["num_updates"], 0)

                    # Resetting the solver re-initializes the penalty
                    solver.reset()
                    solver._initialize()
                    np.testing.assert_allclose(solver.data.penalty.numpy()[0][key], initial, rtol=1e-6)

    def test_05_spectral_solve_converges(self):
        for sparse in (False, True):
            for use_acceleration in (False, True):
                with self.subTest(sparse=sparse, use_acceleration=use_acceleration):
                    test = TestSetup(
                        builder_fn=basics.build_box_on_plane,
                        max_world_contacts=8,
                        device=self.default_device,
                        sparse=sparse,
                    )
                    test.build()
                    # NOTE: The spectral rule only adapts the penalty to the ratio of the primal and dual
                    # residuals, so the complementarity residual may stagnate once both have converged
                    config = PADMMSolver.Config(
                        penalty_update_method="spectral",
                        penalty_update_freq=10,
                        compl_tolerance=1e-5,
                        max_iterations=2000,
                    )
                    solver = PADMMSolver(
                        model=test.model,
                        config=config,
                        use_acceleration=use_acceleration,
                        use_graph_conditionals=False,
                    )
                    solver.coldstart()
                    solver.solve(problem=test.problem)
                    status = solver.data.status.numpy()[0]
                    self.assertTrue(bool(status["converged"]), msg=f"status: {status}")
                    self.assertLessEqual(status["r_p"], config.primal_tolerance)
                    self.assertLessEqual(status["r_d"], config.dual_tolerance)
                    self.assertLessEqual(status["r_c"], config.compl_tolerance)

                    # The solution satisfies the constraint-space dynamics: v_plus = D lambda + v_f
                    problem = test.problem
                    v_plus = wp.zeros_like(solver.data.solution.v_plus)
                    problem.gemv(x=solver.data.solution.lambdas, y=v_plus, preconditioned=False)
                    n = test.data.info.num_total_cts.numpy()[0]
                    v_plus_ref = (v_plus.numpy() + problem.data.v_f.numpy() / problem.data.P.numpy())[:n]
                    np.testing.assert_allclose(solver.data.solution.v_plus.numpy()[:n], v_plus_ref, atol=1e-4)


def project_to_dual_cone_reference(x: np.ndarray, mu: float) -> np.ndarray:
    """Projects a 3D vector (normal last) onto the dual of the Coulomb friction cone of coefficient `mu`."""
    xt, xn = x[:2], x[2]
    n = np.linalg.norm(xt)
    if mu * n <= xn:
        return x.copy()
    if n <= -mu * xn:
        return np.zeros(3)
    a = (n + mu * xn) / (1.0 + mu * mu)
    return np.array([a * xt[0] / n, a * xt[1] / n, mu * a])


def consistent_warmstart_velocity_reference(problem: DualProblem, D_p: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    """Computes the reference warm-start velocity `z_0 = Proj_{K*}(sigma_0 + Gamma(sigma_0))` of a world."""
    d = problem.data
    vio, dim = int(d.vio.numpy()[w]), int(d.dim.numpy()[w])
    sigma = D_p.astype(np.float64) @ y[vio : vio + dim] + d.v_f.numpy()[vio : vio + dim]
    bcgo, nbc = int(d.bcgo.numpy()[w]), int(d.nbc.numpy()[w])
    lcgo, nl = int(d.lcgo.numpy()[w]), int(d.nl.numpy()[w])
    ccgo, nc = int(d.ccgo.numpy()[w]), int(d.nc.numpy()[w])
    bcio, cio = int(d.bcio.numpy()[w]), int(d.cio.numpy()[w])
    mu, lower, upper = d.mu.numpy(), d.bound_lower.numpy(), d.bound_upper.numpy()
    z = np.zeros(dim)
    for i in range(dim):
        if bcgo <= i < bcgo + nbc:
            b, y_i = bcio + i - bcgo, y[vio + i]
            z[i] = max(sigma[i], 0.0) if y_i <= lower[b] else (min(sigma[i], 0.0) if y_i >= upper[b] else 0.0)
        elif lcgo <= i < lcgo + nl:
            z[i] = max(sigma[i], 0.0)
    for c in range(nc):
        k = ccgo + 3 * c
        s = sigma[k : k + 3].copy()
        s[2] += mu[cio + c] * np.linalg.norm(s[:2])
        z[k : k + 3] = project_to_dual_cone_reference(s, mu[cio + c])
    return z


class TestPADMMConsistentWarmStart(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def _warmstart(self, test: TestSetup, warmstart_velocity: str, use_acceleration: bool) -> PADMMSolver:
        config = PADMMSolver.Config(warmstart_velocity=warmstart_velocity, max_iterations=500)
        solver = PADMMSolver(
            model=test.model,
            config=config,
            warmstart=PADMMWarmStartMode.INTERNAL,
            use_acceleration=use_acceleration,
            use_graph_conditionals=False,
        )
        test.build()
        solver.coldstart()
        solver.solve(problem=test.problem)
        test.build()
        solver.warmstart(test.problem, test.model, test.data, test.limits, test.contacts)
        return solver

    def test_01_consistent_velocity_matches_reference(self):
        for builder_fn in (basics.build_box_on_plane, basics.build_boxes_fourbar):
            for sparse in (False, True):
                for use_acceleration in (False, True):
                    with self.subTest(builder=builder_fn.__name__, sparse=sparse, use_acceleration=use_acceleration):
                        test = TestSetup(
                            builder_fn=builder_fn, max_world_contacts=8, device=self.default_device, sparse=sparse
                        )
                        solver = self._warmstart(test, "consistent", use_acceleration)
                        state = solver.data.state
                        y = (state.y_hat if use_acceleration else state.y_p).numpy()
                        z = (state.z_hat if use_acceleration else state.z_p).numpy()
                        # NOTE: The regularization of the previous solve is removed from the reference operator
                        test.problem.data.eta.zero_()
                        test.problem.update_regularization()
                        D_np = [D.copy() for D in extract_delassus(test.problem.delassus, only_active_dims=True)]
                        d = test.problem.data
                        for w in range(test.model.size.num_worlds):
                            vio, dim = int(d.vio.numpy()[w]), int(d.dim.numpy()[w])
                            z_ref = consistent_warmstart_velocity_reference(test.problem, D_np[w], y, w)
                            scale = max(1.0, float(np.max(np.abs(z_ref))))
                            np.testing.assert_allclose(z[vio : vio + dim], z_ref, atol=1e-4 * scale)

                            # The augmented contact velocities lie in the dual friction cones
                            mu = d.mu.numpy()[d.cio.numpy()[w] : d.cio.numpy()[w] + d.nc.numpy()[w]]
                            for c, mu_c in enumerate(mu):
                                k = vio + int(d.ccgo.numpy()[w]) + 3 * c
                                self.assertLessEqual(mu_c * np.linalg.norm(z[k : k + 2]), z[k + 2] + 1e-5 * scale)

    def test_02_consistent_velocity_converges(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                iterations = {}
                for warmstart_velocity in ("cached", "consistent"):
                    test = TestSetup(
                        builder_fn=basics.build_box_on_plane,
                        max_world_contacts=8,
                        device=self.default_device,
                        sparse=sparse,
                    )
                    solver = self._warmstart(test, warmstart_velocity, use_acceleration=True)
                    solver.solve(problem=test.problem)
                    status = solver.data.status.numpy()[0]
                    self.assertTrue(bool(status["converged"]))
                    iterations[warmstart_velocity] = int(status["iterations"])
                # NOTE: Re-solving the same problem, the cached velocities are already consistent with the
                # reactions, so the consistent warm-start must neither help nor hurt significantly
                self.assertLessEqual(iterations["consistent"], int(1.2 * iterations["cached"]) + 2)


class TestPADMMRelativeTolerances(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

    def tearDown(self):
        self.default_device = None

    def _solve(self, sparse: bool, use_acceleration: bool, **kwargs) -> tuple[TestSetup, PADMMSolver]:
        test = TestSetup(
            builder_fn=basics.build_boxes_fourbar, max_world_contacts=8, device=self.default_device, sparse=sparse
        )
        test.build()
        solver = PADMMSolver(
            model=test.model,
            config=PADMMSolver.Config(max_iterations=500, **kwargs),
            use_acceleration=use_acceleration,
            use_graph_conditionals=False,
        )
        solver.coldstart()
        solver.solve(problem=test.problem)
        return test, solver

    def test_01_zero_relative_tolerances_are_identical(self):
        for use_acceleration in (False, True):
            with self.subTest(use_acceleration=use_acceleration):
                _, default = self._solve(sparse=False, use_acceleration=use_acceleration)
                _, relative = self._solve(
                    sparse=False,
                    use_acceleration=use_acceleration,
                    primal_tolerance_rel=0.0,
                    dual_tolerance_rel=0.0,
                    compl_tolerance_rel=0.0,
                )
                np.testing.assert_array_equal(relative.data.status.numpy(), default.data.status.numpy())
                np.testing.assert_array_equal(
                    relative.data.solution.lambdas.numpy(), default.data.solution.lambdas.numpy()
                )

    def test_02_relative_tolerances_terminate_solve(self):
        eps_rel = 1e-4
        tolerances = {
            "primal_tolerance": 0.0,
            "dual_tolerance": 0.0,
            "compl_tolerance": 0.0,
            "primal_tolerance_rel": eps_rel,
            "dual_tolerance_rel": eps_rel,
            "compl_tolerance_rel": eps_rel,
        }
        for sparse in (False, True):
            for use_acceleration in (False, True):
                with self.subTest(sparse=sparse, use_acceleration=use_acceleration):
                    test, solver = self._solve(sparse=sparse, use_acceleration=use_acceleration, **tolerances)
                    status = solver.data.status.numpy()[0]
                    self.assertTrue(bool(status["converged"]))
                    self.assertLess(status["iterations"], 500)

                    # The residuals satisfy the relative criteria w.r.t. the final iterates,
                    # which the solver has already mapped back to physical units
                    n = test.data.info.num_total_cts.numpy()[0]
                    state = solver.data.state
                    x, y, z = (a.numpy()[:n] for a in (state.x, state.y, state.z))
                    n_xy = max(np.max(np.abs(x)), np.max(np.abs(y)))
                    n_z = np.max(np.abs(z))
                    slack = 1.0 + 1e-5
                    self.assertLessEqual(status["r_p"], eps_rel * n_xy * slack)
                    self.assertLessEqual(status["r_d"], eps_rel * n_z * slack)
                    self.assertLessEqual(status["r_c"], eps_rel * n_xy * n_z * slack)


###
# Test execution
###

if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
