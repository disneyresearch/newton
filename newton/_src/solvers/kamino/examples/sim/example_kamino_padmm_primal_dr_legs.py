# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example for the primal PADMM solver with DR Legs.
#
# Shows how to simulate DR Legs with multiple worlds using the primal
# Proximal-ADMM forward dynamics solver of SolverKamino, which solves each
# iteration in the generalized velocities using the constraint-augmented
# mass matrix `M + J^T W J` instead of the Delassus matrix. The dual PADMM
# solver can be selected instead to compare both, since their iterates are
# identical up to the inversion of the penalty parameter.
#
# Command: python newton/_src/solvers/kamino/examples/sim/example_kamino_padmm_primal_dr_legs.py --world-count 16
#
###########################################################################

import argparse

import warp as wp

import newton
import newton.examples
from newton.solvers import SolverKamino


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        # Set simulation run-time configurations
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = 0.01
        self.sim_substeps = max(1, round(self.frame_dt / self.sim_dt))
        self.sim_time = 0.0
        self.frame = 0
        self.world_count = args.world_count if args else 1
        self.use_kamino_contacts = args.use_kamino_contacts if args else True
        self.dynamics_solver = args.dynamics_solver if args else "padmm_primal"
        self.sparse = args.sparse if args else False
        self.log_status = args.log_status if args else False
        self.viewer = viewer
        self.device = wp.get_device()

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverKamino.register_custom_attributes(robot_builder)
        robot_builder.default_shape_cfg.margin = 1e-6
        robot_builder.default_shape_cfg.gap = 1e-2

        # Load the DR Legs USD and add it to the builder, using the same
        # asset revision as the main DR Legs example of SolverKamino
        asset_path = newton.utils.download_asset("disneyresearch", ref="261cd1f429619d8ef4f546bd788ab9dea906b5e1")
        asset_file = str(asset_path / "dr_legs" / "usd" / "dr_legs_with_meshes_and_boxes.usda")
        robot_builder.add_usd(
            asset_file,
            joint_ordering=None,
            force_show_colliders=True,
            force_position_velocity_actuation=True,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        # Set joint armature, viscous damping and implicit PD gains, whose implicit
        # joint dynamics enter the forward dynamics problem as a constraint compliance
        robot_builder.joint_armature = [0.011] * robot_builder.joint_dof_count
        robot_builder.joint_damping = [0.044] * robot_builder.joint_dof_count
        robot_builder.joint_target_ke = [
            10.0 if mode != newton.JointTargetMode.NONE else 0.0 for mode in robot_builder.joint_target_mode
        ]
        robot_builder.joint_target_kd = [
            2.0 if mode != newton.JointTargetMode.NONE else 0.0 for mode in robot_builder.joint_target_mode
        ]

        # Create the multi-world model by duplicating the single-robot
        # builder for the specified number of worlds
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_shape_cfg.margin = 1e-6
        builder.default_shape_cfg.gap = 1e-2
        for _ in range(self.world_count):
            builder.add_world(robot_builder)

        # Add a global ground plane applied to all worlds
        builder.add_ground_plane()

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)
        self.model.rigid_contact_max = 72 * self.world_count

        # Create and configure the Kamino solver for the given model
        self.config = SolverKamino.Config.from_model(
            self.model,
            dynamics_solver=self.dynamics_solver,
            sparse_jacobian=self.sparse,
            sparse_dynamics=self.sparse,
        )
        self.config.use_fk_solver = True
        self.config.use_collision_detector = self.use_kamino_contacts
        self.config.constraints.delta = 1e-3
        dense_linear_solver_type = args.linear_solver_type if args else "LLTB"
        self.config.dynamics.linear_solver_type = "CR" if self.sparse else dense_linear_solver_type
        contact_compliance = args.contact_compliance if args else 0.0
        self.config.compliance.contacts_normal = contact_compliance
        self.config.compliance.contacts_tangential = contact_compliance

        # Configure the selected PADMM solver with equivalent settings, i.e.
        # the primal penalty `rho` is the inverse of the dual penalty
        rho = args.rho if args else 100.0
        padmm_config = self.config.padmm_primal if self.dynamics_solver == "padmm_primal" else self.config.padmm
        padmm_config.max_iterations = args.max_iterations if args else 200
        padmm_config.primal_tolerance = 1e-4
        padmm_config.dual_tolerance = 1e-4
        padmm_config.compl_tolerance = 1e-4
        padmm_config.rho_0 = rho if self.dynamics_solver == "padmm_primal" else 1.0 / rho
        padmm_config.use_acceleration = True
        padmm_config.use_graph_conditionals = args.use_graph_conditionals if args else True
        self.solver = SolverKamino(self.model, config=self.config)

        # Create state and control data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Configure CD components based on whether we want to use Kamino's
        # internal collision detector or Newton's collision pipeline
        self.collision_pipeline = None if self.use_kamino_contacts else newton.CollisionPipeline(self.model)
        self.contacts = newton.CollisionPipeline(self.model).contacts()
        self.observables = self.solver.observables({newton.solvers.SolverObservableFlags.CONTACT_F})

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)

        # Reset the simulation state to a valid initial configuration above the ground
        self.base_q = wp.zeros(shape=(self.world_count,), dtype=wp.transformf)
        q_base = wp.transformf((0.0, 0.0, 0.4), wp.quat_identity(dtype=wp.float32))
        self.base_q.assign([q_base] * self.world_count)
        reset_config = SolverKamino.ResetConfig(base_pose=SolverKamino.ResetConfig.FromBaseQ(base_q=self.base_q))
        self.solver.reset(state=self.state_0, config=reset_config)
        self.solver.reset(state=self.state_1, config=reset_config)

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.capture()

        # If only a single-world is created, set initial
        # camera position for better view of the system
        if self.world_count == 1 and hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(1.34, 0.0, 0.25), -7.0, -180.0)

    def capture(self):
        self.graph = None
        if self.device.is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # simulate() performs one frame's worth of updates
    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.collision_pipeline is not None:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt, observables=self.observables
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame += 1
        if self.log_status:
            self._print_solver_status()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_1, observables=self.observables)
        self.viewer.end_frame()

    def test_final(self):
        # NOTE: Comparisons with NaN are false, so these checks also fail for non-finite states
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > -0.01 and q[2] < 1.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all body velocities are bounded",
            lambda q, qd: wp.length(wp.spatial_top(qd)) < 10.0 and wp.length(wp.spatial_bottom(qd)) < 100.0,
        )

    def _print_solver_status(self):
        """Prints a summary of the terminal solver status of the last step over all worlds."""
        status = self.solver.status.numpy()
        converged = status["converged"].astype(bool)
        iterations = status["iterations"]
        print(
            f"[{self.dynamics_solver}] frame {self.frame:5d}: "
            f"converged {int(converged.sum())}/{converged.size}, "
            f"iterations mean {iterations.mean():6.1f} max {iterations.max():4d}, "
            f"max r_p {status['r_p'].max():.2e}, max r_d {status['r_d'].max():.2e}, max r_c {status['r_c'].max():.2e}"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_kamino_contacts_arg(parser)
        parser.add_argument(
            "--dynamics-solver",
            choices=("padmm_primal", "padmm"),
            default="padmm_primal",
            help="Kamino PADMM dynamics solver to use, i.e. the primal or the dual formulation.",
        )
        parser.add_argument(
            "--sparse",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Use sparse Jacobians and matrix-free dynamics solved with the Conjugate Residual solver.",
        )
        parser.add_argument(
            "--linear-solver-type",
            choices=("LLTB", "LLTBRCM"),
            default="LLTB",
            type=str.upper,
            help="Kamino direct linear solver to use for dense dynamics.",
        )
        parser.add_argument(
            "--rho",
            type=float,
            default=100.0,
            help="Initial PADMM penalty in primal units, i.e. the dual PADMM uses its inverse.",
        )
        parser.add_argument(
            "--max-iterations",
            type=int,
            default=200,
            help="Maximum number of PADMM iterations per step.",
        )
        parser.add_argument(
            "--contact-compliance",
            type=float,
            default=0.0,
            help="Compliance of the normal and tangential contact constraints (zero for rigid contacts).",
        )
        parser.add_argument(
            "--log-status",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Print a summary of the solver status at every frame.",
        )
        parser.add_argument(
            "--no-graph-conditionals",
            dest="use_graph_conditionals",
            action="store_false",
            help="Disable CUDA graph conditional nodes in Kamino PADMM.",
        )
        parser.set_defaults(world_count=1)
        parser.set_defaults(use_kamino_contacts=True)
        parser.set_defaults(use_graph_conditionals=True)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
