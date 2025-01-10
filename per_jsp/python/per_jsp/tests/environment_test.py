import unittest
from typing import List, Tuple
import numpy as np

from per_jsp.environment.job_shop_environment import JobShopEnvironment, Job, Operation, Action
import unittest
from dataclasses import dataclass, field
import numpy as np
from typing import List, Set, Tuple
import logging

logger = logging.getLogger(__name__)

class JobShopEnvironmentTests(unittest.TestCase):
    def setUp(self):
        """Create a simple test environment."""
        # Two jobs, two machines test case
        self.job1 = Job(operations=[
            Operation(duration=3, machine=0),
            Operation(duration=2, machine=1)
        ])
        self.job2 = Job(operations=[
            Operation(duration=2, machine=1),
            Operation(duration=4, machine=0)
        ])
        self.env = JobShopEnvironment([self.job1, self.job2])

    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(len(self.env.jobs), 2)
        self.assertEqual(self.env.num_machines, 2)
        self.assertEqual(self.env.total_time, 0)

        # Verify state initialization
        self.assertEqual(self.env.current_state.job_progress.shape, (2, 2))
        self.assertEqual(self.env.current_state.machine_availability.shape, (2,))
        self.assertEqual(self.env.current_state.next_operation_for_job.shape, (2,))
        self.assertEqual(self.env.current_state.completed_jobs.shape, (2,))

        # Verify all arrays are properly initialized to zeros
        self.assertTrue(np.all(self.env.current_state.job_progress == 0))
        self.assertTrue(np.all(self.env.current_state.machine_availability == 0))
        self.assertTrue(np.all(self.env.current_state.next_operation_for_job == 0))
        self.assertTrue(np.all(self.env.current_state.completed_jobs == False))

    def test_valid_action_sequence(self):
        """Test a valid sequence of actions."""
        # Valid sequence: Job1-Op1 -> Job2-Op1 -> Job1-Op2 -> Job2-Op2
        actions = [
            Action(job=0, machine=0, operation=0),
            Action(job=1, machine=1, operation=0),
            Action(job=0, machine=1, operation=1),
            Action(job=1, machine=0, operation=1)
        ]

        for i, action in enumerate(actions):
            with self.subTest(f"Step {i+1}"):
                state = self.env.step(action)
                errors = self.env.verify_state()
                self.assertEqual(len(errors), 0, f"Verification errors: {errors}")

    def test_invalid_machine(self):
        """Test action with invalid machine."""
        action = Action(job=0, machine=1, operation=0)  # Wrong machine
        with self.assertRaises(ValueError):
            self.env.step(action)

    def test_operation_sequence(self):
        """Test operation sequence constraints."""
        # Try to execute second operation before first
        action = Action(job=0, machine=1, operation=1)
        with self.assertRaises(ValueError):
            self.env.step(action)

    def test_machine_capacity(self):
        """Test machine capacity constraints."""
        # Execute first operation
        self.env.step(Action(job=0, machine=0, operation=0))

        # Try to use same machine before first operation completes
        with self.assertRaises(ValueError):
            self.env.step(Action(job=1, machine=0, operation=1))

    def test_job_completion(self):
        """Test job completion detection."""
        actions = [
            Action(job=0, machine=0, operation=0),
            Action(job=0, machine=1, operation=1)
        ]

        for action in actions:
            self.env.step(action)

        self.assertTrue(self.env.current_state.completed_jobs[0])
        self.assertFalse(self.env.current_state.completed_jobs[1])

    def test_reset(self):
        """Test environment reset."""
        # Execute some actions
        actions = [
            Action(job=0, machine=0, operation=0),
            Action(job=1, machine=1, operation=0)
        ]
        for action in actions:
            self.env.step(action)

        # Reset environment
        self.env.reset()

        # Verify reset state
        self.assertTrue(np.all(self.env.current_state.job_progress == 0))
        self.assertTrue(np.all(self.env.current_state.machine_availability == 0))
        self.assertTrue(np.all(self.env.current_state.next_operation_for_job == 0))
        self.assertTrue(np.all(self.env.current_state.completed_jobs == False))
        self.assertEqual(self.env.total_time, 0)
        self.assertEqual(len(self.env.action_history), 0)

    def test_dependencies(self):
        """Test job dependencies."""
        # Create jobs with dependencies
        job1 = Job(operations=[Operation(duration=2, machine=0)])
        job2 = Job(operations=[Operation(duration=3, machine=0)])
        job2.dependent_jobs = [0]  # Job 2 depends on Job 1

        env = JobShopEnvironment([job1, job2])

        # Try to execute job2 before job1
        with self.assertRaises(ValueError):
            env.step(Action(job=1, machine=0, operation=0))

    def test_get_possible_actions(self):
        """Test possible actions generation."""
        possible_actions = self.env.get_possible_actions()

        # Initially, only first operations of both jobs should be possible
        self.assertEqual(len(possible_actions), 2)
        self.assertTrue(any(a.job == 0 and a.operation == 0 for a in possible_actions))
        self.assertTrue(any(a.job == 1 and a.operation == 0 for a in possible_actions))

    def test_makespan_calculation(self):
        """Test makespan calculation."""
        actions = [
            Action(job=0, machine=0, operation=0),  # Duration 3
            Action(job=1, machine=1, operation=0),  # Duration 2
            Action(job=0, machine=1, operation=1),  # Duration 2
            Action(job=1, machine=0, operation=1)   # Duration 4
        ]

        for action in actions:
            self.env.step(action)

        # Calculate expected makespan
        self.assertTrue(self.env.total_time > 0)
        self.assertEqual(self.env.total_time,
                         max(self.env.current_state.machine_availability))

    def test_machine_utilization(self):
        """Test machine utilization calculation."""
        actions = [
            Action(job=0, machine=0, operation=0),
            Action(job=0, machine=1, operation=1)
        ]

        for action in actions:
            self.env.step(action)

        utilization = self.env.get_machine_utilization()
        self.assertEqual(len(utilization), self.env.num_machines)
        self.assertTrue(all(0 <= u <= 1 for u in utilization))

    def test_error_conditions(self):
        """Test various error conditions."""
        tests = [
            # Invalid job
            (Action(job=2, machine=0, operation=0), ValueError),
            # Invalid operation
            (Action(job=0, machine=0, operation=2), ValueError),
            # Invalid machine
            (Action(job=0, machine=2, operation=0), ValueError),
        ]

        for action, expected_error in tests:
            with self.subTest(f"Testing {action}"), self.assertRaises(expected_error):
                self.env.step(action)

if __name__ == '__main__':
    unittest.main()