/*!
 * ATSP Solver – Manipulator Movement Optimizer
 * ------------------------------------------------
 * Version: v0.1.0
 *
 * Description:
 * This program solves the Asymmetric Traveling Salesman Problem (ATSP)
 * for a manipulator movement optimization scenario using a Mixed Integer
 * Linear Programming (MILP) approach with the `good_lp` crate.
 *
 * Key Features:
 * - Reads a square cost matrix from a CSV file (positions × positions).
 * - Computes the optimal route starting and ending at Home (index 0).
 * - Uses MTZ (Miller–Tucker–Zemlin) constraints to prevent subtours.
 * - Outputs the optimal movement time and route.
 *
 * Usage:
 *   milpr.exe <path_to_csv>
 *
 * Assumptions:
 * - The manipulator can reach any position from any other position without collision.
 * - The cost matrix is complete and square; diagonal elements are zero.
 *
 * Dependencies:
 * - good_lp crate for MILP modeling and solving.
 *
 * AD, 2025-11-27
 */

use good_lp::{
	Expression, ProblemVariables, Solution, SolverModel, default_solver,
	variable,
};
use std::env;
use std::error::Error;

mod util;
use crate::util::{load_cost_matrix, spinner};

/// Solve ATSP and return (optimal_time, route)
fn solve_atsp(cost_matrix: Vec<Vec<f64>>) -> (f64, Vec<String>) {
	let n = cost_matrix.len();
	assert!(n > 1, "Matrix must have at least 2 nodes");

	// Compute sequential path cost for reference
	let mut seq_cost = 0.0;
	for i in 0..n - 1 {
		seq_cost += cost_matrix[i][i + 1];
	}
	seq_cost += cost_matrix[n - 1][0]; // return to Home
	println!(
		"Sequential movement (Home > 1 > 2 > ... > Home): {:.3} seconds",
		seq_cost
	);

	for row in &cost_matrix {
		assert_eq!(row.len(), n, "Matrix must be square");
	}

	// Variable creation
	let total_bin = n * (n - 1);
	println!(
		"Creating {} binary variables and {} MTZ variables...",
		total_bin, n
	);
	let mut vars = ProblemVariables::new();

	// Decision variables x[i][j] -> {0,1}, i != j
	let mut x = vec![vec![vars.add(variable().binary()); n]; n];
	for i in 0..n {
		for j in 0..n {
			if i != j {
				x[i][j] = vars.add(variable().binary());
			}
		}
	}

	// MTZ (Miller–Tucker–Zemlin) variables u[i] -> integer
	let mut u = vec![vars.add(variable().integer().min(0)); n];
	for i in 1..n {
		u[i] = vars.add(variable().integer().min(1).max((n - 1) as f64));
	}

	// Objective & Problem
	println!("Building objective with {} terms...", total_bin);
	let mut objective = Expression::default();
	for i in 0..n {
		for j in 0..n {
			if i != j {
				objective += cost_matrix[i][j] * x[i][j];
			}
		}
	}

	let mut problem = vars.minimise(objective.clone()).using(default_solver);

	// Degree constraints
	println!("Adding {} degree constraints...", 2 * n);
	for i in 0..n {
		let mut row_out = Expression::default();
		let mut row_in = Expression::default();
		for j in 0..n {
			if i != j {
				row_out += x[i][j];
				row_in += x[j][i];
			}
		}
		problem.add_constraint(row_out.eq(1.0));
		problem.add_constraint(row_in.eq(1.0));
	}

	// MTZ constraints to ensure a single Hamiltonian cycle.
	let mtz_count = (n - 1) * (n - 2);
	println!("Adding {} MTZ constraints...", mtz_count);
	for i in 1..n {
		for j in 1..n {
			if i != j {
				problem.add_constraint(
					u[i] - u[j] + ((n as f64) - 1.0) * x[i][j]
				 		<< (n as f64) - 2.0,
				);
			}
		}
	}

	// Start from HOME (u[0] = 0)
	problem.add_constraint(Expression::from(u[0]).eq(0.0));

	// Solve
	let sp_solve = spinner("Solving (MILP)...");
	let solution = problem.solve().unwrap();
	sp_solve.finish_with_message("Solved");

	let optimal_time = solution.eval(&objective);

	// Reconstruct route
	let mut route = Vec::new();
	route.push("Home".to_string());
	let mut current = 0usize;

	for _ in 0..n - 1 {
		for j in 0..n {
			if current != j && solution.value(x[current][j]) > 0.5 {
				route.push(j.to_string());
				current = j;
				break;
			}
		}
	}
	route.push("Home".to_string());

	(optimal_time, route)
}

fn main() -> Result<(), Box<dyn Error>> {
	let args: Vec<String> = env::args().collect();
	if args.len() < 2 {

	    let version = env!("CARGO_PKG_VERSION");
    	println!("ATSP Solver - Manipulator Movement Optimizer version: {}", version);
		eprintln!("Usage: {} <path_to_csv>", args[0]);
		std::process::exit(1);
	}

	let cost_matrix = load_cost_matrix(&args[1])?;
	let (time, path) = solve_atsp(cost_matrix);

	println!("Optimal movement: {:.3} seconds", time);
	println!("Route: {}", path.join(" > "));
	Ok(())
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::error::Error;

	fn assert_almost_equal(a: f64, b: f64, tol: f64) {
		assert!((a - b).abs() < tol, "Expected {:.3}, got {:.3}", b, a);
	}

	#[test]
	fn test_solve_atsp_files() -> Result<(), Box<dyn Error>> {
		let expected = 5.479;
		let tol = 0.001;

		for i in 1..=5 {
			let file_path = format!("csv/test{}.csv", i);
			let cost_matrix = load_cost_matrix(&file_path)?;
			let (time, _path) = solve_atsp(cost_matrix);
			assert_almost_equal(time, expected, tol);
		}

		Ok(())
	}
}
