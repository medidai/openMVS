////////////////////////////////////////////////////////////////////
// LeastAbsoluteDeviationSolver.cpp
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#include "Common.h"
#include "LeastAbsoluteDeviationSolver.h"
#include <Eigen/SparseCholesky>
#ifdef _USE_SUITESPARSE
#include <Eigen/CholmodSupport>
#endif

using namespace SEACAVE;


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace SEACAVE {

struct LeastAbsoluteDeviationLinearSolverImpl
{
	virtual ~LeastAbsoluteDeviationLinearSolverImpl() = default;
	virtual bool Compute(const Eigen::SparseMatrix<double>& A) = 0;
	virtual bool Solve(const Eigen::VectorXd& b, Eigen::VectorXd* x) = 0;
};

namespace {

Eigen::VectorXd Shrinkage(const Eigen::VectorXd& a, const double kappa)
{
	const Eigen::VectorXd a_plus_kappa = a.array() + kappa;
	const Eigen::VectorXd a_minus_kappa = a.array() - kappa;
	return a_plus_kappa.cwiseMin(0) + a_minus_kappa.cwiseMax(0);
}

struct SimplicialLLTLinearSolver
    : public LeastAbsoluteDeviationLinearSolverImpl
{
	bool Compute(const Eigen::SparseMatrix<double>& A) override
	{
		linear_solver_.compute(A.transpose() * A);
		return linear_solver_.info() == Eigen::Success;
	}

	bool Solve(const Eigen::VectorXd& b, Eigen::VectorXd* x) override
	{
		x->noalias() = linear_solver_.solve(b);
		return linear_solver_.info() == Eigen::Success;
	}

	private:
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> linear_solver_;
};

#ifdef _USE_SUITESPARSE
struct SupernodalCholmodLLTLinearSolver
    : public LeastAbsoluteDeviationLinearSolverImpl
{
	bool Compute(const Eigen::SparseMatrix<double>& A) override
	{
		linear_solver_.compute(A.transpose() * A);
		return linear_solver_.info() == Eigen::Success;
	}

	bool Solve(const Eigen::VectorXd& b, Eigen::VectorXd* x) override
	{
		x->noalias() = linear_solver_.solve(b);
		return linear_solver_.info() == Eigen::Success;
	}

	private:
	Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> linear_solver_;
};
#endif

std::shared_ptr<LeastAbsoluteDeviationLinearSolverImpl> CreateLinearSolver(
    const LeastAbsoluteDeviationSolver::Options::SolverType& solver_type,
    const Eigen::SparseMatrix<double>& A)
{
	switch (solver_type) {
	case LeastAbsoluteDeviationSolver::Options::SolverType::SimplicialLLT:
		return std::make_shared<SimplicialLLTLinearSolver>();
	#ifdef _USE_SUITESPARSE
	case LeastAbsoluteDeviationSolver::Options::SolverType::SupernodalCholmodLLT:
		return std::make_shared<SupernodalCholmodLLTLinearSolver>();
	#endif
	default:
		VERBOSE("error: unknown linear solver type, using SimplicialLLT");
		return std::make_shared<SimplicialLLTLinearSolver>();
	}
}

} // namespace

LeastAbsoluteDeviationSolver::LeastAbsoluteDeviationSolver(
    const Options& options, const Eigen::SparseMatrix<double>& A) :
    options_(options),
    A_(A),
    linear_solver_(CreateLinearSolver(options_.solver_type, A))
{
	ASSERT(options_.rho > 0);
	ASSERT(options_.alpha > 0);
	ASSERT(options_.max_num_iterations > 0);
	ASSERT(options_.absolute_tolerance >= 0);
	ASSERT(options_.relative_tolerance >= 0);
	if (A.rows() < A.cols()) {
		DEBUG("warning: underdetermined systems may not be well-supported");
	}

	linear_solver_->Compute(A_);
}

bool LeastAbsoluteDeviationSolver::Solve(const Eigen::VectorXd& b,
                                         Eigen::VectorXd* x) const
{
	ASSERT(x != nullptr);

	Eigen::VectorXd z = Eigen::VectorXd::Zero(A_.rows());
	Eigen::VectorXd z_old(A_.rows());
	Eigen::VectorXd u = Eigen::VectorXd::Zero(A_.rows());

	Eigen::VectorXd Ax(A_.rows());
	Eigen::VectorXd Ax_hat(A_.rows());

	const double b_norm = b.norm();
	const double eps_pri_threshold =
	    std::sqrt(A_.rows()) * options_.absolute_tolerance;
	const double eps_dual_threshold =
	    std::sqrt(A_.cols()) * options_.absolute_tolerance;

	for (int i = 0; i < options_.max_num_iterations; ++i) {
		if (!linear_solver_->Solve(A_.transpose() * (b + z - u), x)) {
			return false;
		}

		Ax.noalias() = A_ * *x;
		Ax_hat.noalias() = options_.alpha * Ax + (1 - options_.alpha) * (z + b);

		std::swap(z, z_old);
		z.noalias() = Shrinkage(Ax_hat - b + u, 1 / options_.rho);

		u.noalias() += Ax_hat - z - b;

		const double r_norm = (Ax - z - b).norm();
		const double s_norm = (-options_.rho * A_.transpose() * (z - z_old)).norm();
		const double eps_pri =
		    eps_pri_threshold + options_.relative_tolerance * std::max(b_norm, std::max(Ax.norm(), z.norm()));
		const double eps_dual =
		    eps_dual_threshold + options_.relative_tolerance * (options_.rho * A_.transpose() * u).norm();

		if (r_norm < eps_pri && s_norm < eps_dual) {
			break;
		}
	}

	return true;
}
/*----------------------------------------------------------------*/


bool TestLeastAbsoluteDeviationSolver()
{
	// Test case: solve a simple least absolute deviation problem
	// Create a sparse matrix A (3x2) and vector b (3x1)
	// Problem: min || A x - b ||_1

	// A = [[1, 0], [0, 1], [1, 1]]
	// b = [1, 2, 2]
	// Expected solution minimizes sum of absolute residuals:
	// residuals = [x0-1, x1-2, (x0+x1)-2]

	typedef Eigen::Triplet<double> T;
	std::vector<T> triplets;
	triplets.push_back(T(0, 0, 1.0));
	triplets.push_back(T(1, 1, 1.0));
	triplets.push_back(T(2, 0, 1.0));
	triplets.push_back(T(2, 1, 1.0));

	Eigen::SparseMatrix<double> A(3, 2);
	A.setFromTriplets(triplets.begin(), triplets.end());

	Eigen::VectorXd b(3);
	b << 1.0, 2.0, 2.0;

	LeastAbsoluteDeviationSolver::Options options;
	options.max_num_iterations = 1000;
	options.absolute_tolerance = 1e-4;
	options.relative_tolerance = 1e-2;

	LeastAbsoluteDeviationSolver solver(options, A);

	Eigen::VectorXd x = Eigen::VectorXd::Zero(2);
	if (!solver.Solve(b, &x)) {
		VERBOSE("ERROR: LeastAbsoluteDeviationSolver::Solve failed!");
		return false;
	}

	// Verify solution is reasonably close (solver uses ADMM with tolerances)
	// The solver should converge to minimize || A x - b ||_1
	const Eigen::VectorXd residuals = A * x - b;
	const double l1_norm = residuals.lpNorm<1>();

	// For this problem: A = [[1,0], [0,1], [1,1]], b = [1,2,2]
	// Expected solution: x ≈ [2/3, 5/3] which minimizes L1 residuals
	const double expected_x0 = 2.0 / 3.0; // 0.666667
	const double expected_x1 = 5.0 / 3.0; // 1.666667
	const double x_tolerance = 0.01; // 1% tolerance for solution values

	// Check solution values
	if (ABS(x(0) - expected_x0) > x_tolerance || ABS(x(1) - expected_x1) > x_tolerance) {
		VERBOSE("ERROR: LeastAbsoluteDeviationSolver solution incorrect: "
		        "got x=[%f, %f], expected x=[%f, %f]",
		        x(0), x(1), expected_x0, expected_x1);
		return false;
	}

	// Check L1 norm (should be minimal, around 1.0 for this problem)
	if (l1_norm > 1.5) {
		VERBOSE("ERROR: LeastAbsoluteDeviationSolver L1 norm too large: %f "
		        "(expected around 1.0)",
		        l1_norm);
		return false;
	}

	VERBOSE("LeastAbsoluteDeviationSolver test passed (x=[%f, %f], L1=%f)",
	        x(0), x(1), l1_norm);
	return true;
}
/*----------------------------------------------------------------*/

} // namespace SEACAVE
