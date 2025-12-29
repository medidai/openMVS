////////////////////////////////////////////////////////////////////
// ConfidenceInterval.h
//
// Copyright 2025 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MATH_CONFIDENCEINTERVAL_H_
#define _MATH_CONFIDENCEINTERVAL_H_


// I N C L U D E S /////////////////////////////////////////////////

#include <boost/math/distributions/students_t.hpp>


// D E F I N E S ///////////////////////////////////////////////////


namespace SEACAVE {

// S T R U C T S ///////////////////////////////////////////////////

/**
 * @brief Calculates the two-tailed t-critical value for a given confidence level and degrees of freedom.
 *
 * @tparam T The floating point type (float or double).
 * @param samples_size The size of the sample (number of data points).
 * @param confidenceLevel The desired confidence level (e.g., 0.95 for 95%).
 * @return The positive t-critical value as type T.
 * @throws std::runtime_error if confidence level is invalid (delegated from Boost).
 * @throws std::domain_error if degrees_of_freedom is not positive (from Boost).
 */
template <typename T>
T ComputeTCriticalValue(size_t samples_size, T confidenceLevel) {
	static_assert(std::is_floating_point<T>::value, "T must be a floating point type");
	ASSERT(samples_size > 1);
	ASSERT(confidenceLevel > T(0) && confidenceLevel < T(1));

	// Boost.Math typically uses double for precision in distributions
	const double degrees_of_freedom = static_cast<double>(samples_size - 1);
	const double alpha = 1.0 - static_cast<double>(confidenceLevel); // Significance level
	const double alpha_over_2 = alpha / 2.0;	 // For two-tailed interval

	// Create a Student's t-distribution object
	boost::math::students_t dist(degrees_of_freedom);

	// Calculate the t-critical value using the quantile function (inverse CDF)
	// We need the upper critical value, so we use quantile(complement(alpha/2)).
	// quantile(complement(p)) gives the value x such that P(X > x) = p.
	// This corresponds to the positive t-value for the upper tail.
	double tCritical = boost::math::quantile(boost::math::complement(dist, alpha_over_2));

	return static_cast<T>(tCritical);
}

// Structure to hold the confidence interval bounds
template <typename T>
struct TConfidenceInterval {
	T lowerBound;
	T upperBound;
    union {
	    T tCritical;
        T mean;
    };
};

/**
 * @brief Calculates the confidence interval for a sample using the Student's t-distribution.
 * Uses Boost.Math to determine the t-critical value.
 *
 * @tparam T The floating point type (float or double) for data and calculations.
 * @param data A vector of sample data points of type T.
 * @param confidenceLevel The desired confidence level (e.g., 0.95 for 95%) as type T.
 * @return A ConfidenceInterval<T> struct containing the lower and upper bounds and the t-critical value used.
 * @throws std::runtime_error if the sample size is less than 2 or confidence level is invalid.
 */
template <typename T>
TConfidenceInterval<T> ComputeConfidenceIntervalTCritical(const std::vector<T>& data, T confidenceLevel) {
	static_assert(std::is_floating_point<T>::value, "T must be a floating point type");
	const size_t n = data.size();
	ASSERT(n > 1);
	ASSERT(confidenceLevel > T(0) && confidenceLevel < T(1));

	// Calculate the mean and standard deviation using MeanStd
	const MeanStd<T> meanStdDev(data.data(), n);
	const T mean = meanStdDev.GetMean();

	// Determine the t-critical value using the helper function
	T tCritical = ComputeTCriticalValue<T>(n, confidenceLevel);

	// Calculate the margin of error:
	//   Standard Error of the Mean (SEM) = sampleStdDev / sqrt(n)
	//   Margin of Error (ME) = tCritical * SEM
	T marginOfError = tCritical * (meanStdDev.GetSampleStdDev() / std::sqrt(static_cast<T>(n)));

	// Calculate the confidence interval
	return TConfidenceInterval<T>{
		mean - marginOfError,
		mean + marginOfError,
		tCritical
	};
}


// Compute the Median of a vector
template<typename T>
T Median(std::vector<T>& v) {
    size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n/2, v.end());
    T med = v[n/2];
    if (n % 2 == 0) {
        std::nth_element(v.begin(), v.begin() + n/2 - 1, v.end());
        med = (med + v[n/2 - 1]) / 2;
    }
    return med;
}
template<typename T>
T Median(const std::vector<T>& v) {
    std::vector<T> copy(v);
    return Median(copy);
}

// Compute X84 confidence interval as in:
// "Robust Statistics: the Approach Based on Influence Functions", Hampel et al. 1986
template<typename T>
TConfidenceInterval<T> ComputeConfidenceIntervalX84(std::vector<T>& data, double confidence = 0.95) {
    static_assert(std::is_floating_point<T>::value, "Template parameter must be float or double.");

    size_t n = data.size();
    if (n == 0) throw std::invalid_argument("Input data is empty.");

    // 1. Compute the median
    T med = Median(data);

    // 2. Compute absolute deviations from the median
    std::vector<T> abs_devs(n);
    for (size_t i = 0; i < n; ++i)
        abs_devs[i] = std::abs(data[i] - med);

    // 3. Compute MAD (Median Absolute Deviation)
    T mad = Median(abs_devs);

    // 4. X84 threshold: 5.2 * MAD
    const T threshold = T(5.2) * mad;

    // 5. Select inliers: |x_i - med| < threshold
    std::vector<T> inliers;
    for (size_t i = 0; i < n; ++i)
        if (std::abs(data[i] - med) < threshold)
            inliers.push_back(data[i]);
    if (inliers.empty()) throw std::runtime_error("No inliers found by X84 rule.");

    // 6. Robust mean and scale estimation from inliers
    T robust_mean = Median(inliers); // or use mean of inliers for more efficiency
    std::vector<T> inlier_devs(inliers.size());
    for (size_t i = 0; i < inliers.size(); ++i)
        inlier_devs[i] = std::abs(inliers[i] - robust_mean);
    T robust_mad = Median(inlier_devs);

    // 7. Convert MAD to robust estimate of standard deviation
    // For normal distribution, sigma ≈ 1.4826 * MAD
    T robust_sigma = static_cast<T>(1.4826) * robust_mad;

    // 8. Compute confidence interval (normal approximation)
    // z_alpha/2 for 95% confidence ≈ 1.96
    double z = 1.96;
    if (confidence == 0.99) z = 2.576;
    else if (confidence == 0.90) z = 1.645;

    T margin = static_cast<T>(z) * robust_sigma / std::sqrt(static_cast<T>(inliers.size()));

	// Calculate the confidence interval
	return TConfidenceInterval<T>{
		robust_mean - margin,
		robust_mean + margin,
		robust_mean
	};
}
template<typename T>
TConfidenceInterval<T> ComputeConfidenceIntervalX84(const std::vector<T>& data, double confidence = 0.95) {
    std::vector<T> copy(data);
    return ComputeConfidenceIntervalX84(copy, confidence);
}
/*----------------------------------------------------------------*/



/**
 * @brief Test function for ConfidenceInterval calculations.
 *
 * This function tests the confidence interval calculation using a sample of values.
 * It verifies the calculated t-critical values and confidence interval bounds against expected values.
 *
 * @return true if all tests pass, false otherwise.
 */
static bool TestConfidenceInterval() {
	// Example: A vector of 3D reprojection errors (in pixels)
	std::vector<double> reprojection_errors{
		0.8, 1.1, 0.5, 1.5, 0.9, 1.2, 0.7, 1.0, 1.3, 0.6,
		1.4, 0.8, 0.9, 1.1, 1.0
	}; // n = 15
	constexpr double eps = 1.e-4;

	// --- Calculate 95% Confidence Interval ---
	constexpr double confidence_95 = 0.95;
	TConfidenceInterval<double> ci_95 = ComputeConfidenceIntervalTCritical(reprojection_errors, confidence_95);
	// Expected values for 95% CI:
	// t-critical: 2.1448
	// Interval: [0.8274, 1.1459]
	if (!equal(ci_95.tCritical, 2.144786, eps))
		return false;
	if (!equal(ci_95.lowerBound, 0.827444, eps))
		return false;
	if (!equal(ci_95.upperBound, 1.145888, eps))
		return false;

	// --- Calculate 99% Confidence Interval ---
	constexpr double confidence_99 = 0.99;
	TConfidenceInterval<double> ci_99 = ComputeConfidenceIntervalTCritical(reprojection_errors, confidence_99);
	// Expected values for 99% CI:
	// t-critical: 2.9768
	// Interval: [0.7657, 1.2077]
	if (!equal(ci_99.tCritical, 2.97684, eps))
		return false;
	if (!equal(ci_99.lowerBound, 0.76567, eps))
		return false;
	if (!equal(ci_99.upperBound, 1.20765, eps))
		return false;

    // --- Calculate X84 Confidence Interval ---
    TConfidenceInterval<double> x84_ci = ComputeConfidenceIntervalX84(reprojection_errors, confidence_95);
	// Expected values for 95% CI:
	// mean: 1
	// Interval: [0.84994, 1.15006]
	if (!equal(x84_ci.mean, 1.0, eps))
		return false;
	if (!equal(x84_ci.lowerBound, 0.84994, eps))
		return false;
	if (!equal(x84_ci.upperBound, 1.15006, eps))
		return false;
	return true; // Indicate success
}
/*----------------------------------------------------------------*/

} // namespace SEACAVE

#endif // _MATH_CONFIDENCEINTERVAL_H_
