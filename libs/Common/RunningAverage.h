////////////////////////////////////////////////////////////////////
// RunningAverage.h
//
// Copyright (c) 2014-2025 SEACAVE
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef _SEACAVE_RUNNING_AVERAGE_H_
#define _SEACAVE_RUNNING_AVERAGE_H_


// I N C L U D E S /////////////////////////////////////////////////


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace SEACAVE {

/**
 * @brief Template class for computing running average over a fixed-size window
 *
 * Maintains a circular buffer of the last N values with O(1) average computation.
 * Uses a running sum to avoid iterating through the buffer on each GetAverage() call.
 * Supports any numeric type.
 *
 * @tparam T The numeric type (float, double, int, etc.)
 * @tparam SIZE The window size (number of values to keep)
 * @tparam WT The working type for internal sum calculation. Defaults to T if T is floating-point, otherwise double.
 *
 * @example
 * TRunningAverage<float, 10> avgRatio;
 * avgRatio.Add(0.8f);
 * avgRatio.Add(0.85f);
 * float runningAvg = avgRatio.GetAverage();  // O(1) operation
 */
template<typename T, uint32_t SIZE, typename WT = typename std::conditional<std::is_floating_point<T>::value, T, double>::type>
class TRunningAverage
{
public:
	/**
	 * @brief Initialize the running average buffer
	 */
	TRunningAverage() : index(0), count(0), sum(T(0)) {
		static_assert(SIZE > 0, "RunningAverage window size must be greater than 0");
		std::memset(buffer, 0, sizeof(buffer));
	}

	/**
	 * @brief Add a new value to the running average
	 *
	 * Time complexity: O(1)
	 * Updates the running sum based on whether we're replacing an old value or adding to an empty slot.
	 *
	 * @param value The value to add
	 */
	TRunningAverage& Add(T value) {
		// If buffer is full, subtract the old value that we're about to overwrite
		if (count == SIZE)
			sum -= buffer[index];

		// Add the new value
		buffer[index] = value;
		sum += value;
		index = (index + 1) % SIZE;

		// Increment count until we reach SIZE
		if (count < SIZE)
			++count;
		return *this;
	}
	TRunningAverage& operator+=(T value) {
		return Add(value);
	}

	/**
	 * @brief Get the current running average
	 *
	 * Time complexity: O(1)
	 *
	 * @return The average of all values in the window (in working type WT for precision)
	 */
	WT GetAverage() const {
		if (count == 0)
			return WT(0);
		return WT(sum) / WT(count);
	}

	/**
	 * @brief Get the number of values currently in the buffer
	 * @return Number of values (0 to SIZE)
	 */
	uint32_t GetCount() const {
		return count;
	}

	/**
	 * @brief Get the window size
	 * @return Maximum number of values the buffer can hold
	 */
	static uint32_t GetWindowSize() {
		return SIZE;
	}

	/**
	 * @brief Check if the buffer is full
	 * @return true if count == SIZE
	 */
	bool IsFull() const {
		return count == SIZE;
	}

	/**
	 * @brief Clear the running average buffer
	 */
	void Clear() {
		index = 0;
		count = 0;
		sum = T(0);
		std::memset(buffer, 0, sizeof(buffer));
	}

private:
	T buffer[SIZE];     ///< Circular buffer for values
	uint32_t index;     ///< Current write position in buffer
	uint32_t count;     ///< Number of valid values in buffer
	T sum;              ///< Running sum of all values in buffer (for O(1) average)
};
/*----------------------------------------------------------------*/

} // namespace SEACAVE

#endif // _SEACAVE_RUNNING_AVERAGE_H_
