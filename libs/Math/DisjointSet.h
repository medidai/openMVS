////////////////////////////////////////////////////////////////////
// DisjointSet.h
//
// Copyright 2025 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MATH_DISJOINTSET_H_
#define _MATH_DISJOINTSET_H_


// I N C L U D E S /////////////////////////////////////////////////

#include <vector>
#include <utility>


// D E F I N E S ///////////////////////////////////////////////////


namespace SEACAVE {

// S T R U C T S ///////////////////////////////////////////////////

/**
 * @brief Disjoint-set data structure for union-find
 */
template <typename T = uint32_t>
class DisjointSet
{
public:
	typedef T Type;

protected:
	std::vector<Type> parent; // Parent pointer for each element (representative if parent[x] == x)
	std::vector<Type> rank;   // Upper bound on tree height for union-by-rank heuristic

public:
	// Initialize with each element in its own set and rank 0.
	DisjointSet(size_t n) : parent(n), rank(n, 0) {
		for (size_t i = 0; i < n; ++i)
			parent[i] = static_cast<Type>(i);
	}

	// Find representative with path compression.
	Type Find(Type x) {
		if (parent[x] != x)
			parent[x] = Find(parent[x]);
		return parent[x];
	}

	// Standard union-by-rank merge; no metadata guards.
	void Union(Type x, Type y) {
		const Type px = Find(x);
		const Type py = Find(y);
		if (px == py) return;

		if (rank[px] < rank[py]) {
			parent[px] = py;
		} else if (rank[px] > rank[py]) {
			parent[py] = px;
		} else {
			parent[py] = px;
			rank[px]++;
		}
	}

	// Union with a guard+merge callback.
	// The callback operates on the finalized root ordering (dst, src)
	// after union-by-rank selection. It must perform any necessary
	// validation and metadata merge; returning false vetoes the union.
	// Return true if the sets are now united (or were already united), false if blocked
	template <typename GuardMergeFn>
	bool UnionIf(Type x, Type y, GuardMergeFn&& guardMerge) {
		Type px = Find(x);
		Type py = Find(y);
		if (px == py)
			return true;

		// Decide destination/source roots using rank heuristic
		Type dst = px;
		Type src = py;
		if (rank[dst] < rank[src])
			std::swap(dst, src);

		// Callback performs guard and merge; veto if false
		if (!guardMerge(dst, src))
			return false;

		parent[src] = dst;
		if (rank[dst] == rank[src])
			++rank[dst];
		return true;
	}
};
/*----------------------------------------------------------------*/

} // namespace SEACAVE

#endif // _MATH_DISJOINTSET_H_
