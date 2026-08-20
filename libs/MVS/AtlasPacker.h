/*
* AtlasPacker.h
*
* Copyright (c) 2014-2025 SEACAVE
*
* Author(s):
*
*      cDc <cdc.seacave@gmail.com>
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*
* Additional Terms:
*
*      You are required to preserve legal notices and author attributions in
*      that material or in the Appropriate Legal Notices displayed by works
*      containing it.
*/

#ifndef _MVS_ATLASPACKER_H_
#define _MVS_ATLASPACKER_H_


// I N C L U D E S /////////////////////////////////////////////////


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {

// Skyline-based rectangle bin packer with min-waste heuristic and optional 90-degree rotation.
//
// Achieves higher packing efficiency than standard MaxRects/Skyline/Guillotine by combining:
//   - Pre-sorting rects by max dimension (descending) to place large rects first
//   - Skyline representation for O(S) position lookup per rect (S = skyline segments, typically << N)
//   - Min-waste scoring that penalizes positions creating dead space below the rect
//   - 90-degree rotation to find tighter fits
//
// The skyline tracks the top contour of placed rectangles as a list of horizontal segments.
// For each candidate rect, every skyline segment is evaluated as a potential left edge. The
// rect may span multiple segments; its bottom sits at the max height of all covered segments.
// The "waste" is the area between the rect's bottom and the existing skyline — dead space that
// can never be used. Minimizing waste produces denser packings than simple bottom-left placement.
//
// Time complexity:  O(N * S) per insertion, O(N^2 * S) total (S typically O(sqrt(N)))
// Space complexity: O(N + S)
// Typical occupancy: 85-95% depending on rect size distribution
class AtlasPacker
{
public:
	// A simple rectangle
	typedef cv::Rect Rect;
	// A rectangle that stores an index of origin
	typedef struct {
		Rect rect;
		uint32_t patchIdx;
	} RectWIdx;
	/// A list of rectangles
	typedef CLISTDEF0(Rect) RectArr;
	/// A list of rectangles along their original indices
	typedef CLISTDEF0(RectWIdx) RectWIdxArr;

	/// Instantiates a bin of the given size.
	/// @param allowRotation If true, rects may be rotated 90 degrees for better fit.
	AtlasPacker(int width, int height, bool allowRotation = true);

	/// (Re)initializes the packer to an empty bin.
	void Init(int width, int height, bool allowRotation = true);

	/// Inserts the given list of rectangles. Rects are pre-sorted internally for optimal
	/// packing. Placed rects are removed from the input array and returned with their final
	/// positions (x, y, width, height — with width/height possibly swapped if rotated).
	/// Rects that don't fit remain in the input array.
	RectWIdxArr Insert(RectWIdxArr& rects);

	/// Computes the ratio of used surface area to the total bin area.
	float Occupancy() const;

	/// Computes an approximate texture atlas size.
	/// If mult > 0, the returned size is a multiple of that value; otherwise a power of two.
	static int ComputeTextureSize(const RectArr& rects, int mult = 0);
	static int ComputeTextureSize(const RectWIdxArr& rects, int mult = 0);

	/// Returns true if rect a is fully contained within rect b.
	static inline bool IsContainedIn(const Rect& a, const Rect& b) {
		return a.x >= b.x && a.y >= b.y
			&& a.x+a.width <= b.x+b.width
			&& a.y+a.height <= b.y+b.height;
	}

protected:
	struct SkylineNode {
		int x;     // left x coordinate of the segment
		int y;     // height (top edge) of the segment
		int width; // width of the segment
	};

	int binWidth;
	int binHeight;
	bool allowRotation;
	std::vector<SkylineNode> skyline;
	int usedArea;

	/// Find the best skyline position for a rect of given size.
	int FindBestPosition(int rectWidth, int rectHeight,
	                     int& outX, int& outY, bool& outRotated) const;

	/// Compute the placement y-coordinate and wasted area at a given skyline segment.
	bool ComputeWaste(int skylineIdx, int rectWidth, int rectHeight,
	                  int& outY, int& outWaste) const;

	/// Update the skyline after placing a rect at the given position.
	void PlaceRect(int x, int y, int rectWidth, int rectHeight);

	/// Merge adjacent skyline segments at the same height.
	void MergeSkyline();
};
/*----------------------------------------------------------------*/

} // namespace MVS

#endif // _MVS_ATLASPACKER_H_
