/*
* AtlasPacker.cpp
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

#include "Common.h"
#include "AtlasPacker.h"

using namespace MVS;


// S T R U C T S ///////////////////////////////////////////////////

AtlasPacker::AtlasPacker(int width, int height, bool allowRotation)
{
	Init(width, height, allowRotation);
}

void AtlasPacker::Init(int width, int height, bool allowRotation)
{
	binWidth = width;
	binHeight = height;
	this->allowRotation = allowRotation;
	usedArea = 0;
	skyline.clear();
	skyline.push_back({0, 0, binWidth});
}

float AtlasPacker::Occupancy() const
{
	return (float)usedArea / ((float)binWidth * binHeight);
}


// Compute the y-coordinate and wasted area for placing a rect at a given skyline position.
// The rect's left edge aligns with skyline[skylineIdx].x. It may span multiple segments.
// The bottom of the rect sits at the maximum height of all covered segments.
// Waste = total area between that max height and each segment's height, over the overlap width.
bool AtlasPacker::ComputeWaste(int skylineIdx, int rectWidth, int rectHeight,
                               int& outY, int& outWaste) const
{
	const int x = skyline[skylineIdx].x;

	// horizontal fit check
	if (x + rectWidth > binWidth)
		return false;

	// first pass: find the maximum height across all covered skyline segments
	outY = 0;
	int widthLeft = rectWidth;
	for (size_t i = skylineIdx; widthLeft > 0 && i < skyline.size(); ++i) {
		if (skyline[i].y > outY)
			outY = skyline[i].y;
		const int overlap = MINF(skyline[i].x + skyline[i].width, x + rectWidth) - MAXF(skyline[i].x, x);
		if (overlap > 0)
			widthLeft -= overlap;
	}

	// vertical fit check
	if (outY + rectHeight > binHeight)
		return false;

	// second pass: compute wasted area below the rect
	outWaste = 0;
	widthLeft = rectWidth;
	for (size_t i = skylineIdx; widthLeft > 0 && i < skyline.size(); ++i) {
		const int overlap = MINF(skyline[i].x + skyline[i].width, x + rectWidth) - MAXF(skyline[i].x, x);
		if (overlap > 0) {
			outWaste += (outY - skyline[i].y) * overlap;
			widthLeft -= overlap;
		}
	}

	return true;
}


// Find the best position across all skyline segments and both orientations.
// Returns the skyline index, or -1 if no valid position exists.
int AtlasPacker::FindBestPosition(int rectWidth, int rectHeight,
                                  int& outX, int& outY, bool& outRotated) const
{
	int bestWaste = INT_MAX;
	int bestY = INT_MAX;
	int bestIdx = -1;
	outRotated = false;

	const int numOrientations = (allowRotation && rectWidth != rectHeight) ? 2 : 1;

	for (int orient = 0; orient < numOrientations; ++orient) {
		const int w = (orient == 0) ? rectWidth : rectHeight;
		const int h = (orient == 0) ? rectHeight : rectWidth;

		for (size_t i = 0; i < skyline.size(); ++i) {
			int y, waste;
			if (!ComputeWaste((int)i, w, h, y, waste))
				continue;
			// prefer less waste; break ties by lower y (bottom-left)
			if (waste < bestWaste || (waste == bestWaste && y < bestY)) {
				bestWaste = waste;
				bestY = y;
				bestIdx = (int)i;
				outX = skyline[i].x;
				outY = y;
				outRotated = (orient == 1);
			}
		}
	}

	return bestIdx;
}


// Update the skyline after placing a rect at (x, y) with dimensions (w, h).
// The new top edge at y+h replaces all skyline segments in [x, x+w].
// Segments partially overlapping at the edges are clipped.
void AtlasPacker::PlaceRect(int x, int y, int rectWidth, int rectHeight)
{
	const int rectRight = x + rectWidth;
	const int newY = y + rectHeight;

	std::vector<SkylineNode> newSkyline;
	newSkyline.reserve(skyline.size() + 2);

	bool inserted = false;
	for (size_t i = 0; i < skyline.size(); ++i) {
		const int segLeft = skyline[i].x;
		const int segRight = segLeft + skyline[i].width;

		if (segRight <= x || segLeft >= rectRight) {
			// no overlap — emit segment; insert new node before first segment past the rect
			if (!inserted && segLeft >= rectRight) {
				newSkyline.push_back({x, newY, rectWidth});
				inserted = true;
			}
			newSkyline.push_back(skyline[i]);
		} else {
			// overlapping segment — clip left/right surviving parts
			if (segLeft < x)
				newSkyline.push_back({segLeft, skyline[i].y, x - segLeft});
			if (!inserted) {
				newSkyline.push_back({x, newY, rectWidth});
				inserted = true;
			}
			if (segRight > rectRight)
				newSkyline.push_back({rectRight, skyline[i].y, segRight - rectRight});
		}
	}
	if (!inserted)
		newSkyline.push_back({x, newY, rectWidth});

	skyline = std::move(newSkyline);
	MergeSkyline();
	usedArea += rectWidth * rectHeight;
}


// Merge adjacent skyline segments at the same height into a single wider segment.
void AtlasPacker::MergeSkyline()
{
	if (skyline.size() <= 1)
		return;
	size_t j = 0;
	for (size_t i = 1; i < skyline.size(); ++i) {
		if (skyline[j].y == skyline[i].y) {
			skyline[j].width += skyline[i].width;
		} else {
			++j;
			if (j != i)
				skyline[j] = skyline[i];
		}
	}
	skyline.resize(j + 1);
}


// Main entry point: pack as many rects as possible into the bin.
// Pre-sorts rects by max(width, height) descending — placing large rects first dramatically
// reduces fragmentation. For each rect, the best position (min waste, both orientations) is
// found across the entire skyline. Rects that don't fit remain in the input array.
AtlasPacker::RectWIdxArr AtlasPacker::Insert(RectWIdxArr& rects)
{
	RectWIdxArr placed(0u, rects.size());

	if (rects.empty())
		return placed;

	// sort by max dimension descending, then by area descending as tiebreaker
	rects.Sort([](const RectWIdx& a, const RectWIdx& b) {
		const int maxA = MAXF(a.rect.width, a.rect.height);
		const int maxB = MAXF(b.rect.width, b.rect.height);
		if (maxA != maxB) return maxA > maxB;
		return a.rect.area() > b.rect.area();
	});

	RectWIdxArr unplaced(0u, rects.size());

	for (uint32_t i = 0; i < rects.size(); ++i) {
		const RectWIdx& r = rects[i];
		int bestX, bestY;
		bool rotated;

		const int idx = FindBestPosition(r.rect.width, r.rect.height, bestX, bestY, rotated);

		if (idx >= 0) {
			const int w = rotated ? r.rect.height : r.rect.width;
			const int h = rotated ? r.rect.width : r.rect.height;
			PlaceRect(bestX, bestY, w, h);
			placed.emplace_back(RectWIdx{Rect(bestX, bestY, w, h), r.patchIdx});
		} else {
			unplaced.emplace_back(r);
		}
	}

	rects = std::move(unplaced);
	return placed;
}


// Compute the appropriate texture atlas size
// (an approximation since the packing is a heuristic)
// (if mult > 0, the returned size is a multiple of that value, otherwise is a power of two)
int AtlasPacker::ComputeTextureSize(const RectArr& rects, int mult)
{
	int area(0), maxSizePatch(0);
	FOREACHPTR(pRect, rects) {
		const Rect& rect = *pRect;
		area += rect.area();
		const int sizePatch(MAXF(rect.width, rect.height));
		if (maxSizePatch < sizePatch)
			maxSizePatch = sizePatch;
	}
	// compute the approximate area
	// considering the best case scenario for the packing algorithm: 0.9 fill
	area = CEIL2INT((1.f/0.9f)*(float)area);
	// compute texture size...
	const int sizeTex(MAXF(CEIL2INT(SQRT((float)area)), maxSizePatch));
	if (mult > 0) {
		// ... as multiple of mult
		return ((sizeTex+mult-1)/mult)*mult;
	}
	// ... as power of two
	return POWI(2, CEIL2INT<unsigned>(LOGN((float)sizeTex) / LOGN(2.f)));
}

int AtlasPacker::ComputeTextureSize(const RectWIdxArr& rectsWIdx, int mult)
{
	RectArr rects(rectsWIdx.GetSize());
	FOREACH(i, rectsWIdx)
		rects[i] = rectsWIdx[i].rect;
	return ComputeTextureSize(rects, mult);
}
/*----------------------------------------------------------------*/
