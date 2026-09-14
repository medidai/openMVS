////////////////////////////////////////////////////////////////////
// ImportCOLMAP.h
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef _SFM_IMPORTCOLMAP_H_
#define _SFM_IMPORTCOLMAP_H_


// I N C L U D E S /////////////////////////////////////////////////

#include "Camera.h"


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace SFM {

// forward declarations to avoid circular includes
class SFM_API Scene;

/**
 * @brief Import a COLMAP binary format scene file
 * Imports cameras, images, tracks, and image pair matches from a COLMAP binary file.
 * Images are matched to existing images by filename stem (without extension).
 * This function should be called after Import() to add COLMAP reconstruction data.
 * @param fileName input COLMAP binary file path
 * @return true on success
 */
SFM_API bool ImportCOLMAP(const String& fileName, Scene& scene,
	bool importCameras = true, bool importRelativePoses = true, bool importPoses = true, bool importTracks = true);
/*----------------------------------------------------------------*/

} // namespace SFM

#endif // _SFM_IMPORTCOLMAP_H_
