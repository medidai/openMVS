////////////////////////////////////////////////////////////////////
// GeodeticTransforms.h
//
// Copyright 2025 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef _MATH_GEODETICTRANSFORMS_H_
#define _MATH_GEODETICTRANSFORMS_H_


// I N C L U D E S /////////////////////////////////////////////////


// D E F I N E S ///////////////////////////////////////////////////


namespace SEACAVE {

// S T R U C T S ///////////////////////////////////////////////////

/**
 * @brief WGS84 (World Geodetic System 1984) ellipsoid constants
 *
 * Standard reference ellipsoid used by GPS and most global coordinate systems.
 */
struct WGS84 {
	static constexpr double A = 6378137.0;              ///< Semi-major axis (equatorial radius) in meters
	static constexpr double F = 1.0 / 298.257223563;    ///< Flattening
	static constexpr double B = A * (1.0 - F);          ///< Semi-minor axis (polar radius) ≈ 6356752.314245 m
	static constexpr double E2 = 2.0 * F - F * F;       ///< First eccentricity squared ≈ 0.00669437999014
	static constexpr double E_PRIME2 = (A * A - B * B) / (B * B); ///< Second eccentricity squared
};
/*----------------------------------------------------------------*/

/**
 * @brief Convert WGS84 geodetic coordinates to ECEF (Earth-Centered Earth-Fixed)
 *
 * @param lat_deg Latitude in degrees (positive North, negative South)
 * @param lon_deg Longitude in degrees (positive East, negative West)
 * @param alt Altitude in meters above WGS84 ellipsoid (NOT above sea level)
 * @param x Output: ECEF X coordinate in meters
 * @param y Output: ECEF Y coordinate in meters
 * @param z Output: ECEF Z coordinate in meters
 *
 * ECEF origin is Earth's center of mass, Z-axis through North Pole, X-axis through Prime Meridian.
 */
MATH_API void WGS84ToECEF(double lat_deg, double lon_deg, double alt,
                          double& x, double& y, double& z);

/**
 * @brief Convert ECEF coordinates to WGS84 geodetic coordinates
 *
 * Uses iterative Bowring method for accurate conversion.
 *
 * @param x ECEF X coordinate in meters
 * @param y ECEF Y coordinate in meters
 * @param z ECEF Z coordinate in meters
 * @param lat_deg Output: Latitude in degrees
 * @param lon_deg Output: Longitude in degrees
 * @param alt Output: Altitude in meters above WGS84 ellipsoid
 */
MATH_API void ECEFToWGS84(double x, double y, double z,
                          double& lat_deg, double& lon_deg, double& alt);

/**
 * @brief Convert ECEF coordinates to local ENU (East-North-Up) frame
 *
 * @param x ECEF X coordinate of point to convert
 * @param y ECEF Y coordinate of point to convert
 * @param z ECEF Z coordinate of point to convert
 * @param x0 ECEF X coordinate of local origin
 * @param y0 ECEF Y coordinate of local origin
 * @param z0 ECEF Z coordinate of local origin
 * @param lat0_deg Latitude of local origin in degrees (for rotation matrix)
 * @param lon0_deg Longitude of local origin in degrees (for rotation matrix)
 * @param east Output: East coordinate in meters (X-axis of ENU frame)
 * @param north Output: North coordinate in meters (Y-axis of ENU frame)
 * @param up Output: Up coordinate in meters (Z-axis of ENU frame, vertical)
 *
 * ENU is a local Cartesian coordinate system with origin at (lat0, lon0, alt0):
 * - East axis: tangent to ellipsoid, pointing East
 * - North axis: tangent to ellipsoid, pointing North
 * - Up axis: normal to ellipsoid, pointing away from Earth's center
 */
MATH_API void ECEFToENU(double x, double y, double z,
                        double x0, double y0, double z0,
                        double lat0_deg, double lon0_deg,
                        double& east, double& north, double& up);

/**
 * @brief Convert local ENU coordinates to ECEF
 *
 * Inverse of ECEFToENU.
 *
 * @param east East coordinate in meters
 * @param north North coordinate in meters
 * @param up Up coordinate in meters
 * @param x0 ECEF X coordinate of local origin
 * @param y0 ECEF Y coordinate of local origin
 * @param z0 ECEF Z coordinate of local origin
 * @param lat0_deg Latitude of local origin in degrees
 * @param lon0_deg Longitude of local origin in degrees
 * @param x Output: ECEF X coordinate
 * @param y Output: ECEF Y coordinate
 * @param z Output: ECEF Z coordinate
 */
MATH_API void ENUToECEF(double east, double north, double up,
                        double x0, double y0, double z0,
                        double lat0_deg, double lon0_deg,
                        double& x, double& y, double& z);

/**
 * @brief Convert WGS84 geodetic coordinates directly to local ENU frame
 *
 * Convenience function combining WGS84ToECEF and ECEFToENU.
 *
 * @param lat_deg Latitude of point to convert (degrees)
 * @param lon_deg Longitude of point to convert (degrees)
 * @param alt Altitude of point to convert (meters above ellipsoid)
 * @param lat0_deg Latitude of local origin (degrees)
 * @param lon0_deg Longitude of local origin (degrees)
 * @param alt0 Altitude of local origin (meters above ellipsoid)
 * @param east Output: East coordinate in meters
 * @param north Output: North coordinate in meters
 * @param up Output: Up coordinate in meters
 */
MATH_API void WGS84ToENU(double lat_deg, double lon_deg, double alt,
                         double lat0_deg, double lon0_deg, double alt0,
                         double& east, double& north, double& up);

/**
 * @brief Convert local ENU coordinates directly to WGS84 geodetic coordinates
 *
 * Convenience function combining ENUToECEF and ECEFToWGS84.
 *
 * @param east East coordinate in meters
 * @param north North coordinate in meters
 * @param up Up coordinate in meters
 * @param lat0_deg Latitude of local origin (degrees)
 * @param lon0_deg Longitude of local origin (degrees)
 * @param alt0 Altitude of local origin (meters above ellipsoid)
 * @param lat_deg Output: Latitude (degrees)
 * @param lon_deg Output: Longitude (degrees)
 * @param alt Output: Altitude (meters above ellipsoid)
 */
MATH_API void ENUToWGS84(double east, double north, double up,
                         double lat0_deg, double lon0_deg, double alt0,
                         double& lat_deg, double& lon_deg, double& alt);
/*----------------------------------------------------------------*/

} // namespace SEACAVE

#endif // _MATH_GEODETICTRANSFORMS_H_
