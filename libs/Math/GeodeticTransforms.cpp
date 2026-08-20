////////////////////////////////////////////////////////////////////
// GeodeticTransforms.cpp
//
// Copyright 2025 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#include "Common.h"
#include "GeodeticTransforms.h"

using namespace SEACAVE;


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

void SEACAVE::WGS84ToECEF(double lat_deg, double lon_deg, double alt,
                          double& x, double& y, double& z)
{
	const double lat = D2R(lat_deg);
	const double lon = D2R(lon_deg);
	
	const double sin_lat = SIN(lat);
	const double cos_lat = COS(lat);
	const double sin_lon = SIN(lon);
	const double cos_lon = COS(lon);
	
	// Radius of curvature in prime vertical (N)
	const double N = WGS84::A / SQRT(1.0 - WGS84::E2 * sin_lat * sin_lat);
	
	x = (N + alt) * cos_lat * cos_lon;
	y = (N + alt) * cos_lat * sin_lon;
	z = (N * (1.0 - WGS84::E2) + alt) * sin_lat;
}

void SEACAVE::ECEFToWGS84(double x, double y, double z,
                          double& lat_deg, double& lon_deg, double& alt)
{
	const double p = SQRT(x * x + y * y);
	const double theta = ATAN2(z * WGS84::A, p * WGS84::B);
	
	const double sin_theta = SIN(theta);
	const double cos_theta = COS(theta);
	
	const double lat = ATAN2(
		z + WGS84::E_PRIME2 * WGS84::B * sin_theta * sin_theta * sin_theta,
		p - WGS84::E2 * WGS84::A * cos_theta * cos_theta * cos_theta
	);
	
	const double lon = ATAN2(y, x);
	
	const double sin_lat = SIN(lat);
	const double N = WGS84::A / SQRT(1.0 - WGS84::E2 * sin_lat * sin_lat);
	alt = p / COS(lat) - N;
	
	lat_deg = R2D(lat);
	lon_deg = R2D(lon);
}

void SEACAVE::ECEFToENU(double x, double y, double z,
                        double x0, double y0, double z0,
                        double lat0_deg, double lon0_deg,
                        double& east, double& north, double& up)
{
	const double lat0 = D2R(lat0_deg);
	const double lon0 = D2R(lon0_deg);
	
	const double sin_lat = SIN(lat0);
	const double cos_lat = COS(lat0);
	const double sin_lon = SIN(lon0);
	const double cos_lon = COS(lon0);
	
	// Translation: point relative to origin
	const double dx = x - x0;
	const double dy = y - y0;
	const double dz = z - z0;
	
	// Rotation matrix from ECEF to ENU
	east  = -sin_lon * dx + cos_lon * dy;
	north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz;
	up    =  cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz;
}

void SEACAVE::ENUToECEF(double east, double north, double up,
                        double x0, double y0, double z0,
                        double lat0_deg, double lon0_deg,
                        double& x, double& y, double& z)
{
	const double lat0 = D2R(lat0_deg);
	const double lon0 = D2R(lon0_deg);
	
	const double sin_lat = SIN(lat0);
	const double cos_lat = COS(lat0);
	const double sin_lon = SIN(lon0);
	const double cos_lon = COS(lon0);
	
	// Inverse rotation matrix from ENU to ECEF
	const double dx = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up;
	const double dy =  cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up;
	const double dz =                    cos_lat * north           + sin_lat * up;
	
	x = x0 + dx;
	y = y0 + dy;
	z = z0 + dz;
}

void SEACAVE::WGS84ToENU(double lat_deg, double lon_deg, double alt,
                         double lat0_deg, double lon0_deg, double alt0,
                         double& east, double& north, double& up)
{
	// Convert both points to ECEF
	double x, y, z, x0, y0, z0;
	WGS84ToECEF(lat_deg, lon_deg, alt, x, y, z);
	WGS84ToECEF(lat0_deg, lon0_deg, alt0, x0, y0, z0);
	
	// Convert to ENU
	ECEFToENU(x, y, z, x0, y0, z0, lat0_deg, lon0_deg, east, north, up);
}

void SEACAVE::ENUToWGS84(double east, double north, double up,
                         double lat0_deg, double lon0_deg, double alt0,
                         double& lat_deg, double& lon_deg, double& alt)
{
	// Convert origin to ECEF
	double x0, y0, z0;
	WGS84ToECEF(lat0_deg, lon0_deg, alt0, x0, y0, z0);
	
	// Convert ENU to ECEF
	double x, y, z;
	ENUToECEF(east, north, up, x0, y0, z0, lat0_deg, lon0_deg, x, y, z);
	
	// Convert ECEF to WGS84
	ECEFToWGS84(x, y, z, lat_deg, lon_deg, alt);
}
/*----------------------------------------------------------------*/
