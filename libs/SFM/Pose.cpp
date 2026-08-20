////////////////////////////////////////////////////////////////////
// Pose.cpp
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#include "Common.h"
#include "Pose.h"

using namespace SFM;


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

Matrix4x4 Pose3D::GetP4fromRC() const {
	Matrix3x4 P3;
	AssembleProjectionMatrix(R, C, P3);
	Matrix4x4 RC4;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 4; ++j)
			RC4(i, j) = P3(i, j);
	RC4(3, 0) = RC4(3, 1) = RC4(3, 2) = 0;
	RC4(3, 3) = 1;
	return RC4;
}
PMatrix Pose3D::GetPfromRC() const
{
	PMatrix P;
	AssembleProjectionMatrix(R, C, P);
	return P;
}
void Pose3D::DecomposePfromRC(const PMatrix& P)
{
	DecomposeProjectionMatrix(P, R, C);
} // DecomposeP_RC
/*----------------------------------------------------------------*/
