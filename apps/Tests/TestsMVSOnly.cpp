/*
 * TestsMVSOnly.cpp
 *
 * Copyright (c) 2014-2026 SEACAVE
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#include "../../libs/MVS.h"
#include "TestsMVS.h"

#define APPNAME _T("Tests")

DEFINE_LOG_NAME(lt, _T("Test    "));

int main()
{
	#ifdef _MSC_VER
	std::setvbuf(stdout, NULL, _IONBF, 0);
	std::setvbuf(stderr, NULL, _IONBF, 0);
	#else
	std::setvbuf(stdout, NULL, _IOLBF, 0);
	std::setvbuf(stderr, NULL, _IOLBF, 0);
	#endif
	OPEN_LOG();
	OPEN_LOGCONSOLE();
	Initialize(APPNAME);
	const bool success = MVS::ConfidenceCompat23Test() && MVS::DMapCompat23Test();
	Finalize();
	CLOSE_LOGCONSOLE();
	CLOSE_LOG();
	return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
/*----------------------------------------------------------------*/
