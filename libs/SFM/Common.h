////////////////////////////////////////////////////////////////////
// Common.h
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef _SFM_COMMON_H_
#define _SFM_COMMON_H_


// I N C L U D E S /////////////////////////////////////////////////

#if defined(SFM_EXPORTS) && !defined(Common_EXPORTS)
#define Common_EXPORTS
#endif

#include "../Common/Common.h"
#include "../Math/Common.h"
#include "../IO/Common.h"
#include "../Common/BS_thread_pool.hpp"

#ifndef SFM_API
#define SFM_API GENERAL_API
#endif
#ifndef SFM_TPL
#define SFM_TPL GENERAL_TPL
#endif


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////


#endif // _SFM_COMMON_H_

