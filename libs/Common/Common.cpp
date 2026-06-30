////////////////////////////////////////////////////////////////////
// Common.cpp
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

// Source file that includes just the standard includes
// Common.pch will be the pre-compiled header
// Common.obj will contain the pre-compiled type information

#include "Common.h"

// On Linux the upstream Breakpad MiniDumper is Windows-only, so a crash (e.g. a
// SIGSEGV deep inside dense reconstruction) dies with no backtrace - the parent
// only sees rc=139. Install a lightweight execinfo-based signal handler that
// prints a symbolized backtrace to stderr (captured by the pipeline logs) and
// then re-raises the signal so the process still core-dumps as before.
#if defined(__linux__)
#include <execinfo.h>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <unistd.h>

namespace {
void SeacaveCrashSignalHandler(int sig) {
	// backtrace()/backtrace_symbols_fd() are the async-signal-safe pair
	// (unlike backtrace_symbols(), which allocates); keep everything else here
	// minimal and reentrancy-friendly since we run from a crashing context.
	void* frames[64];
	const int numFrames = backtrace(frames, (int)(sizeof(frames) / sizeof(frames[0])));
	char header[96];
	const int len = snprintf(header, sizeof(header),
		"\n=== OpenMVS fatal signal %d; backtrace (%d frames) ===\n", sig, numFrames);
	if (len > 0)
		(void)!write(STDERR_FILENO, header, (size_t)len);
	backtrace_symbols_fd(frames, numFrames, STDERR_FILENO);
	static const char footer[] = "=== end OpenMVS backtrace ===\n";
	(void)!write(STDERR_FILENO, footer, sizeof(footer) - 1);
	// SA_RESETHAND restored the default disposition on entry, so re-raising now
	// produces the original signal/core-dump behaviour (parent still sees rc=139).
	raise(sig);
}

void InstallCrashSignalHandlers() {
	struct sigaction sa;
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = SeacaveCrashSignalHandler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESETHAND | SA_NODEFER;
	sigaction(SIGSEGV, &sa, nullptr);
	sigaction(SIGABRT, &sa, nullptr);
	sigaction(SIGBUS, &sa, nullptr);
	sigaction(SIGFPE, &sa, nullptr);
	sigaction(SIGILL, &sa, nullptr);
}
} // namespace
#endif // __linux__

namespace SEACAVE {
// Tagged GENERAL_API to match the `extern GENERAL_API` declarations in
// Common.h so MSVC actually emits these as exported entries in Common.dll.
#if TD_VERBOSE == TD_VERBOSE_ON
GENERAL_API int g_nVerbosityLevel(2);
#endif
#if TD_VERBOSE == TD_VERBOSE_DEBUG
GENERAL_API int g_nVerbosityLevel(3);
#endif

GENERAL_API String g_strWorkingFolder;
GENERAL_API String g_strWorkingFolderFull;
} // namespace SEACAVE

#ifdef _USE_BOOST
#ifdef BOOST_NO_EXCEPTIONS
#if (BOOST_VERSION / 100000) > 1 || (BOOST_VERSION / 100 % 1000) > 72
#include <boost/assert/source_location.hpp>
#endif
namespace boost {
	void throw_exception(std::exception const & e) {
		VERBOSE("exception thrown: %s", e.what());
		ASSERT("boost exception thrown" == NULL);
		exit(EXIT_FAILURE);
	}
	#if (BOOST_VERSION / 100000) > 1 || (BOOST_VERSION / 100 % 1000) > 72
	void throw_exception(std::exception const & e, boost::source_location const & loc) {
		std::ostringstream ostr; ostr << loc;
		VERBOSE("exception thrown at %s: %s", ostr.str().c_str(), e.what());
		ASSERT("boost exception thrown" == NULL);
		exit(EXIT_FAILURE);
	}
	#endif
} // namespace boost
#endif
#endif

void SEACAVE::Initialize(LPCTSTR appname, unsigned nMaxThreads, int nProcessPriority) {
	// initialize thread options
	Process::setCurrentProcessPriority((Process::Priority)nProcessPriority);
	#ifdef _USE_OPENMP
	if (nMaxThreads != 0)
		omp_set_num_threads(nMaxThreads);
	#endif

	#ifdef _USE_BREAKPAD
	// initialize crash memory dumper
	MiniDumper::Create(appname, WORKING_FOLDER);
	#endif

	#if defined(__linux__)
	// Breakpad above is Windows-only; on Linux install an execinfo backtrace
	// handler so fatal signals leave a stack trace in the logs.
	InstallCrashSignalHandlers();
	#endif

	// initialize random number generator
	Util::Init();
}

void SEACAVE::Finalize() {
	#if TD_VERBOSE != TD_VERBOSE_OFF
	// print memory statistics
	Util::LogMemoryInfo();
	#endif
}
/*----------------------------------------------------------------*/
