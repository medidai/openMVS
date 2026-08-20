#include "LiteWindow.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>

// Macro enabling headless GLFW initialization and context creation
#ifndef GLFW_HEADLESS_EGL
#define GLFW_HEADLESS_EGL 0
#endif

namespace {
// Track initialization and outstanding windows so GLFW terminates cleanly.
static bool g_glfwReady = false;
static int g_windowCount = 0;

void ErrorCallback(int code, const char* description) {
	std::cerr << "GLFW error " << code << ": " << (description ? description : "") << '\n';
}
}  // namespace

LiteWindowGLFW::LiteWindowGLFW() : _window(nullptr) {}

LiteWindowGLFW::~LiteWindowGLFW() {
	if (_window) {
		if (glfwGetCurrentContext() == _window)
			glfwMakeContextCurrent(nullptr);
		glfwDestroyWindow(_window);
		_window = nullptr;
		if (--g_windowCount == 0 && g_glfwReady) {
			glfwTerminate();
			g_glfwReady = false;
		}
	}
}

int LiteWindowGLFW::IsValid() const { return _window != nullptr; }

void LiteWindowGLFW::MakeCurrent() {
	if (_window)
		glfwMakeContextCurrent(_window);
}

void LiteWindowGLFW::Create(int x, int y, const char* display) {
	if (_window) return;
	if (!g_glfwReady) {
		glfwSetErrorCallback(ErrorCallback);
		#ifdef __linux__
		// Use the display parameter if provided (for X11 systems)
		if (display && *display)
			setenv("DISPLAY", display, 1);  // override DISPLAY environment variable
		#if GLFW_HEADLESS_EGL
		// Use EGL for headless context if requested
		glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_NULL);
		#endif
		#endif
		g_glfwReady = glfwInit();
	}
	if (!g_glfwReady) return;

	// Reset hints to default first to clear any previous failed attempts
	glfwDefaultWindowHints();

	// Headless / EGL specific
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	#if GLFW_HEADLESS_EGL
	glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
	#endif

	// SiftGPU compatibility (OpenGL 2.1)
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

	#if GLFW_HEADLESS_EGL
	// Buffer requirements - often fixes "No suitable EGLConfig"
	glfwWindowHint(GLFW_RED_BITS, 8);
	glfwWindowHint(GLFW_GREEN_BITS, 8);
	glfwWindowHint(GLFW_BLUE_BITS, 8);
	glfwWindowHint(GLFW_ALPHA_BITS, 8);
	glfwWindowHint(GLFW_DEPTH_BITS, 24);
	glfwWindowHint(GLFW_STENCIL_BITS, 8);
	#endif

	#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_FALSE);
	#endif

	const int width = (x > 0 ? x : 1);
	const int height = (y > 0 ? y : 1);
	_window = glfwCreateWindow(width, height, "siftgpu", nullptr, nullptr);
	if (_window) {
		++g_windowCount;
		glfwMakeContextCurrent(_window);
		glfwSwapInterval(0);  // Disable vsync for compute-style workloads.
	}
}
