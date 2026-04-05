#include "LiteWindow.h"

#if defined(SIFTGPU_EGL)

#include <EGL/egl.h>
#include <iostream>

LiteWindowEGL::LiteWindowEGL() : _display(EGL_NO_DISPLAY), _context(EGL_NO_CONTEXT), _surface(EGL_NO_SURFACE) {}

LiteWindowEGL::~LiteWindowEGL() {
	if (_display != EGL_NO_DISPLAY) {
		eglMakeCurrent(static_cast<EGLDisplay>(_display), EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
		if (_context != EGL_NO_CONTEXT) eglDestroyContext(static_cast<EGLDisplay>(_display), static_cast<EGLContext>(_context));
		if (_surface != EGL_NO_SURFACE) eglDestroySurface(static_cast<EGLDisplay>(_display), static_cast<EGLSurface>(_surface));
		eglTerminate(static_cast<EGLDisplay>(_display));
	}
	_display = EGL_NO_DISPLAY;
	_context = EGL_NO_CONTEXT;
	_surface = EGL_NO_SURFACE;
}

int LiteWindowEGL::IsValid() const {
	return _context != EGL_NO_CONTEXT && _surface != EGL_NO_SURFACE && _display != EGL_NO_DISPLAY;
}

void LiteWindowEGL::MakeCurrent() {
	if (IsValid()) {
		eglMakeCurrent(static_cast<EGLDisplay>(_display), static_cast<EGLSurface>(_surface), static_cast<EGLSurface>(_surface), static_cast<EGLContext>(_context));
	}
}

void LiteWindowEGL::Create(int x, int y, const char* /*display*/) {
	if (IsValid()) return;

	EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
	if (dpy == EGL_NO_DISPLAY) {
		std::cerr << "EGL: failed to get display" << std::endl;
		return;
	}
	if (eglInitialize(dpy, nullptr, nullptr) != EGL_TRUE) {
		std::cerr << "EGL: failed to initialize" << std::endl;
		return;
	}

	if (eglBindAPI(EGL_OPENGL_API) != EGL_TRUE) {
		std::cerr << "EGL: failed to bind OpenGL API" << std::endl;
		eglTerminate(dpy);
		return;
	}

	const EGLint cfgAttribs[] = {
		EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
		EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
		EGL_RED_SIZE, 8,
		EGL_GREEN_SIZE, 8,
		EGL_BLUE_SIZE, 8,
		EGL_ALPHA_SIZE, 8,
		EGL_DEPTH_SIZE, 24,
		EGL_NONE
	};

	EGLConfig config = nullptr;
	EGLint numConfigs = 0;
	if (eglChooseConfig(dpy, cfgAttribs, &config, 1, &numConfigs) != EGL_TRUE || numConfigs == 0) {
		std::cerr << "EGL: no matching config" << std::endl;
		eglTerminate(dpy);
		return;
	}

	const EGLint pbufferAttribs[] = {
		EGL_WIDTH, 1,
		EGL_HEIGHT, 1,
		EGL_NONE,
	};
	EGLSurface surf = eglCreatePbufferSurface(dpy, config, pbufferAttribs);
	if (surf == EGL_NO_SURFACE) {
		std::cerr << "EGL: failed to create pbuffer surface" << std::endl;
		eglTerminate(dpy);
		return;
	}

	const EGLint ctxAttribs[] = {
		EGL_CONTEXT_MAJOR_VERSION, 2,
		EGL_CONTEXT_MINOR_VERSION, 1,
		EGL_NONE
	};
	EGLContext ctx = eglCreateContext(dpy, config, EGL_NO_CONTEXT, ctxAttribs);
	if (ctx == EGL_NO_CONTEXT) {
		std::cerr << "EGL: failed to create context" << std::endl;
		eglDestroySurface(dpy, surf);
		eglTerminate(dpy);
		return;
	}

	if (eglMakeCurrent(dpy, surf, surf, ctx) != EGL_TRUE) {
		std::cerr << "EGL: failed to make context current" << std::endl;
		eglDestroyContext(dpy, ctx);
		eglDestroySurface(dpy, surf);
		eglTerminate(dpy);
		return;
	}

	_display = dpy;
	_surface = surf;
	_context = ctx;
}

#else
LiteWindowEGL::LiteWindowEGL() : _display(nullptr), _context(nullptr), _surface(nullptr) {}
LiteWindowEGL::~LiteWindowEGL() {}
int LiteWindowEGL::IsValid() const { return 0; }
void LiteWindowEGL::MakeCurrent() {}
void LiteWindowEGL::Create(int, int, const char*) {}
#endif
