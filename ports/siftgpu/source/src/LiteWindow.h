#ifndef LITE_WINDOW_H
#define LITE_WINDOW_H

#include <cstddef>

struct GLFWwindow;

class LiteWindow {
 public:
  LiteWindow() {}
  virtual ~LiteWindow() {}

  virtual int IsValid() const { return 0; }
  virtual void MakeCurrent() {}
  virtual void Create(int x = -1, int y = -1, const char* display = NULL) {}
};

class LiteWindowGLFW : public LiteWindow {
 public:
  LiteWindowGLFW();
  ~LiteWindowGLFW() override;

  int IsValid() const override;
  void MakeCurrent() override;
  void Create(int x = -1, int y = -1, const char* display = NULL) override;

 private:
  GLFWwindow* _window;
};

class LiteWindowEGL : public LiteWindow {
 public:
  LiteWindowEGL();
  ~LiteWindowEGL() override;

  int IsValid() const override;
  void MakeCurrent() override;
  void Create(int x = -1, int y = -1, const char* display = NULL) override;

 private:
  void* _display;
  void* _context;
  void* _surface;
};

#endif
