#pragma once

#include "particle.h"
#include <SDL2/SDL.h>
#include <iostream>

class GUI {
  int width, height;
  SDL_Window *window;
  SDL_Renderer *renderer;

public:
  GUI(int width, int height);
  bool init(char *name);
  void eventHandler(bool &quit, bool &sim);
  void render(Particle *par_arr, int len);
  void clear();
  ~GUI();
};
