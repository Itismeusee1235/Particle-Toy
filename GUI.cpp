#include "GUI.h"
#include "Vector.cuh"
#include "particle.h"

GUI::GUI(int width, int height) {
  GUI::width = width;
  GUI::height = height;
}

bool GUI::init(char *name) {
  if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
    printf("Failed to init SDL, %s ,", SDL_GetError());
    return false;
  }

  window = SDL_CreateWindow(name, 0, 0, width, height, SDL_WINDOW_SHOWN);

  if (window == NULL) {
    printf("Failed to make window, %s, ", SDL_GetError());
    return false;
  }

  renderer = SDL_CreateRenderer(window, -1, 0);
  if (renderer == NULL) {
    printf("Failed to make renderer, %s, ", SDL_GetError());
    return false;
  }

  return true;
}

void GUI::eventHandler(bool &quit, bool &sim) {
  SDL_Event ev;
  while (SDL_PollEvent(&ev)) {
    if (ev.type == SDL_QUIT) {
      quit = true;
      ;
      return;
    } else if (ev.type == SDL_KEYDOWN) {
      SDL_Keycode key = ev.key.keysym.sym;
      if (key == SDLK_q) {
        quit = true;
        return;
      }
      if (key == SDLK_SPACE) {
        sim = !sim;
      }
    }
  }
}

void GUI::render(Particle *par_arr, int len) {

  SDL_Rect par{0, 0, 1, 1};
  SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);
  for (int i = 0; i < len; i++) {
    par.x = (int)par_arr[i].pos.x;
    par.y = (int)par_arr[i].pos.y;
    double speed = magnitude(par_arr[i].vel);
    double norm_speed = speed / 250;

    SDL_Color color;
    if (norm_speed < 0.5) {
      double t = norm_speed / 0.5;
      color.r = (1.0 - t) * 255;
      color.g = t * 255;
      color.b = 0;
    } else {
      double t = (norm_speed - 0.5) / 0.5;
      color.r = 0;
      color.g = (1.0 - t) * 255;
      color.b = t * 255;
    }
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, 0xFF);
    // std::cout << par.x << " " << par.y << std::endl;
    SDL_RenderFillRect(renderer, &par);
  }

  SDL_RenderPresent(renderer);
}

void GUI::clear() {

  SDL_SetRenderDrawColor(renderer, 0x0, 0x0, 0x0, 0xFF);
  SDL_RenderClear(renderer);
}

GUI::~GUI() {
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
}
