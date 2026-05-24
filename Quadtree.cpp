#include "Vector.cuh"
#include "particle.h"
#include <SDL2/SDL.h>
#include <bits/stdc++.h>
#include <iostream>
const int max_count = 1;
const float ratio = 0.5f;

typedef struct QuadNode {
  QuadNode *children[4] = {nullptr, nullptr, nullptr, nullptr};
  int x, y;
  int w, h;
  Vector com;
  int count = 0;
  Particle *particle;

  QuadNode() {}
  QuadNode(int x, int y, int w, int h) {
    this->x = x;
    this->y = y;
    this->w = w;
    this->h = h;
  }
};

void Subdivide(QuadNode &root) {
  int x = root.x;
  int y = root.y;
  int w = root.w / 2;
  int h = root.h / 2;
  root.children[0] = new QuadNode(x + w, y, w, h);
  root.children[1] = new QuadNode(x, y, w, h);
  root.children[2] = new QuadNode(x, y + h, w, h);
  root.children[3] = new QuadNode(x + w, y + h, w, h);
}

void update(Particle &par) {
  par.pos += par.vel;
  if (par.pos.x < 0) {
    par.pos.x = 0;
    par.vel.x *= -1;
  } else if (par.pos.x > 1000) {
    par.pos.x = 1000;
    par.vel.x *= -1;
  }
  if (par.pos.y < 0) {
    par.pos.y = 0;
    par.vel.y *= -1;
  } else if (par.pos.y > 1000) {
    par.pos.y = 1000;
    par.vel.y *= -1;
  }
}

bool contains(QuadNode &node, Particle &particle) {
  return (node.x < particle.pos.x && particle.pos.x < node.x + node.w &&
          node.y < particle.pos.y && particle.pos.y < node.y + node.h);
}

bool insert(QuadNode &root, Particle &particle) {
  if (!contains(root, particle)) {
    return false;
  }

  if (root.count < max_count) {
    root.particle = &particle;
  } else {
    if (root.children[0] == nullptr) {
      Subdivide(root);
      for (int i = 0; i < 4; i++) {
        if (insert(*root.children[i], *root.particle)) {
          break;
        }
      }
      root.particle = nullptr;
    }
    for (int i = 0; i < 4; i++) {
      if (insert(*root.children[i], particle)) {
        break;
      }
    }
  }

  root.count++;
  return true;
}

void display(QuadNode &root, SDL_Renderer *renderer) {
  SDL_Rect rect;
  rect.x = root.x;
  rect.y = root.y;
  rect.w = root.w;
  rect.h = root.h;

  SDL_RenderDrawRect(renderer, &rect);
  if (root.children[0] != nullptr) {
    for (int i = 0; i < 4; i++) {
      display(*root.children[i], renderer);
    }
  }
}

int main() {
  int W = 1000;
  int H = 1000;
  int cN = 30;
  std::vector<Particle> particles;

  SDL_Init(SDL_INIT_EVERYTHING);
  SDL_Window *window =
      SDL_CreateWindow("Test", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                       W, H, SDL_WINDOW_SHOWN);
  SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, 0);

  for (int i = 0; i < cN; i++) {
    Particle ne;
    ne.pos.x = rand() % W;
    ne.pos.y = rand() & H;
    // ne.vel.x = rand() % 2;
    // ne.vel.y = rand() % 2;
    particles.push_back(ne);
  }

  bool quit = false;
  SDL_Rect pixel{0, 0, 1, 1};
  float last_time = 0;

  while (!quit) {
    float current_time = SDL_GetTicks() / 1000;
    float delta_time = current_time - last_time;
    last_time = current_time;

    QuadNode tree(0, 0, W, H);
    for (int i = 0; i < particles.size(); i++) {
      insert(tree, particles[i]);
    }

    SDL_Event ev;
    int x, y;
    SDL_GetMouseState(&x, &y);
    while (SDL_PollEvent(&ev)) {
      if (ev.type == SDL_QUIT) {
        quit = true;
        break;
      }
      if (ev.type == SDL_MOUSEBUTTONDOWN) {
        Particle par;
        par.pos.x = x;
        par.pos.y = y;
        particles.push_back(par);
        insert(tree, par);
      }
    }
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0xFF);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 0x0, 0xFF, 0x0, 0x50);
    display(tree, renderer);
    SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0xFF, 0xFF);

    for (int i = 0; i < particles.size(); i++) {
      pixel.x = particles[i].pos.x;
      pixel.y = particles[i].pos.y;
      SDL_RenderFillRect(renderer, &pixel);
      update(particles[i]);
    }
    SDL_RenderPresent(renderer);
    if (delta_time < 1.0f / 60.0f) {
      SDL_Delay(1000.f / 60.0f - delta_time * 1000);
    }
  }
  SDL_Quit();
  return 0;
}
