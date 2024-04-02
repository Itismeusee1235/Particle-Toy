#include "particle.h"
#include <SFML/Graphics.hpp>
#include <bits/stdc++.h>

using namespace sf;

void addCircle(Vector2i center, const int diameter,
               std::vector<Particle *> &particles, const int screenWidth,
               const int screenHeight, const int p_size) {
  for (int i = 0; i <= diameter; i++) {
    for (int j = 0; j <= diameter; j++) {
      double dist = sqrt((double)(i - diameter / 2) * (i - diameter / 2) +
                         (j - diameter / 2) * (j - diameter / 2));
      if (dist > diameter / 2 - 0.65 && dist < diameter / 2 + 0.2) {
        float x = i - diameter / 2 + center.x;
        float y = j - diameter / 2 + center.y;
        particles.push_back(new Particle(screenWidth, screenHeight, p_size,
                                         Vector2f(x, y), Vector2f(0, 0)));
      }
    }
  }
}

void addBlob(Vector2i center, const int diameter,
             std::vector<Particle *> &particles, const int screenWidth,
             const int screenHeight, const int p_size) {
  for (int i = 0; i < diameter; i++) {
    for (int j = 0; j < diameter; j++) {
      int x = center.x + i - diameter / 2;
      int y = center.y + j - diameter / 2;
      particles.push_back(new Particle(screenWidth, screenHeight, p_size,
                                       Vector2f(x, y), Vector2f(0, 0)));
    }
  }
}

int main() {
  srand(time(0));

  const int screenWidth = 800;
  const int screenHeight = 800;
  const int p_size = 1;
  int fac = 10;

  std::vector<Vector2i> centers;
  std::vector<Particle *> particles;

  RenderWindow win(VideoMode(screenWidth, screenHeight), "Particle");
  // win.setFramerateLimit(60);

  bool attract = true;
  bool shift = false;
  bool follow_mouse = false;

  centers.push_back(Vector2i(10, 10));
  centers.push_back(Vector2i(790, 10));
  centers.push_back(Vector2i(790, 790));
  centers.push_back(Vector2i(10, 790));
  Clock c;
  int count = 0;

  while (win.isOpen()) {
    float t = c.restart().asSeconds();
    float fps = 1 / t;
    if (fps <= 55) {
      if (count >= 5) {
        cout << particles.size();
        win.close();
      }
      count++;
    }
    cout << particles.size() << "\n";
    Event ev;
    while (win.pollEvent(ev)) {
      if (ev.type == Event::Closed) {
        win.close();
      } else if (ev.type == Event::KeyPressed) {
        if (ev.key.code == Keyboard::Space) {
          attract = !attract;
        } else if (ev.key.code == Keyboard::M) {
          follow_mouse = !follow_mouse;
        } else if (ev.key.code == Keyboard::P) {
          Vector2i pos = Mouse::getPosition(win);
          if (shift) {
            addBlob(pos, 20, particles, screenWidth, screenHeight, p_size);
          } else {
            particles.push_back(new Particle(screenWidth, screenHeight, p_size,
                                             Vector2f(pos.x, pos.y),
                                             Vector2f(0, 0)));
          }
        } else if (ev.key.code == Keyboard::F) {
          fac *= -1;
        } else if (ev.key.code == Keyboard::C) {
          particles.clear();
          centers.clear();
        } else if (ev.key.code == Keyboard::LShift) {
          shift = true;
        }
      } else if (ev.type == Event::KeyReleased) {
        if (ev.key.code == Keyboard::LShift) {
          shift = false;
        }
      } else if (ev.type == Event::MouseButtonPressed) {
        Vector2i pos = Mouse::getPosition(win);
        if (Mouse::isButtonPressed(Mouse::Left)) {
          addCircle(pos, 200, particles, screenWidth, screenHeight, p_size);
        } else {
          centers.push_back(pos);
        }
      }
    }

    particles.push_back(new Particle(screenHeight, screenWidth, p_size,
                                     Vector2f(400, 400),
                                     Vector2f(rand() % 9 - 4, rand() % 9 - 4)));

    Vector2i pos = Mouse::getPosition(win);
    for (auto i : particles) {
      if (attract) {
        if (follow_mouse) {
          i->Attract(pos, fac);
        } else {
          for (auto center : centers) {
            i->Attract(center, fac);
          }
        }
      }
      i->Friction(0.99);
      i->Move();
    }

    win.clear(Color::Black);
    for (auto i : particles) {
      i->Draw(win);
    }
    CircleShape cir(2);
    cir.setFillColor(Color::Red);
    for (auto i : centers) {
      cir.setPosition(Vector2f(i.x - 1, i.y - 1));
      win.draw(cir);
    }

    win.display();
  }

  return 0;
}