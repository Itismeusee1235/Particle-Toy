#include "particle.h"

Particle::Particle() {
  Vector pos(0, 0);
  Vector vel(0, 0);
  mass = 1;
}
Particle::Particle(float x, float y) {
  Vector pos(x, y);
  mass = 1;
}
