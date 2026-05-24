#pragma once
#include "Vector.cuh"
#include <cmath>
#include <cstdint>

struct Particle {
  Vector pos;
  Vector vel;
  uint16_t mass;

  Particle();
  Particle(float x, float y);
};
