#include "GUI.h"
#include "Vector.cuh"
#include "particle.h"
#include <cuda_runtime.h>
#include <time.h>

const int W = 1000;
const int H = 1000;
const int N = 3000;
const double gravFac = 30;
const double e = 2;
const double maxVel = 500;

__global__ void updateVelocities(Particle *pars, int len, double deltaTime) {

  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= len) {
    return;
  }

  // if(i == 0)
  // {
  //   return;
  // }

  Vector net_force(0, 0);
  for (int j = 0; j < len; j++) {
    if (i == j) {
      continue;
    }

    double dist = magnitude(pars[i].pos - pars[j].pos);
    if (dist <= 1e-6) {
      // double angle = (i * 73 + j * 91) % 360; // Simple hash
      // double rad = angle * (M_PI / 180.0);    // degrees to radians
      // Vector dir(cos(rad), sin(rad));
      // net_force -= dir * 70.0;
      continue;
    }

    double force = gravFac * dist * pars[j].mass / powf(dist * dist + e, 1.5f);

    Vector dir = normalize(pars[i].pos - pars[j].pos);
    net_force += dir * force;
  }
  if (magnitude(net_force) != 0) {
    pars[i].vel += net_force * -1;
  }

  if (magnitude(pars[i].vel) > maxVel) {
    pars[i].vel *= (maxVel / magnitude(pars[i].vel));
  }
}

__global__ void updatePositions(Particle *pars, int len, double deltaTime) {

  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= len) {
    return;
  }
  // if (i == 0) {
  //   return;
  // }
  pars[i].pos += pars[i].vel * deltaTime;

  Vector pos = pars[i].pos;

  if (pos.x >= W) {
    pos.x = W;
    pars[i].vel.x *= -1;
  } else if (pos.x < 0) {
    pos.x = 0;
    pars[i].vel.x *= -1;
  }
  if (pos.y >= W) {
    pos.y = W;
    pars[i].vel.y *= -1;
  } else if (pos.y < 0) {
    pos.y = 0;
    pars[i].vel.y *= -1;
  }
  pars[i].pos = pos;
  pars[i].vel *= 0.99999f;
}

int main() {
  srand(time(0));
  GUI gui(W, H);
  int c_x = 500;
  int c_y = 500;
  int radius = 100;

  if (!gui.init("Particles")) {
    return -1;
  }

  Particle particles[N];

  double angle = 2 * M_PI / N;
  std::cout << angle << std::endl;

  Vector pos;
  for (int i = 0; i < N; i++) {
    pos.x = random() % W;
    pos.y = random() % H;
    particles[i].pos = pos;
  }

  // particles[0].pos = Vector(500, 500);
  // particles[0].vel = Vector(0, 0);
  // particles[0].mass = 50;

  Particle *d_pars;
  cudaMalloc(&d_pars, N * sizeof(Particle));
  cudaMemcpy(d_pars, particles, N * sizeof(Particle), cudaMemcpyHostToDevice);
  dim3 threadDim(16);
  dim3 gridDim((N + threadDim.x - 1) / threadDim.x);

  bool quit = false;
  bool sim = false;

  double last_time = SDL_GetTicks();

  while (!quit) {
    gui.clear();
    double currentTime = SDL_GetTicks();
    gui.eventHandler(quit, sim);

    double deltaTime = (currentTime - last_time) * 0.001f;

    std::cout << 1.0f / deltaTime << std::endl;

    if (sim) {
      updateVelocities<<<gridDim, threadDim>>>(d_pars, N, deltaTime);
      cudaDeviceSynchronize();

      updatePositions<<<gridDim, threadDim>>>(d_pars, N, deltaTime);
      cudaDeviceSynchronize();

      cudaMemcpy(particles, d_pars, N * sizeof(Particle),
                 cudaMemcpyDeviceToHost);
    }
    gui.render(particles, N);

    last_time = currentTime;
  }

  cudaFree(d_pars);
  return 0;
}
