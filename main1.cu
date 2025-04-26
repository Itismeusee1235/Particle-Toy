#include "GUI.h"
#include "Vector.cuh"
#include "particle.h"
#include <cuda_runtime.h>

const int W = 1000;
const int H = 1000;
const int N = 400000;
const int CN = 4;
const double gravFac = 30;
const double e = 2;
const double maxVel = 500;

__global__ void updateVelocities(Particle *pars, Particle *centers, int len,
                                 int cen_len, double deltaTime) {

  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= len || i <= 0) {
    return;
  }

  Vector net_force(0, 0);
  for (int j = 0; j < cen_len; j++) {

    double dist = magnitude(pars[i].pos - centers[j].pos);
    if (dist <= 1e-6) {
      double angle = (i * 73 + j * 91) % 360; // Simple hash
      double rad = angle * (M_PI / 180.0);    // degrees to radians
      Vector dir(cos(rad), sin(rad));
      net_force -= dir * 70.0;
      continue;
    }

    double force =
        gravFac * dist * centers[j].mass / powf(dist * dist + e, 1.5f);

    Vector dir = normalize(pars[i].pos - centers[j].pos);
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
  if (i >= len || i <= 0) {
    return;
  }
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
  GUI gui(W, H);

  if (!gui.init("Particles")) {
    return -1;
  }

  Particle *particles = new Particle[N];
  Particle centers[CN];

  Vector pos;
  for (int i = 0; i < N; i++) {
    pos.x = random() % W;
    pos.y = random() % H;
    particles[i].pos = pos;
  }

  for (int i = 0; i < CN; i++) {
    centers[i].pos.x = (pow(-1, i % 2 + 1) * 250) + 500;
    centers[i].pos.y = (pow(-1, i / 2 + 1) * 250) + 500;
    centers[i].mass = 75;
  }
  Particle *d_pars;
  Particle *c_pars;

  cudaMalloc(&d_pars, N * sizeof(Particle));
  cudaMemcpy(d_pars, particles, N * sizeof(Particle), cudaMemcpyHostToDevice);

  cudaMalloc(&c_pars, CN * sizeof(Particle));
  cudaMemcpy(c_pars, centers, CN * sizeof(Particle), cudaMemcpyHostToDevice);

  dim3 threadDim(16);
  dim3 gridDim((N + threadDim.x - 1) / threadDim.x);

  bool quit = false;
  bool sim = false;

  double last_time = SDL_GetTicks();

  Particle out[CN + N];

  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  std::cout << "Free memory: " << free_mem << " bytes\n";
  std::cout << "Total memory: " << total_mem << " bytes\n";

  while (!quit) {
    gui.clear();

    double currentTime = SDL_GetTicks();
    gui.eventHandler(quit, sim);

    double deltaTime = (currentTime - last_time) * 0.001f;

    std::cout << 1.0f / deltaTime << std::endl;

    if (sim) {
      updateVelocities<<<gridDim, threadDim>>>(d_pars, c_pars, N, CN,
                                               deltaTime);

      updatePositions<<<gridDim, threadDim>>>(d_pars, N, deltaTime);

      cudaMemcpy(out, d_pars, N * sizeof(Particle), cudaMemcpyDeviceToHost);
    }
    for (int i = N; i < CN + N; i++) {
      out[i] = centers[i - N];
    }
    gui.render(out, N + CN);

    last_time = currentTime;
    cudaMemGetInfo(&free_mem, &total_mem);
  }

  cudaFree(d_pars);
  return 0;
}
