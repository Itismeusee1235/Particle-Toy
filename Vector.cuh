#ifndef VECTOR_CUH
#define VECTOR_CUH
#include <cmath>
#include <cuda_runtime.h>

struct Vector {
  float x;

  float y;

  __host__ __device__ Vector() {
    this->x = 0;
    this->y = 0;
  }
  __host__ __device__ Vector(double x, double y) {
    this->x = x;
    this->y = y;
  }

  __host__ __device__ Vector operator+(const Vector other) {
    Vector out;
    out.x = x + other.x;
    out.y = y + other.y;
    return out;
  }
  __host__ __device__ Vector operator-(const Vector other) {
    Vector out;
    out.x = x - other.x;
    out.y = y - other.y;
    return out;
  }
  __host__ __device__ Vector operator*(const double fac) {
    Vector out;
    out.x = x * fac;
    out.y = y * fac;
    return out;
  }
  __host__ __device__ Vector operator/(const double fac) {
    Vector out;
    out.x = x / fac;
    out.y = y / fac;
    return out;
  }
  __host__ __device__ void operator+=(const Vector other) {
    this->x += other.x;
    this->y += other.y;
  }
  __host__ __device__ void operator-=(const Vector other) {
    this->x -= other.x;
    this->y -= other.y;
  }
  __host__ __device__ void operator*=(const double fac) {
    this->x *= fac;
    this->y *= fac;
  }
};

inline double __host__ __device__ magnitude(Vector vec) {
  return sqrt(vec.x * vec.x + vec.y * vec.y);
}

inline Vector __host__ __device__ normalize(Vector vec) {
  Vector out;
  double mag = magnitude(vec);
  out.x = vec.x / mag;
  out.y = vec.y / mag;
  return out;
}

#endif
