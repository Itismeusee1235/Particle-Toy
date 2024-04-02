#ifndef PARTICLE_H
#define PARTICLE_H

#include <stdlib.h>
#include <SFML/Graphics.hpp>
#include <bits/stdc++.h>
using namespace sf;
using namespace std;

class Particle
{
  private:
  Vector2f pos;
  Vector2f vel;
  RectangleShape shape;
  int screenW;
  int screenH;

  private:
  inline Vector2f getNormal(Vector2i p);
  inline double getDist(Vector2i p);
  inline void setColor();

  public:
  Particle(int screenWidth, int screenHeight,int particle_size);
  Particle(int screenWidth, int screenHeight, int particle_size, Vector2f p, Vector2f v);
  inline void Move();
  inline void Friction(float fac);
  inline void Draw(RenderWindow& win);
  inline void Attract(Vector2i src, int fac);
};

Particle::Particle(int screenWidth, int screenHeight, int particle_size)
{
  screenH = screenHeight;
  screenW = screenWidth;
  pos.x = rand()%screenWidth;
  pos.y = rand()%screenHeight;
  vel.x = rand()%3 - 1;
  vel.y = rand()%3 - 1;

  shape = RectangleShape(Vector2f(particle_size, particle_size));
  shape.setFillColor(Color::White);
}

Particle::Particle(int screenWidth, int screenHeight, int particle_size, Vector2f p, Vector2f v)
{
  pos = p;
  vel = v;
  screenH = screenHeight;
  screenW = screenWidth;
  shape = RectangleShape(Vector2f(particle_size, particle_size));

  shape.setFillColor(Color::White);
  //setColor();
  
}

inline void Particle::setColor()
{

  int c = rand()%7;
  if(c == 0)
  {
    shape.setFillColor(Color::White);
  }
  else if (c == 1)
  {
    shape.setFillColor(Color::Blue);
  }
  else if (c == 2)
  {
    shape.setFillColor(Color::Cyan);
  }
  else if (c == 3)
  {
    shape.setFillColor(Color::Green);
  }
  else if (c == 4)
  {
    shape.setFillColor(Color::Magenta);
  }
  else if (c == 5)
  {
    shape.setFillColor(Color::Red);
  }
  else if (c == 6)
  {
    shape.setFillColor(Color::Yellow);
  }
  
}

inline double Particle::getDist(Vector2i p)
{
  const double dx = p.x - pos.x;
  const double dy = p.y - pos.y;
  return sqrt((dx*dx) + (dy*dy));
}

inline Vector2f Particle::getNormal(Vector2i p)
{
  double dist = getDist(p);
  if(dist == 0.0f)
  {
    dist = 1;
  }
  const float dx = (p.x - pos.x);
  const float dy = (p.y - pos.y);
  return Vector2f(dx*(1/dist), dy*(1/dist)); //unit vector from particle to p
}

inline void Particle::Attract(Vector2i src, int fac)
{
  double dist = getDist(src);
  dist = fmax(dist, 0.5);
  Vector2f normal = getNormal(src);
  //cout << dist << " " << normal.x << " " << normal.y << endl;
  vel.x += (fac*normal.x)/dist;
  vel.y += (fac*normal.y)/dist;
}

inline void Particle::Move()
{
  pos.x += vel.x;
  pos.y += vel.y;

  if(pos.x < 0)
  {
    pos.x += screenW;
  } 
  else if (pos.x>screenW)
  {
    pos.x-=screenW;
  }
  if(pos.y < 0)
  {
    pos.y += screenH;
  }
  else if (pos.y >= screenH)
  {
    pos.y -= screenH;
  }
  
  
}

inline void Particle::Draw(RenderWindow& win)
{  
  shape.setPosition(pos);
  win.draw(shape);
}

inline void Particle::Friction(float fac)
{
  vel.x *= fac;
  vel.y *= fac;
}

#endif