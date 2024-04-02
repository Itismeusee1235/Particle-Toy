#include<bits/stdc++.h>
using namespace std;

int main()
{
  int i  = 0;
  srand(time(0));
  int num = rand();
  while (true)
  {
    i++;
    i = i%INT16_MAX;
    if(i%20000 == 0)
    {
      cout << num << endl;
      num = rand()%10;
    }
  }
  return 0;
  
}