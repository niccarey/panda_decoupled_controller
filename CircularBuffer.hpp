#include <cmath>
using namespace std;

class CircularBuffer
{
    int rear, front;
    int size;
    double *arr;

public:
    CircularBuffer(int s)
    {
      // initialise empty
      front = rear = -1;
      size = s;
      arr = new double[s];
    }

    bool isfull();

    void pushVal(double value);
    double popVal();

    void displayBuffer();

    double sumBuffer();
    double meanBuffer();
};


bool CircularBuffer::isfull(){
    if ((front == 0 && rear == size-1) || (rear == (front-1)%size-1))
        return true;

    else
        return false;
  }


void CircularBuffer::pushVal(double value){
  if ((front == 0 && rear == size-1) || (rear == (front-1)%size-1))
  {// remove the rear value and discard
    double throwaway = this->popVal();
  }
  else if (front == -1) // first element
  {
    front = rear = 0;
    arr[rear] = value;
  }
  else if (rear == size-1 && front !=0){
    rear = 0;
    arr[rear] = value;
  }
  else{
    rear++;
    arr[rear] = value;
  }
}


double CircularBuffer::popVal()
{
  if (front == -1)
  {
    // queue is empty
    return 0.0;
  }

  double data = arr[front];
  arr[front] = -1;
  if (front == rear)
  {
    front = -1;
    rear = -1;
  }
  else if (front == size-1)
      front = 0;
  else
      front++;

  return data;
}

void CircularBuffer::displayBuffer()
{
    if (front == -1){
      return;
    }
    if (rear >= front)
    {
      for (int i = front; i <= rear; i++)
          printf("%f ", arr[i]);
    }

    else {
      for (int i = front; i < size; i++)
          printf("%f ", arr[i]);

      for (int i = 0; i< rear; i++)
          printf("%f ", arr[i]);
    }
}

double CircularBuffer::sumBuffer()
{
  double sum = 0;
  if (front == -1){
    return 0.0;
  }
  if (rear >= front)
  {
    for (int i = front; i <= rear; i++)
        sum += arr[i];
  }

  else {
    for (int i = front; i < size; i++)
        sum += arr[i];

    for (int i = 0; i< rear; i++)
        sum += arr[i];
  }

  return sum;
}


double CircularBuffer::meanBuffer()
{
  double sum = this->sumBuffer();
  return (sum/size);
}
