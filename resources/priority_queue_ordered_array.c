#include<stdio.h>
#include<limits.h>
#define MAX 100


// denotes where the last item in priority queue is
// initialized to -1 since no item is in queue
int idx = -1;

// pqVal holds data for each index item
// pqPriority holds priority for each index item
int pqVal[MAX];
int pqPriority[MAX];



int isEmpty ()
{
    return idx == -1;
}

int
isFull ()
{
    return idx == MAX - 1;
}

// Insert the element in maintaining items in sorted order 
// of their priority
void enqueue (int data, int priority)
{
    if (!isFull ())
    {

        // first item being entered
        if (idx == -1)
        {
            idx++;		// increase the index
            pqVal[idx] = data;
            pqPriority[idx] = priority;
            return;
        }
        else
        {
            // Increase the index
            idx++;
            // in reverse order
            for (int i = idx - 1; i >= 0; i--)
            {
                // shift all items rightwards with higher priority
                // than the element we trying to insert
                if (pqPriority[i] >= priority)
                {
                    pqVal[i + 1] = pqVal[i];
                    pqPriority[i + 1] = pqPriority[i];
                }
                else
                {
                    // insert item just before where
                    // lower priority index was found
                    pqVal[i + 1] = data;
                    pqPriority[i + 1] = priority;
                    break;
                }

            }
        }

    }
}

// returns item with highest priority
// note highest priority in max priority queue is last item in array
int peek ()
{
    return idx;
}

// just reducing index would mean we have dequed
// the value would be sitll there but we can say that 
// no more than a garbage value
void dequeue ()
{
    idx--;
}


void display ()
{
    for (int i = 0; i <= idx; i++)
    {
        printf ("(%d, %d)\n", pqVal[i], pqPriority[i]);
    }
}

// Driver Code
int main ()
{
    // To enqueue items as per priority
    enqueue (25, 1);
    enqueue (10, 10);
    enqueue (15, 50);
    enqueue (20, 100);
    enqueue (30, 5);
    enqueue (40, 7);

    printf ("Before Dequeue : \n");
    display ();

    // // Dequeue the top element
    dequeue ();			// 20 dequeued
    dequeue ();			// 15 dequeued

    printf ("\nAfter Dequeue : \n");
    display ();

    return 0;
}