#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <time.h>

int main ()

{
  int segment_id;
  char bogus;
  char* shared_memory;
  struct shmid_ds shmbuffer;
  int segment_size;
  const int shared_segment_size = 0x6400;

  time_t t;
  time(&t);

  printf("Enter the shared memory id: ");
  scanf("%d", &segment_id);

  /* Reattach the shared memory segment, at a different address.  */
  shared_memory = (char*) shmat (segment_id, (void*) 0x5000000, 0);
  printf ("shared memory reattached at address %p\n", shared_memory);

  /* Add to the segment */
  sprintf (shared_memory, "Modified the segment at %s", ctime(&t));

  /* Print out the string from shared memory.  */
  printf ("The contents of the shared memory is:\n%s", shared_memory);
  printf ("#################################################\n");


  return 0;

}
