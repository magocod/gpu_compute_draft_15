#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>

int main ()

{
  int segment_id;

  printf("Enter the shared memory id: ");
  scanf("%d", &segment_id);

  /* Deallocate the segment.  */
  shmctl (segment_id, IPC_RMID, 0);

  return 0;

}