use libc::shmctl;

fn main() {
    let shm_id = 65596;

    unsafe {
        /* Detach the shared memory segment.  */
        shmctl(shm_id, 0, std::ptr::null_mut());
    }
}
