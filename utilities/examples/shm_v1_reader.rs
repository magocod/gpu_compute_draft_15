use libc::{shmat, shmdt};
use std::ffi::c_void;
use utilities::tmp::SharedValues;

fn main() {
    let shm_id = 65596;

    unsafe {
        let shme = shmat(shm_id, std::ptr::null_mut(), 0) as *mut SharedValues;

        println!("The contents of the shared memory is:");
        let payload = &mut *(shme);
        println!("payload: {:?}", payload);
        println!("update");
        payload.index += 1;
        payload.arr[5] = 50;
        println!("payload: {:?}", payload);

        /* Detach the shared memory segment.  */
        shmdt(shme as *mut c_void);
    }
}
