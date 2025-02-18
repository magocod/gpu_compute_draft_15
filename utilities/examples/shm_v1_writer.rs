use libc::{shmat, shmdt, shmget, IPC_CREAT, IPC_EXCL, S_IRGRP, S_IRUSR, S_IWGRP, S_IWUSR};
use std::ffi::c_void;
use utilities::tmp::SharedValues;

pub fn main() {
    let shared_segment_size = std::mem::size_of::<SharedValues>();

    let shm_id = unsafe {
        shmget(
            0,
            shared_segment_size,
            IPC_CREAT
                | IPC_EXCL
                | S_IRUSR as i32
                | S_IWUSR as i32
                | S_IRGRP as i32
                | S_IWGRP as i32,
        )
    };

    // shmat to attach to shared memory
    let shme = unsafe { shmat(shm_id, std::ptr::null(), 0) as *mut SharedValues };

    // mutate the value
    let shared_values = unsafe { &mut *(shme) };
    shared_values.index += 1;
    shared_values.arr[0] = 10;

    println!("Data written in memory: ");
    println!("shm_id: {}", shm_id);
    println!("{:?}", shared_values);

    // detach from shared memory
    unsafe {
        shmdt(shme as *mut c_void);
    }
}
