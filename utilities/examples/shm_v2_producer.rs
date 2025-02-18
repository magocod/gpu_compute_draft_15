use libc::{ftruncate, mmap, mode_t, shm_open, size_t, MAP_SHARED, O_CREAT, O_RDWR, PROT_WRITE};
use std::ffi::{c_void, CString};
use utilities::tmp::SharedValues;

fn main() {
    /* the size (in bytes) of shared memory object */
    let size = std::mem::size_of::<SharedValues>();

    /* name of the shared memory object */
    // let name = CString::new("/sys/kernel/debug/mmap_example").unwrap();
    let name = CString::new("d").unwrap();

    /* create the shared memory object */
    let shm_fd = unsafe { shm_open(name.as_ptr(), O_RDWR | O_CREAT, 0666 as mode_t) };

    if shm_fd < 0 {
        panic!("open");
    }

    /* configure the size of the shared memory object */
    unsafe {
        ftruncate(shm_fd, size as i64);
    }

    let shme: *mut SharedValues;

    /* memory map the shared memory object */
    shme = unsafe {
        mmap(
            0 as *mut c_void,
            size as size_t,
            PROT_WRITE,
            MAP_SHARED,
            shm_fd,
            0,
        ) as *mut SharedValues
    };

    // mutate the value
    let shared_values = unsafe { &mut *(shme) };
    shared_values.index += 1;
    shared_values.arr[0] = 10;

    println!("Data written in memory: ");
    println!("shm_fd: {}", shm_fd);
    // println!("{:?}", shared_values);

    loop {
        // pass
    }
}
