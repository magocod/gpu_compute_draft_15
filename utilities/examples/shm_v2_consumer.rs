use libc::{mmap, shm_open, shm_unlink, MAP_SHARED, O_RDONLY, PROT_READ};
use std::ffi::CString;
use utilities::tmp::SharedValues;

pub fn main() {
    /* the size (in bytes) of shared memory object */
    let size = std::mem::size_of::<SharedValues>();

    /* name of the shared memory object */
    // const char* name = "OS";
    // let name = CString::new("RS_1").unwrap();
    let name = CString::new("d").unwrap();

    /* open the shared memory object */
    let shm_fd = unsafe { shm_open(name.as_ptr(), O_RDONLY, 0666) };

    if shm_fd < 0 {
        panic!("open");
    }

    /* memory map the shared memory object */
    let shme = unsafe {
        mmap(std::ptr::null_mut(), size, PROT_READ, MAP_SHARED, shm_fd, 0) as *mut SharedValues
    };

    let shared_values = unsafe { &mut *(shme) };

    println!("Data read in memory: ");
    println!("shm_fd: {}", shm_fd);
    println!("shared_values: {:?}", shared_values);

    /* remove the shared memory object */
    unsafe {
        shm_unlink(name.as_ptr());
    }
}
