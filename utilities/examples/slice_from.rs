fn main() {
    let mut arr = [0; 10];

    println!("v1 {:?}", &arr);

    let p = arr.as_mut_ptr();

    let slice = std::ptr::slice_from_raw_parts_mut(p, 10);
    unsafe {
        (*slice)[0] = 5;
        (*slice)[7] = 3;
    }
    println!("slice {:?}", &slice);

    println!("v1 {:?}", &arr);
}
