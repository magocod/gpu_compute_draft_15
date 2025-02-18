use std::path::PathBuf;

fn _generate_hsa_bindings() {
    println!("cargo:rustc-link-search=/opt/rocm/lib");

    println!("cargo:include=/opt/rocm/include");
    println!("cargo:include=/opt/rocm/include/hsa");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-I/opt/rocm/include/hsa")
        .clang_arg("-I/opt/rocm/include")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from("./tmp");
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=/opt/rocm/lib");

    // Tell cargo to tell rustc to link the system shared library.
    println!("cargo:rustc-link-lib=dylib=hsa-runtime64");

    // generate bindings
    // _generate_hsa_bindings();
}
