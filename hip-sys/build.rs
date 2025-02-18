use std::path::PathBuf;

fn _generate_hiprtc_bindings() {
    println!("cargo:rustc-link-search=/opt/rocm/lib");

    println!("cargo:include=/opt/rocm/include");
    println!("cargo:include=/opt/rocm/include/hip");

    let bindings = bindgen::Builder::default()
        .header("wrapper_hiprtc.h")
        .clang_arg("-I/opt/rocm/include/hip")
        .clang_arg("-I/opt/rocm/include")
        .clang_arg("-D__HIP_PLATFORM_AMD__")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from("./tmp");
    bindings
        .write_to_file(out_path.join("hiprtc_bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn _generate_hip_runtime_bindings() {
    println!("cargo:rustc-link-search=/opt/rocm/lib");

    println!("cargo:include=/opt/rocm/include");
    println!("cargo:include=/opt/rocm/include/hip");

    let bindings = bindgen::Builder::default()
        .header("wrapper_hip_runtime.h")
        .clang_arg("-I/opt/rocm/include/hip")
        .clang_arg("-I/opt/rocm/include")
        .clang_arg("-D__HIP_PLATFORM_AMD__")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from("./tmp");
    bindings
        .write_to_file(out_path.join("hip_runtime_bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=/opt/rocm/lib");

    // Tell cargo to tell rustc to link the system shared library.
    println!("cargo:rustc-link-lib=dylib=hiprtc");
    println!("cargo:rustc-link-lib=dylib=amdhip64");

    // generate bindings
    // _generate_hiprtc_bindings();
    // _generate_hip_runtime_bindings();
}
