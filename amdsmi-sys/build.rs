fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search=native=/opt/rocm/lib");

    // Tell cargo to tell rustc to link the system shared library.
    println!("cargo:rustc-link-lib=dylib=amd_smi");
}
