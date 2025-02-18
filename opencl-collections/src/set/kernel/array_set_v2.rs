use crate::set::config::{SetSrc, SetType};
use crate::set::kernel::common_replace;

const GLOBALS: &str = r#"
    __global int array_set__SET_ID[SET_CAPACITY];
    __global int array_set_entries__SET_ID[SET_CAPACITY];
    "#;

const BASE_FUNCTIONS: &str = r#"    
    bool array_set_is_full__SET_ID() {

        for (int i = 0; i < SET_CAPACITY; i++) {
            if ( array_set_entries__SET_ID[i] == 0 ) {
                return false;
            }
        }
        
        return true;

    }

    int array_set_get_index__SET_ID(int* k) {

        for (int i = 0; i < SET_CAPACITY; i++) {
            if ( array_set__SET_ID[i] == *k ) {
                return i;
            }
        }

        return -1;
    }

    int array_set_insert__SET_ID(int* k) {
    
        for (int i = 0; i < SET_CAPACITY; i++) {
            if ( array_set__SET_ID[i] == *k ) {
                return i;
            }
        }
                
        for (int i = 0; i < SET_CAPACITY; i++) {
            
            if (array_set_entries__SET_ID[i] != 0) {
                continue;
            }
            
            int exist_index = array_set_get_index__SET_ID(k);

            if (exist_index >= 0) {
                return exist_index;
            }
        
            int r = atomic_fetch_add(&array_set_entries__SET_ID[i], 1);
        
            if ( r == 0 ) {

                array_set__SET_ID[i] = *k;

                return i;

            }
        }

        return -1;
    }

    int array_set_remove__SET_ID(int* k) {

        for (int i = 0; i < SET_CAPACITY; i++) {
            if ( array_set__SET_ID[i] == *k ) {
            
                array_set__SET_ID[i] = -1;
                array_set_entries__SET_ID[i] = 0;
                
                return i;
            }
        }

        return -1;
    }
    "#;

const BASE_KERNELS: &str = r#"
    kernel void array_set_debug__SET_ID(
        global int* items_output
        ) {
        int i = get_global_id(0);

        items_output[i] = array_set__SET_ID[i];
        items_output[i + SET_CAPACITY] = array_set_entries__SET_ID[i];
        
    }
    
    kernel void array_set_reset__SET_ID() {
        int i = get_global_id(0);

        array_set__SET_ID[i] = -1;
        array_set_entries__SET_ID[i] = 0;
    }
    
    kernel void write_in_array_set__SET_ID(
        global int* items_input,
        global int* indices_output
        ) {
        int i = get_global_id(0);
    
        indices_output[i] = array_set_insert__SET_ID(&items_input[i]);
    }
    
    kernel void remove_in_array_set__SET_ID(
        global int* items_input,
        global int* indices_output
        ) {
        int i = get_global_id(0);

        indices_output[i] = array_set_remove__SET_ID(&items_input[i]);
    }
    "#;

impl SetSrc {
    pub fn generate_array_set_program_source_v2(&self) -> String {
        let blocks = self.get_configs_by_type(SetType::ArraySet);

        if blocks.is_empty() {
            return String::new();
        }

        let mut globals = String::new();

        let mut base_functions = String::new();
        let mut base_kernels = String::new();

        for config in blocks {
            let template = common_replace(GLOBALS, config);
            globals.push_str(&template);

            let template = common_replace(BASE_FUNCTIONS, config);
            base_functions.push_str(&template);

            let template = common_replace(BASE_KERNELS, config);
            base_kernels.push_str(&template);
        }

        format!(
            "

    /// *** ARRAY SET V2 SRC *** ///

    /// constants
    const int SET_VALUE_EXIST = 0;
    const int SET_VALUE_NO_EXIST = 1;
    const int SET_FULL = 2;

    /// globals
    {globals}

    /// kernels
    {base_functions}
    {base_kernels}

    /// *** ARRAY SET V2 SRC *** ///
    "
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a() {
        let mut set_src = SetSrc::new();
        set_src.add(8);

        let program_source = set_src.generate_array_set_program_source_v2();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut set_src = SetSrc::new();
        set_src.add(8);
        set_src.add(32);
        set_src.add(16);

        let program_source = set_src.generate_array_set_program_source_v2();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_is_empty() {
        let mut set_src = SetSrc::new();
        set_src.add(8);

        let program_source = set_src.generate_array_set_program_source_v2();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
