pub const SET_KERNEL_SOURCE: &str = r#"
    const int SET_CAPACITY = CURRENT_CAPACITY;
    const int SET_MAX_CAPACITY = SET_CAPACITY - 1;
    
    __device__ int array_set[SET_CAPACITY];
    __device__ int array_set_entries[SET_CAPACITY];
    
    __device__ bool array_set_is_full() {

        for (int i = 0; i < SET_CAPACITY; i++) {
            if ( array_set_entries[i] == 0 ) {
                return false;
            }
        }
        
        return true;

    }

    __device__ int array_set_get_index(int* k) {

        for (int i = 0; i < SET_CAPACITY; i++) {
            if ( array_set[i] == *k ) {
                return i;
            }
        }

        return -1;
    }

    __device__ int array_set_insert(int* k) {
    
        for (int i = 0; i < SET_CAPACITY; i++) {
            if ( array_set[i] == *k ) {
                return i;
            }
        }
                
        for (int i = 0; i < SET_CAPACITY; i++) {
            
            if (array_set_entries[i] != 0) {
                continue;
            }
            
            int exist_index = array_set_get_index(k);

            if (exist_index >= 0) {
                return exist_index;
            }
                    
            int r = atomicAdd(&array_set_entries[i], 1);
        
            if ( r == 0 ) {

                array_set[i] = *k;

                return i;

            }
        }

        return -1;
    }

    __device__ int array_set_remove(int* k) {

        for (int i = 0; i < SET_CAPACITY; i++) {
            if ( array_set[i] == *k ) {
            
                array_set[i] = -1;
                array_set_entries[i] = 0;
                
                return i;
            }
        }

        return -1;
    }
    
    extern "C" __global__ void array_set_debug(
        int* items_output
        ) {
        int i = threadIdx.x;

        items_output[i] = array_set[i];
        items_output[i + SET_CAPACITY] = array_set_entries[i];
        
    }
    
    extern "C" __global__ void array_set_reset() {
        int i = threadIdx.x;

        array_set[i] = -1;
        array_set_entries[i] = 0;
    }

    extern "C" __global__ void write_in_array_set(
        int* items_input,
        int* indices_output
        ) {
        int i = threadIdx.x;
    
        indices_output[i] = array_set_insert(&items_input[i]);
    }
    
    extern "C" __global__ void remove_in_array_set(
        int* items_input,
        int* indices_output
        ) {
        int i = threadIdx.x;

        indices_output[i] = array_set_remove(&items_input[i]);
    }
    
    "#;

pub fn generate_array_set_program_source(capacity: usize) -> String {
    SET_KERNEL_SOURCE.replace("CURRENT_CAPACITY", &capacity.to_string())
}
