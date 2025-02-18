use crate::cache::config::{CacheSrc, CacheType};
use crate::cache::kernel::common_replace;

const GLOBALS: &str = r#"
    // ...
    __global int mini_lru_last_priority__CACHE_ID = 1;
    __global int mini_lru_top__CACHE_ID = 0;

    __global int mini_lru_keys__CACHE_ID[CACHE_CAPACITY];
    __global int mini_lru_values__CACHE_ID[CACHE_CAPACITY];
    __global int mini_lru_priorities__CACHE_ID[CACHE_CAPACITY];
    
    __global int mini_lru_array_set__CACHE_ID[CACHE_CAPACITY];
    __global int mini_lru_array_set_entries__CACHE_ID[CACHE_CAPACITY];
    "#;

const BASE_FUNCTIONS: &str = r#"

    void mini_lru_selection_sort__CACHE_ID() {
    
        for (int i = 0; i < CACHE_CAPACITY - 1; i++) {
            int min_idx = i;
    
            // Find the minimum element in the remaining unsorted array
            for (int j = i + 1; j < CACHE_CAPACITY; j++) {
                if (mini_lru_priorities__CACHE_ID[j] > mini_lru_priorities__CACHE_ID[min_idx]) {
                    min_idx = j;
                }
            }
    
            // Swap the found minimum element with the first element
            
            int temp_key = mini_lru_keys__CACHE_ID[min_idx];
            
            mini_lru_keys__CACHE_ID[min_idx] = mini_lru_keys__CACHE_ID[i];
            mini_lru_keys__CACHE_ID[i] = temp_key;
            
            int temp_value = mini_lru_values__CACHE_ID[min_idx];
            
            mini_lru_values__CACHE_ID[min_idx] = mini_lru_values__CACHE_ID[i];
            mini_lru_values__CACHE_ID[i] = temp_value;
            
            int temp_priority = mini_lru_priorities__CACHE_ID[min_idx];
            
            mini_lru_priorities__CACHE_ID[min_idx] = mini_lru_priorities__CACHE_ID[i];
            mini_lru_priorities__CACHE_ID[i] = temp_priority;
            
        }
        
    }

    int mini_lru_array_set_get_available_index__CACHE_ID() {

        for (int i = 0; i < CACHE_CAPACITY; i++) {

            if (mini_lru_array_set__CACHE_ID[i] == -1) {
                return i;
            }

        }

        return -2;
    }

    int mini_lru_array_set_get_index__CACHE_ID(int* k) {

        for (int i = 0; i < CACHE_CAPACITY; i++) {
            if ( mini_lru_array_set__CACHE_ID[i] == *k ) {
                return i;
            }
        }

        return -1;
    }

    int mini_lru_array_set_insert__CACHE_ID(int* k) {

        for (int i = 0; i < CACHE_CAPACITY; i++) {
            if ( mini_lru_array_set__CACHE_ID[i] == *k ) {
                return SET_VALUE_EXIST;
            }
        }

        for (int i = 0; i < CACHE_CAPACITY; i++) {

            if (mini_lru_array_set_entries__CACHE_ID[i] != 0) {
                continue;
            }

            int exist_index = mini_lru_array_set_get_index__CACHE_ID(k);

            if (exist_index >= 0) {
                return SET_VALUE_EXIST;
            }

            int r = atomic_fetch_add(&mini_lru_array_set_entries__CACHE_ID[i], 1);

            if ( r == 0 ) {

                mini_lru_array_set__CACHE_ID[i] = *k;

                return SET_VALUE_NO_EXIST;

            }

        }

        return SET_FULL;
    }

    int check_if_mini_lru_key_exists__CACHE_ID(int* k) {

        for (int i = 0; i < CACHE_CAPACITY; i++) {
            if ( mini_lru_keys__CACHE_ID[i] == *k ) {
                return i;
            }
        }

        return -1;
    }

    int mini_lru_insert__CACHE_ID(
        int* key,
        int* value
        ) {

        int entry_index = check_if_mini_lru_key_exists__CACHE_ID(key);

        // exist in cache
        if (entry_index >= 0) {

            int current_priority = atomic_fetch_add(&mini_lru_last_priority__CACHE_ID, 1);

            mini_lru_values__CACHE_ID[entry_index] = *value;
            mini_lru_priorities__CACHE_ID[entry_index] = current_priority;

        } else {

            // cache is not full
            if (mini_lru_top__CACHE_ID <= CACHE_MAX_CAPACITY) {

                entry_index = atomic_fetch_add(&mini_lru_top__CACHE_ID, 1);

                if (entry_index <= CACHE_MAX_CAPACITY) {
                   int current_priority = atomic_fetch_add(&mini_lru_last_priority__CACHE_ID, 1);

                   mini_lru_keys__CACHE_ID[entry_index] = *key;
                   mini_lru_values__CACHE_ID[entry_index] = *value;
                   mini_lru_priorities__CACHE_ID[entry_index] = current_priority;

                   return entry_index;
                }

            }

            // cache is full

            // limit
            int counter = 0;

            while(1) {

                entry_index = mini_lru_array_set_get_available_index__CACHE_ID();
                
                if (entry_index < 0) {
                    // entry_index = -200;
                    break;
                }
                
                int min_priority = mini_lru_priorities__CACHE_ID[entry_index];
                        
                if (counter > CACHE_MAX_CAPACITY) {
                    // entry_index = -200;
                    break;
                }
                
                counter++;
        
                for (int i = 0; i < CACHE_CAPACITY; i++) {
                
                    if (mini_lru_array_set_get_index__CACHE_ID(&i) >= 0) {
                        continue;
                    }
                
                    int priority = mini_lru_priorities__CACHE_ID[i];
        
                    if (priority < min_priority) {
                    
                        min_priority = priority;
                        entry_index = i;
        
                    }
                    
                }
                           
                int r = mini_lru_array_set_insert__CACHE_ID(&entry_index);
                
                // if (r == SET_VALUE_EXIST) {
                //    ...
                // }
               
                if (r == SET_VALUE_NO_EXIST) {
                   int current_priority = atomic_fetch_add(&mini_lru_last_priority__CACHE_ID, 1);
    
                   mini_lru_keys__CACHE_ID[entry_index] = *key;
                   mini_lru_values__CACHE_ID[entry_index] = *value;
                   mini_lru_priorities__CACHE_ID[entry_index] = current_priority;
                   
                   break;
                }
                
                if (r == SET_FULL) {
                  entry_index = -500;
                  break;
                }
                                
            }            

        }

        return entry_index;
    }
    
    int mini_lru_get__CACHE_ID(
        int* key,
        int* value
        ) {
        
        int entry_index = check_if_mini_lru_key_exists__CACHE_ID(key);
    
        if (entry_index >= 0) {
    
           int current_priority = atomic_fetch_add(&mini_lru_last_priority__CACHE_ID, 1);
    
           *value = mini_lru_values__CACHE_ID[entry_index];
           mini_lru_priorities__CACHE_ID[entry_index] = current_priority;
    
        }

        return entry_index;
    }
    "#;

const BASE_KERNELS: &str = r#"
    kernel void mini_lru_reset__CACHE_ID() {
        int i = get_global_id(0);

        mini_lru_keys__CACHE_ID[i] = -1;
        mini_lru_values__CACHE_ID[i] = -1;
        mini_lru_priorities__CACHE_ID[i] = 0;
        
        mini_lru_array_set__CACHE_ID[i] = -1;
        mini_lru_array_set_entries__CACHE_ID[i] = 0;

        if (i == 0) {
            mini_lru_last_priority__CACHE_ID = 1;
            mini_lru_top__CACHE_ID = 0;
        }

    }
    
    kernel void mini_lru_array_set_reset__CACHE_ID() {
        int i = get_global_id(0);

        mini_lru_array_set__CACHE_ID[i] = -1;
        mini_lru_array_set_entries__CACHE_ID[i] = 0;
    }

    kernel void mini_lru_debug__CACHE_ID(
        global int* keys_output,
        global int* values_output,
        global int* priorities_output,
        global int* meta_output,
        global int* set_items_output
        ) {
        int i = get_global_id(0);

        keys_output[i] = mini_lru_keys__CACHE_ID[i];
        values_output[i] = mini_lru_values__CACHE_ID[i];
        priorities_output[i] = mini_lru_priorities__CACHE_ID[i];
        
        set_items_output[i] = mini_lru_array_set__CACHE_ID[i];
        set_items_output[i + CACHE_CAPACITY] = mini_lru_array_set_entries__CACHE_ID[i];

        if (i == 0) {
            meta_output[0] = mini_lru_last_priority__CACHE_ID;
            meta_output[1] = mini_lru_top__CACHE_ID;
        }
    }

    kernel void mini_lru_get_keys__CACHE_ID(
        global int* keys_output,
        global int* priorities_output
        ) {

        int i = get_global_id(0);

        keys_output[i] = mini_lru_keys__CACHE_ID[i];
        priorities_output[i] = mini_lru_priorities__CACHE_ID[i];
    }
    
    kernel void mini_lru_get_sorted_keys__CACHE_ID(
        queue_t q0,
        const uint capacity_device_local_work_size,
        global int* keys_output,
        global int* priorities_output,
        global int* enqueue_kernel_output
        ) {
        
        clk_event_t evt0;

        enqueue_kernel_output[0] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            0,
            NULL,
            &evt0,
            ^{
               mini_lru_selection_sort__CACHE_ID();
            }
        );
    
        enqueue_kernel_output[1] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(CACHE_CAPACITY, capacity_device_local_work_size),
            1,
            &evt0,
            NULL,
            ^{
                int i = get_global_id(0);
            
                keys_output[i] = mini_lru_keys__CACHE_ID[i];
                priorities_output[i] = mini_lru_priorities__CACHE_ID[i];
            }
        );
    
        release_event(evt0);
        
        
    }
    
    kernel void mini_lru_sort__CACHE_ID() {
        mini_lru_selection_sort__CACHE_ID();
    }
    
    kernel void mini_lru_put__CACHE_ID(
        global int* keys_input,
        global int* values_input,
        global int* priorities_input
        ) {
        int i = get_global_id(0);
        int global_size = get_global_size(0);

        mini_lru_keys__CACHE_ID[i] = keys_input[i];
        mini_lru_values__CACHE_ID[i] = values_input[i];
        mini_lru_priorities__CACHE_ID[i] = priorities_input[i];

        if (i == 0) {
            mini_lru_last_priority__CACHE_ID = global_size + 1;
            mini_lru_top__CACHE_ID = global_size;
        }
    }
    
    kernel void write_in_mini_lru__CACHE_ID(
        global int* keys_input,
        global int* values_input,
        global int* indices_output
        ) {

        int i = get_global_id(0);

        indices_output[i] = mini_lru_insert__CACHE_ID(
            &keys_input[i],
            &values_input[i]
        );
    }

    kernel void read_on_mini_lru__CACHE_ID(
        global int* keys_input,
        global int* values_output,
        global int* indices_output
        ) {
        int i = get_global_id(0);

        int pi = -1;

        indices_output[i] = mini_lru_get__CACHE_ID(
            &keys_input[i],
            &pi
        );
        values_output[i] = pi;
    }
    "#;

impl CacheSrc {
    pub fn generate_cache_mini_lru_program_source(&self) -> String {
        let blocks = self.get_configs_by_type(CacheType::MiniLRU);

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
    /// *** MINI LRU SRC *** ///

    /// constants
    const int PUT_MINI_LRU = 0;
    const int GET_MINI_LRU = 1;

    const int SET_VALUE_EXIST = 0;
    const int SET_VALUE_NO_EXIST = 1;
    const int SET_FULL = 2;

    /// globals
    {globals}

    /// kernels



    {base_functions}

    {base_kernels}

    /// *** MINI LRU SRC *** ///
    "
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a() {
        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(8);

        let program_source = cache_src.generate_cache_mini_lru_program_source();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut cache_src = CacheSrc::new();
        cache_src.add_mini_lru(8);
        cache_src.add_mini_lru(32);
        cache_src.add_mini_lru(16);

        let program_source = cache_src.generate_cache_mini_lru_program_source();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_is_empty() {
        let cache_src = CacheSrc::new();

        let program_source = cache_src.generate_cache_mini_lru_program_source();
        println!("{program_source}");
        assert!(program_source.is_empty());
    }
}
