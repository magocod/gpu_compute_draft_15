use crate::cache::config::{CacheSrc, CacheType};
use crate::cache::kernel::common_replace;
use crate::config::ClTypeDefault;

const STRUCT_DEF: &str = r#"
    struct LruEntry {
        int priority;
        int to_index;
    };
    "#;

const GLOBALS: &str = r#"
    // ...
    __global int lru_last_priority__CACHE_ID = 1;
    __global int lru_top__CACHE_ID = 0;

    __global int lru_keys__CACHE_ID[CACHE_CAPACITY][KEY_LEN];
    __global int lru_values__CACHE_ID[CACHE_CAPACITY][VALUE_LEN];
    __global int lru_priorities__CACHE_ID[CACHE_CAPACITY];
    
    __global int lru_array_set__CACHE_ID[CACHE_CAPACITY];
    __global int lru_array_set_entries__CACHE_ID[CACHE_CAPACITY];
    
    // tmp (sort)
    __global int lru_tmp_keys__CACHE_ID[CACHE_CAPACITY][KEY_LEN];
    __global int lru_tmp_values__CACHE_ID[CACHE_CAPACITY][VALUE_LEN];
    
    __global struct LruEntry lru_sort_entries__CACHE_ID[CACHE_CAPACITY];
    
    "#;

const BASE_FUNCTIONS: &str = r#"
    void lru_init_selection_sort__CACHE_ID() {
    
        for (int i = 0; i < CACHE_CAPACITY - 1; i++) {
            int min_idx = i;
    
            // Find the minimum element in the remaining unsorted array
            for (int j = i + 1; j < CACHE_CAPACITY; j++) {
                if (lru_priorities__CACHE_ID[j] > lru_priorities__CACHE_ID[min_idx]) {
                    min_idx = j;
                }
            }
    
            // Swap the found minimum element with the first element
            
            int temp_priority = lru_priorities__CACHE_ID[min_idx];
            
            lru_priorities__CACHE_ID[min_idx] = lru_priorities__CACHE_ID[i];
            lru_priorities__CACHE_ID[i] = temp_priority;
            
        }
        
        for (int i = 0; i < CACHE_CAPACITY; i++) {

            for (int sub_index = 0; sub_index < CACHE_CAPACITY; sub_index++) {
            
                if (lru_sort_entries__CACHE_ID[i].priority == lru_priorities__CACHE_ID[sub_index]) {
                    lru_sort_entries__CACHE_ID[i].to_index = sub_index;
                    break;
                }
                
            }
    
        }
        
    }

    int lru_array_set_get_available_index__CACHE_ID() {
    
        for (int i = 0; i < CACHE_CAPACITY; i++) {
        
            if (lru_array_set__CACHE_ID[i] == -1) {
                return i;
            }
            
        }
        
        return -2;
    }
    
    int lru_array_set_get_index__CACHE_ID(int* k) {

        for (int i = 0; i < CACHE_CAPACITY; i++) {
            if ( lru_array_set__CACHE_ID[i] == *k ) {
                return i;
            }
        }

        return -1;
    }
    
    int lru_array_set_insert__CACHE_ID(int* k) {
    
        for (int i = 0; i < CACHE_CAPACITY; i++) {
            if ( lru_array_set__CACHE_ID[i] == *k ) {
                return SET_VALUE_EXIST_DEF_2;
            }
        }
                
        for (int i = 0; i < CACHE_CAPACITY; i++) {           

            if (lru_array_set_entries__CACHE_ID[i] != 0) {
                continue;
            }
            
            int exist_index = lru_array_set_get_index__CACHE_ID(k);

            if (exist_index >= 0) {
                return SET_VALUE_EXIST_DEF_2;
            }
        
            int r = atomic_fetch_add(&lru_array_set_entries__CACHE_ID[i], 1);
        
            if ( r == 0 ) {

                lru_array_set__CACHE_ID[i] = *k;

                return SET_VALUE_NO_EXIST_DEF_2;

            }
            
        }

        return SET_FULL_DEF_2;
    }

    int check_if_lru_key_exists__CACHE_ID(int key_input_index, int* key) {

        for (int i = 0; i < CACHE_CAPACITY; i++) {

            int exist_index = i;

            for (int key_index = 0; key_index < KEY_LEN; key_index++) {
                if (lru_keys__CACHE_ID[i][key_index] != key[key_index + key_input_index]) {
                    exist_index = -1;
                    break;
                }
            }

            if (exist_index >= 0) {
                return exist_index;
            }

        }

        return -1;
    }

    void lru_set_key__CACHE_ID(int entry_index, int key_input_index, int* key) {

        for (int index = 0; index < KEY_LEN; index++) {
            lru_keys__CACHE_ID[entry_index][index] = key[index + key_input_index];
        }

    }

    void lru_set_value__CACHE_ID(int entry_index, int value_input_index, int* value) {

        for (int index = 0; index < VALUE_LEN; index++) {
            lru_values__CACHE_ID[entry_index][index] = value[index + value_input_index];
        }

    }

    void lru_get_key__CACHE_ID(int entry_index, int output_index, int* key) {

        for (int index = 0; index < KEY_LEN; index++) {
            key[output_index + index] = lru_keys__CACHE_ID[entry_index][index];
        }

    }

    void lru_get_value__CACHE_ID(int entry_index, int output_index, int* value) {

        for (int index = 0; index < VALUE_LEN; index++) {
            value[output_index + index] = lru_values__CACHE_ID[entry_index][index];
        }

    }

    void lru_set_default_output_value__CACHE_ID(int output_index, int* value) {

        for (int index = 0; index < VALUE_LEN; index++) {
            value[output_index + index] = CL_DEFAULT_VALUE;
        }

    }
    
    int lru_insert__CACHE_ID(
        int key_input_index,
        int value_input_index,
        int* key,
        int* value
        ) {
        
        int entry_index = check_if_lru_key_exists__CACHE_ID(key_input_index, key);

        // exist in cache
        if (entry_index >= 0) {
        
            int current_priority = atomic_fetch_add(&lru_last_priority__CACHE_ID, 1);

            lru_set_value__CACHE_ID(entry_index, value_input_index, value);
            lru_priorities__CACHE_ID[entry_index] = current_priority;

        } else {
            
            // cache is not full
            if (lru_top__CACHE_ID <= CACHE_MAX_CAPACITY) {
            
                entry_index = atomic_fetch_add(&lru_top__CACHE_ID, 1);
        
                if (entry_index <= CACHE_MAX_CAPACITY) {
                   int current_priority = atomic_fetch_add(&lru_last_priority__CACHE_ID, 1);
    
                   lru_set_key__CACHE_ID(entry_index, key_input_index, key);
                   lru_set_value__CACHE_ID(entry_index, value_input_index, value);
                   lru_priorities__CACHE_ID[entry_index] = current_priority;
                   
                   return entry_index;
                }
                
            }
            
            // cache is full

            // limit
            int counter = 0;
            
            while(1) {
            
                entry_index = lru_array_set_get_available_index__CACHE_ID();
                
                if (entry_index < 0) {
                    // entry_index = -200;
                    break;
                }
                
                int min_priority = lru_priorities__CACHE_ID[entry_index];
                        
                if (counter > CACHE_MAX_CAPACITY) {
                    // entry_index = -200;
                    break;
                }
                
                counter++;
        
                for (int i = 0; i < CACHE_CAPACITY; i++) {
                
                    if (lru_array_set_get_index__CACHE_ID(&i) >= 0) {
                        continue;
                    }
                
                    int priority = lru_priorities__CACHE_ID[i];
        
                    if (priority < min_priority) {
                    
                        min_priority = priority;
                        entry_index = i;
        
                    }
                    
                }
                           
                int r = lru_array_set_insert__CACHE_ID(&entry_index);
                
                // if (r == SET_VALUE_EXIST_DEF_2) {
                //    ...
                // }
               
                if (r == SET_VALUE_NO_EXIST_DEF_2) {
                   int current_priority = atomic_fetch_add(&lru_last_priority__CACHE_ID, 1);
    
                   lru_set_key__CACHE_ID(entry_index, key_input_index, key);
                   lru_set_value__CACHE_ID(entry_index, value_input_index, value);
                   lru_priorities__CACHE_ID[entry_index] = current_priority;
                   
                   break;
                }
                
                if (r == SET_FULL_DEF_2) {
                  entry_index = -500;
                  break;
                }
                                
            }            

        }

        return entry_index;
    }
    
    int lru_get__CACHE_ID(
        int key_input_index,
        int value_output_index,
        int* key,
        int* value
        ) {
        int entry_index = check_if_lru_key_exists__CACHE_ID(key_input_index, key);

        if (entry_index >= 0) {

            int current_priority = atomic_fetch_add(&lru_last_priority__CACHE_ID, 1);

            lru_get_value__CACHE_ID(entry_index, value_output_index, value);
            lru_priorities__CACHE_ID[entry_index] = current_priority;

        } else {
            lru_set_default_output_value__CACHE_ID(value_output_index, value);
        }

        return entry_index;
    }

    "#;

const BASE_KERNELS: &str = r#"
    kernel void lru_reset__CACHE_ID() {
        int i = get_global_id(0);
        
        for (int key_index = 0; key_index < KEY_LEN; key_index++) {
            lru_keys__CACHE_ID[i][key_index] = -1;
        }
        for (int value_index = 0; value_index < VALUE_LEN; value_index++) {
            lru_values__CACHE_ID[i][value_index] = -1;
        }
        
        lru_priorities__CACHE_ID[i] = 0;
        
        lru_array_set__CACHE_ID[i] = -1;
        lru_array_set_entries__CACHE_ID[i] = 0;

        if (i == 0) {
            lru_last_priority__CACHE_ID = 1;
            lru_top__CACHE_ID = 0;
        }

    }
    
    kernel void lru_array_set_reset__CACHE_ID() {
        int i = get_global_id(0);

        lru_array_set__CACHE_ID[i] = -1;
        lru_array_set_entries__CACHE_ID[i] = 0;
    }

    kernel void lru_debug__CACHE_ID(
        global int* keys_output,
        global int* values_output,
        global int* priorities_output,
        global int* meta_output,
        global int* set_items_output
        ) {

        int i = get_global_id(0);
        int key_output_index = i * KEY_LEN;
        int value_output_index = i * VALUE_LEN;

        for (int index = 0; index < KEY_LEN; index++) {
            keys_output[index + key_output_index] = lru_keys__CACHE_ID[i][index];
        }
        
        for (int index = 0; index < VALUE_LEN; index++) {
            values_output[index + value_output_index] = lru_values__CACHE_ID[i][index];
        }
        
        priorities_output[i] = lru_priorities__CACHE_ID[i];
        
        set_items_output[i] = lru_array_set__CACHE_ID[i];
        set_items_output[i + CACHE_CAPACITY] = lru_array_set_entries__CACHE_ID[i];

        if (i == 0) {
            meta_output[0] = lru_last_priority__CACHE_ID;
            meta_output[1] = lru_top__CACHE_ID;
        }
    }
    
    kernel void lru_get_keys__CACHE_ID(
        global int* keys_output,
        global int* priorities_output
        ) {

        int i = get_global_id(0);
        int key_output_index = i * KEY_LEN;

        for (int index = 0; index < KEY_LEN; index++) {
            keys_output[index + key_output_index] = lru_keys__CACHE_ID[i][index];
        }
        
        priorities_output[i] = lru_priorities__CACHE_ID[i];
    }
    
    kernel void lru_sort__CACHE_ID(
        queue_t q0,
        const int capacity_device_local_work_size,
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
               
                clk_event_t evt1;
                
                // copy values

                enqueue_kernel_output[1] = enqueue_kernel(
                    q0,
                    CLK_ENQUEUE_FLAGS_NO_WAIT,
                    ndrange_1D(CACHE_CAPACITY, capacity_device_local_work_size),
                    0,
                    NULL,
                    &evt1,
                    ^{
                        
                       int i = get_global_id(0);
                    
                       // priorities
                       lru_sort_entries__CACHE_ID[i].priority = lru_priorities__CACHE_ID[i];
                       lru_sort_entries__CACHE_ID[i].to_index = i;
                       
                       // keys
                       for (int index = 0; index < KEY_LEN; index++) {
                           lru_tmp_keys__CACHE_ID[i][index] = lru_keys__CACHE_ID[i][index];
                       }
                       
                       // values
                       for (int index = 0; index < VALUE_LEN; index++) {
                           lru_tmp_values__CACHE_ID[i][index] = lru_values__CACHE_ID[i][index];
                       }
                       
                    }
                );
                
                // calculate sort
            
                enqueue_kernel_output[2] = enqueue_kernel(
                    q0,
                    CLK_ENQUEUE_FLAGS_NO_WAIT,
                    ndrange_1D(1, 1),
                    1,
                    &evt1,
                    NULL,
                    ^{
                       lru_init_selection_sort__CACHE_ID();
                    }
                );
            
                release_event(evt1);
               
            }
        );
    
        enqueue_kernel_output[3] = enqueue_kernel(
            q0,
            // CLK_ENQUEUE_FLAGS_NO_WAIT,
            CLK_ENQUEUE_FLAGS_WAIT_KERNEL,
            ndrange_1D(CACHE_CAPACITY, capacity_device_local_work_size),
            1,
            &evt0,
            NULL,
            ^{
               // confirm sort

               int i = get_global_id(0);
               int to_entry_index = lru_sort_entries__CACHE_ID[i].to_index;

               if (lru_sort_entries__CACHE_ID[i].priority > 0) {

                   // keys
                   for (int index = 0; index < KEY_LEN; index++) {
                       lru_keys__CACHE_ID[to_entry_index][index] = lru_tmp_keys__CACHE_ID[i][index];
                   }

                   // values
                   for (int index = 0; index < VALUE_LEN; index++) {
                       lru_values__CACHE_ID[to_entry_index][index] = lru_tmp_values__CACHE_ID[i][index];
                   }

               }
               
            }
        );
        
        release_event(evt0);
        
    }
    
    kernel void lru_debug_sort__CACHE_ID(
        global int* output
        ) {
        
        int i = get_global_id(0);
        int output_index = i * 2;
        
        output[output_index] = lru_sort_entries__CACHE_ID[i].priority;
        output[output_index + 1] = lru_sort_entries__CACHE_ID[i].to_index;
    }
    
    kernel void lru_get_sorted_keys__CACHE_ID(
        queue_t q0,
        const uint capacity_device_local_work_size,
        global int* keys_output,
        global int* priorities_output,
        global int* enqueue_kernel_output
        ) {
        
        clk_event_t evt0;
        
        enqueue_kernel_output[4] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            0,
            NULL,
            &evt0,
            ^{
               // require indices
               // enqueue_kernel_output[0]
               // enqueue_kernel_output[1]
               // enqueue_kernel_output[2]
               // enqueue_kernel_output[3]
               
               lru_sort__CACHE_ID(
                    q0,
                    capacity_device_local_work_size,
                    enqueue_kernel_output
               );
            }
        );
    
        enqueue_kernel_output[5] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(CACHE_CAPACITY, capacity_device_local_work_size),
            1,
            &evt0,
            NULL,
            ^{
                int i = get_global_id(0);
                int key_output_index = i * KEY_LEN;
        
                for (int index = 0; index < KEY_LEN; index++) {
                    keys_output[index + key_output_index] = lru_keys__CACHE_ID[i][index];
                }
                
                priorities_output[i] = lru_priorities__CACHE_ID[i];
            }
        );
    
        release_event(evt0);
        
        
    }
    
    kernel void lru_put__CACHE_ID(
        global int* keys_input,
        global int* values_input,
        global int* priorities_input
        ) {

        int i = get_global_id(0);
        int global_size = get_global_size(0);
        int key_input_index = i * KEY_LEN;
        int value_input_index = i * VALUE_LEN;

        for (int index = 0; index < KEY_LEN; index++) {
            lru_keys__CACHE_ID[i][index] = keys_input[index + key_input_index];
        }
        
        for (int index = 0; index < VALUE_LEN; index++) {
            lru_values__CACHE_ID[i][index] = values_input[index + value_input_index];
        }
        
        lru_priorities__CACHE_ID[i] = priorities_input[i];

        if (i == 0) {
            lru_last_priority__CACHE_ID = global_size + 1;
            lru_top__CACHE_ID = global_size;
        }
    }
    
    kernel void write_in_lru__CACHE_ID(
        const uint key_len,
        const uint value_len,
        global int* keys_input,
        global int* values_input,
        global int* indices_output
        ) {

        int i = get_global_id(0);
        int key_input_index = i * key_len;
        int value_input_index = i * value_len;

        indices_output[i] = lru_insert__CACHE_ID(
            key_input_index,
            value_input_index,
            keys_input,
            values_input
        );
    }

    kernel void read_on_lru__CACHE_ID(
        const uint key_len,
        const uint value_len,
        global int* keys_input,
        global int* values_output,
        global int* indices_output
        ) {

        int i = get_global_id(0);
        int key_input_index = i * key_len;
        int value_output_index = i * value_len;

        indices_output[i] = lru_get__CACHE_ID(
            key_input_index,
            value_output_index,
            keys_input,
            values_output
        );
    }
    "#;

impl CacheSrc {
    pub fn generate_cache_lru_program_source(&self) -> String {
        let blocks = self.get_configs_by_type(CacheType::LRU);

        if blocks.is_empty() {
            return String::new();
        }

        let mut globals = String::new();

        let mut base_functions = String::new();

        let mut base_kernels = String::new();

        for config in blocks {
            let template = common_replace(GLOBALS, config);
            globals.push_str(&template);

            let template = common_replace(BASE_FUNCTIONS, config)
                .replace("CL_DEFAULT_VALUE", &i32::cl_default().to_string());
            base_functions.push_str(&template);

            let template = common_replace(BASE_KERNELS, config);
            base_kernels.push_str(&template);
        }

        format!(
            "
    /// *** LRU SRC *** ///

    /// constants
    const int PUT_LRU = 0;
    const int GET_LRU = 1;

    const int SET_VALUE_EXIST_DEF_2 = 0;
    const int SET_VALUE_NO_EXIST_DEF_2 = 1;
    const int SET_FULL_DEF_2 = 2;

    /// globals
    {STRUCT_DEF}

    {globals}

    /// kernels

    {base_functions}

    {base_kernels}

    /// *** LRU SRC *** ///
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
        cache_src.add_lru(8, 256, 256);

        let program_source = cache_src.generate_cache_lru_program_source();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(8, 256, 256);
        cache_src.add_lru(32, 512, 256);
        cache_src.add_lru(16, 256, 128);

        let program_source = cache_src.generate_cache_lru_program_source();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_is_empty() {
        let mut cache_src = CacheSrc::new();
        cache_src.add_lru(16, 256, 512);

        let program_source = cache_src.generate_cache_lru_program_source();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
