use crate::config::ClTypeTrait;
use crate::dictionary::config::DictSrc;
use crate::dictionary::kernel::common_replace;

const GLOBALS: &str = r#"
    __global CL_TYPE dict_keys__DICT_ID[DICT_CAPACITY][DICT_KEY_LEN];
    __global CL_TYPE dict_values__DICT_ID[DICT_CAPACITY][DICT_VALUE_LEN];
    
    __global int dict_entries__DICT_ID[DICT_CAPACITY];
    "#;

const BASE_FUNCTIONS: &str = r#"
    int check_if_dict_key_exists__DICT_ID(int key_input_index, CL_TYPE* key) {

        for (int i = 0; i < DICT_CAPACITY; i++) {

            int exist_index = i;

            for (int key_index = 0; key_index < DICT_KEY_LEN; key_index++) {
                if (dict_keys__DICT_ID[i][key_index] != key[key_index + key_input_index]) {
                    exist_index = KEY_NOT_EXIST;
                    break;
                }
            }

            if (exist_index >= 0) {
                return exist_index;
            }

        }

        return KEY_NOT_EXIST;
    }

    void set_dict_key__DICT_ID(int entry_index, int key_input_index, CL_TYPE* key) {

        for (int index = 0; index < DICT_KEY_LEN; index++) {
            dict_keys__DICT_ID[entry_index][index] = key[index + key_input_index];
        }

    }

    void set_dict_value__DICT_ID(int entry_index, int value_input_index, CL_TYPE* value) {

        for (int index = 0; index < DICT_VALUE_LEN; index++) {
            dict_values__DICT_ID[entry_index][index] = value[index + value_input_index];
        }

    }

    void get_dict_key__DICT_ID(int entry_index, int output_index, CL_TYPE* key) {

        for (int index = 0; index < DICT_KEY_LEN; index++) {
            key[output_index + index] = dict_keys__DICT_ID[entry_index][index];
        }

    }

    void get_dict_value__DICT_ID(int entry_index, int output_index, CL_TYPE* value) {

        for (int index = 0; index < DICT_VALUE_LEN; index++) {
            value[output_index + index] = dict_values__DICT_ID[entry_index][index];
        }

    }

    void set_default_output_value__DICT_ID(int output_index, CL_TYPE* value) {

        for (int index = 0; index < DICT_VALUE_LEN; index++) {
            value[output_index + index] = CL_DEFAULT_VALUE;
        }

    }

    void set_default_dict_pair__DICT_ID(int entry_index) {

        for (int index = 0; index < DICT_KEY_LEN; index++) {
            dict_keys__DICT_ID[entry_index][index] = CL_DEFAULT_VALUE;
        }

        for (int index = 0; index < DICT_VALUE_LEN; index++) {
            dict_values__DICT_ID[entry_index][index] = CL_DEFAULT_VALUE;
        }
        
        dict_entries__DICT_ID[entry_index] = 0;

    }

    uint get_dict_key_size__DICT_ID(int entry_index) {

        for (int index = (DICT_KEY_LEN - 1); index >= 0; index--) {
            if (dict_keys__DICT_ID[entry_index][index] != CL_DEFAULT_VALUE) {
                return index + 1;
            }
        }

        return 0;
    }

    int get_size_dict_value__DICT_ID(int entry_index) {

        for (int index = (DICT_VALUE_LEN - 1); index >= 0; index--) {
            if (dict_values__DICT_ID[entry_index][index] != CL_DEFAULT_VALUE) {
                return index + 1;
            }
        }

        return 0;
    }
    
    bool is_dict_key_input_equal_to__DICT_ID(
        int first_dict_key_input_index,
        int second_dict_key_input_index,
        CL_TYPE* keys_input
        ) {
        for (int key_index = 0; key_index < DICT_KEY_LEN; key_index++) {
            if (keys_input[key_index + first_dict_key_input_index] != keys_input[key_index + second_dict_key_input_index]) {
                return false;
            }
        }
        return true;
    }

    bool is_dict_key_equal_to__DICT_ID(
        int entry_index,
        int key_input_index,
        CL_TYPE* keys_input
        ) {
        for (int key_index = 0; key_index < DICT_KEY_LEN; key_index++) {
            if (dict_keys__DICT_ID[entry_index][key_index] != keys_input[key_index + key_input_index]) {
                return false;
            }
        }
        return true;
    }

    bool is_dict_key_empty__DICT_ID(
        int entry_index
        ) {
        for (int key_index = 0; key_index < DICT_KEY_LEN; key_index++) {
            if (dict_keys__DICT_ID[entry_index][key_index] != CL_DEFAULT_VALUE) {
                return false;
            }
        }
        return true;
    }
    
    int get_first_dict_available_entry__DICT_ID() {

        for (int index = 0; index < DICT_KEY_LEN; index++) {
            if (dict_entries__DICT_ID[index] == 0) {
               return index;
            }
        }

        return KEYS_NOT_AVAILABLE;
    }
    "#;

const DICT_FUNCTIONS: &str = r#"
    int dict_insert__DICT_ID(int key_input_index, int value_input_index, CL_TYPE* key, CL_TYPE* value) {
        int entry_index = check_if_dict_key_exists__DICT_ID(key_input_index, key);

        if (entry_index >= 0) {

            set_dict_value__DICT_ID(entry_index, value_input_index, value);

        } else {

            for (int i = 0; i < DICT_CAPACITY; i++) {

                if (dict_entries__DICT_ID[i] != 0) {
                    continue;
                }

                entry_index = check_if_dict_key_exists__DICT_ID(key_input_index, key);

                if (entry_index >= 0) {
                    // entry_index = -100;
                    break;
                }

                int r = atomic_fetch_add(&dict_entries__DICT_ID[i], 1);

                if ( r == 0 ) {

                    set_dict_key__DICT_ID(i, key_input_index, key);
                    set_dict_value__DICT_ID(i, value_input_index, value);

                    entry_index = i;

                    break;
                }

            }

            // FIXME
            if (entry_index < 0) {
                entry_index = KEYS_NOT_AVAILABLE;
            }

        }

        return entry_index;
    }

    int dict_get__DICT_ID(int key_input_index, int value_output_index, CL_TYPE* key, CL_TYPE* value) {
        int entry_index = check_if_dict_key_exists__DICT_ID(key_input_index, key);

        if (entry_index >= 0) {

            get_dict_value__DICT_ID(entry_index, value_output_index, value);

        } else {

            set_default_output_value__DICT_ID(value_output_index, value);

        }

        return entry_index;
    }

    int dict_get_size__DICT_ID(int key_input_index, CL_TYPE* key, uint* value) {
        int entry_index = check_if_dict_key_exists__DICT_ID(key_input_index, key);

        if (entry_index >= 0) {

            *value = get_size_dict_value__DICT_ID(entry_index);

        } else {

            *value = 0;

        }

        return entry_index;
    }

    int dict_remove__DICT_ID(int key_input_index, CL_TYPE* key) {
        int entry_index = check_if_dict_key_exists__DICT_ID(key_input_index, key);

        if (entry_index >= 0) {

            set_default_dict_pair__DICT_ID(entry_index);

        }

        return entry_index;
    }
    "#;

const BASE_KERNELS: &str = r#"
    kernel void dict_reset__DICT_ID() {
        int i = get_global_id(0);
        
        for (int key_index = 0; key_index < DICT_KEY_LEN; key_index++) {
            dict_keys__DICT_ID[i][key_index] = CL_DEFAULT_VALUE;
        }
        for (int value_index = 0; value_index < DICT_VALUE_LEN; value_index++) {
            dict_values__DICT_ID[i][value_index] = CL_DEFAULT_VALUE;
        }

        dict_entries__DICT_ID[i] = 0;
    }

    kernel void dict_debug__DICT_ID(
        global CL_TYPE* keys_output,
        global CL_TYPE* values_output,
        global int* entries_output
        ) {
        int i = get_global_id(0);
        int key_output_index = i * DICT_KEY_LEN;
        int value_output_index = i * DICT_VALUE_LEN;

        for (int index = 0; index < DICT_KEY_LEN; index++) {
            keys_output[index + key_output_index] = dict_keys__DICT_ID[i][index];
        }

        for (int index = 0; index < DICT_VALUE_LEN; index++) {
            values_output[index + value_output_index] = dict_values__DICT_ID[i][index];
        }
        
        entries_output[i] = dict_entries__DICT_ID[i];
    }

    kernel void dict_get_keys__DICT_ID(
        global CL_TYPE* keys_output
        ) {

        int i = get_global_id(0);
        int key_output_index = i * DICT_KEY_LEN;

        for (int index = 0; index < DICT_KEY_LEN; index++) {
            keys_output[index + key_output_index] = dict_keys__DICT_ID[i][index];
        }
    }

    // kernel void dict_get_summary__DICT_ID(
    //     global uint* sizes_output,
    //     global int* meta_output
    //     ) {
    // 
    //     int i = get_global_id(0);
    // 
    //     sizes_output[i] = get_dict_key_size__DICT_ID(i);
    //     sizes_output[i + DICT_CAPACITY] = get_size_dict_value__DICT_ID(i);
    // 
    //     if (i == 0) {
    //         // empty_keys_count
    //         // ...
    //     }
    // }
    
    kernel void dict_get_summary__DICT_ID(
        queue_t q0,
        const uint capacity_device_local_work_size,
        global uint* sizes_output,
        global int* meta_output,
        global int* enqueue_kernel_output
        ) {
    
        clk_event_t evt0;
    
        enqueue_kernel_output[0] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(DICT_CAPACITY, capacity_device_local_work_size),
            0,
            NULL,
            &evt0,
            ^{
                int i = get_global_id(0);
        
                sizes_output[i] = get_dict_key_size__DICT_ID(i);
                sizes_output[i + DICT_CAPACITY] = get_size_dict_value__DICT_ID(i);
            }
        );
    
        enqueue_kernel_output[1] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            1,
            &evt0,
            NULL,
            ^{
               int empty_keys_count = 0;
               
               for (int index = 0; index < DICT_CAPACITY; index++) {
                   if (sizes_output[index] == 0) {
                      empty_keys_count++;
                   }
               }
            
               meta_output[0] = empty_keys_count;
            }
        );
    
        release_event(evt0);

    }

    kernel void read_on_dict__DICT_ID(
        const uint key_len,
        const uint value_len,
        global CL_TYPE* keys_input,
        global CL_TYPE* values_output,
        global int* indices_output
        ) {
        int i = get_global_id(0);
        int key_input_index = i * key_len;
        int value_output_index = i * value_len;

        indices_output[i] = dict_get__DICT_ID(
            key_input_index,
            value_output_index,
            keys_input,
            values_output
        );
    }

    kernel void read_value_size_on_dict__DICT_ID(
        const uint key_len,
        global CL_TYPE* keys_input,
        global uint* sizes_output,
        global int* indices_output
        ) {
        int i = get_global_id(0);
        int key_input_index = i * key_len;

        uint size = 0;

        indices_output[i] = dict_get_size__DICT_ID(
            key_input_index,
            keys_input,
            &size
        );
        sizes_output[i] = size;
    }

    kernel void write_to_dict__DICT_ID(
        const uint key_len,
        const uint value_len,
        global CL_TYPE* keys_input,
        global CL_TYPE* values_input,
        global int* indices_output
        ) {
        int i = get_global_id(0);
        int key_input_index = i * key_len;
        int value_input_index = i * value_len;

        indices_output[i] = dict_insert__DICT_ID(
            key_input_index,
            value_input_index,
            keys_input,
            values_input
        );
    }

    kernel void remove_from_dict__DICT_ID(
        const uint key_len,
        global CL_TYPE* keys_input,
        global int* indices_output
        ) {
        int i = get_global_id(0);
        int key_input_index = i * key_len;

        indices_output[i] = dict_remove__DICT_ID(
            key_input_index,
            keys_input
        );
    }
    "#;

const VERIFY_AND_WRITE_KERNEL: &str = r#"

    __global int dict_write_stack__DICT_ID[DICT_CAPACITY];
    __global int dict_write_stack_top__DICT_ID = -1;
    
    void dict_write_stack_push__DICT_ID(int value) {
    
        if (dict_write_stack_top__DICT_ID >= -1 && dict_write_stack_top__DICT_ID <= DICT_MAX_CAPACITY) {
            int entry_index = (atomic_fetch_add(&dict_write_stack_top__DICT_ID, 1) + 1);

            if (DICT_MAX_CAPACITY >= entry_index) {

                dict_write_stack__DICT_ID[entry_index] = value;

            } else {

                // FIXME stack overflow
                atomic_store(&dict_write_stack_top__DICT_ID, DICT_MAX_CAPACITY);

            }
        }
        
    }
    
    int dict_write_stack_pop__DICT_ID() {
        int entry_index = -1;
    
        if (dict_write_stack_top__DICT_ID >= 0) {
        
            int index = atomic_fetch_sub(&dict_write_stack_top__DICT_ID, 1);

            if (index >= 0) {
                entry_index = dict_write_stack__DICT_ID[index];
            }
            
        }
        
        return entry_index;
    }

    kernel void check_duplicates_dict_key_input__DICT_ID(
        const int keys_global_work_size,
        global CL_TYPE* keys_input,
        global int* indices_output
        ) {

        int i = get_global_id(0);
        int key_input_index = i * DICT_KEY_LEN;

        // input duplicates

        // last index no check next indices
        if (i != (keys_global_work_size - 1)) {

            for (int index = i + 1; index < keys_global_work_size; index++) {

                int second_dict_key_input_index = index * DICT_KEY_LEN;

                bool is_equal = is_dict_key_input_equal_to__DICT_ID(
                    key_input_index,
                    second_dict_key_input_index,
                    keys_input
                );

                if (is_equal == true) {
                    indices_output[i] = DUPLICATE_KEY;
                    break;
                }

            }
        }

        if (indices_output[i] == KEY_NOT_EXIST) {

            // exist in dict

            for (int index = 0; index < DICT_CAPACITY; index++) {

                bool is_equal = is_dict_key_equal_to__DICT_ID(
                    index,
                    key_input_index,
                    keys_input
                );

                if (is_equal == true) {
                    indices_output[i] = index;
                    break;
                }

            }

        }
                
        if (dict_entries__DICT_ID[i] == 0) {
            dict_write_stack_push__DICT_ID(i);
        }

    }

    kernel void confirm_dict_insert__DICT_ID(
        global CL_TYPE* keys_input,
        global CL_TYPE* values_input,
        global int* indices_output
        ) {
        int i = get_global_id(0);
        int key_input_index = i * DICT_KEY_LEN;
        int value_input_index = i * DICT_VALUE_LEN;

        if (indices_output[i] == DUPLICATE_KEY) {

           // pass

        } else if (indices_output[i] == KEY_NOT_EXIST) {

           int entry_index = dict_write_stack_pop__DICT_ID();

           if (entry_index >= 0) {
               set_dict_key__DICT_ID(entry_index, key_input_index, keys_input);
               set_dict_value__DICT_ID(entry_index, value_input_index, values_input);
               
               dict_entries__DICT_ID[i] = 1;
           }

           indices_output[i] = entry_index;

        } else {
           // key exist

           int entry_index = indices_output[i];

           set_dict_key__DICT_ID(entry_index, key_input_index, keys_input);
           set_dict_value__DICT_ID(entry_index, value_input_index, values_input);

        }

    }

    kernel void verify_and_write_in_dict__DICT_ID(
        queue_t q0,
        const uint keys_global_work_size,
        const uint keys_local_work_size,
        global CL_TYPE* keys_input,
        global CL_TYPE* values_input,
        global int* indices_output,
        global int* enqueue_kernel_output
        ) {

        // reset
        for (int index = 0; index < keys_global_work_size; index++) {
            indices_output[index] = KEY_NOT_EXIST;
        }

        clk_event_t evt0;

        enqueue_kernel_output[0] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(keys_global_work_size, keys_local_work_size),
            0,
            NULL,
            &evt0,
            ^{
               check_duplicates_dict_key_input__DICT_ID(
                    keys_global_work_size,
                    keys_input,
                    indices_output
               );
            }
        );

        enqueue_kernel_output[1] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(keys_global_work_size, keys_local_work_size),
            1,
            &evt0,
            NULL,
            ^{
               confirm_dict_insert__DICT_ID(
                    keys_input,
                    values_input,
                    indices_output
               );
            }
        );

        release_event(evt0);

    }
    "#;

const VERIFY_AND_REMOVE_KERNEL: &str = r#"

    __global int dict_remove_stack__DICT_ID[DICT_CAPACITY];
    __global int dict_remove_stack_top__DICT_ID = -1;
    
    void dict_remove_stack_push__DICT_ID(int value) {
    
        if (dict_remove_stack_top__DICT_ID >= -1 && dict_remove_stack_top__DICT_ID <= DICT_MAX_CAPACITY) {
            int entry_index = (atomic_fetch_add(&dict_remove_stack_top__DICT_ID, 1) + 1);

            if (DICT_MAX_CAPACITY >= entry_index) {

                dict_remove_stack__DICT_ID[entry_index] = value;

            } else {

                // FIXME stack overflow
                atomic_store(&dict_remove_stack_top__DICT_ID, DICT_MAX_CAPACITY);

            }
        }
        
    }
    
    int dict_remove_stack_pop__DICT_ID() {
        int entry_index = -1;
    
        if (dict_remove_stack_top__DICT_ID >= 0) {
        
            int index = atomic_fetch_sub(&dict_remove_stack_top__DICT_ID, 1);

            if (index >= 0) {
                entry_index = dict_remove_stack__DICT_ID[index];
            }
            
        }
        
        return entry_index;
    }

    kernel void check_duplicates_dict_key_input__DICT_ID_def_2(
        const int keys_global_work_size,
        global CL_TYPE* keys_input,
        global int* indices_output
        ) {

        int i = get_global_id(0);
        int key_input_index = i * DICT_KEY_LEN;

        // input duplicates

        // last index no check next indices
        if (i != (keys_global_work_size - 1)) {

            for (int index = i + 1; index < keys_global_work_size; index++) {

                int second_dict_key_input_index = index * DICT_KEY_LEN;

                bool is_equal = is_dict_key_input_equal_to__DICT_ID(
                    key_input_index,
                    second_dict_key_input_index,
                    keys_input
                );

                if (is_equal == true) {
                    indices_output[i] = DUPLICATE_KEY;
                    break;
                }

            }
        }

        if (indices_output[i] == KEY_NOT_EXIST) {

            // exist in dict

            for (int index = 0; index < DICT_CAPACITY; index++) {

                bool is_equal = is_dict_key_equal_to__DICT_ID(
                    index,
                    key_input_index,
                    keys_input
                );

                if (is_equal == true) {
                    indices_output[i] = index;
                    break;
                }
                

            }

        }

        if (dict_entries__DICT_ID[i] == 0) {
            dict_write_stack_push__DICT_ID(i);
        }

    }

    kernel void confirm_dict_remove__DICT_ID(
        global CL_TYPE* keys_input,
        global int* indices_output
        ) {
        int i = get_global_id(0);

        int entry_index = indices_output[i];

        if (entry_index >= 0) {
           set_default_dict_pair__DICT_ID(entry_index);
           dict_entries__DICT_ID[entry_index] = 0;
        }

    }

    kernel void verify_and_remove_in_dict__DICT_ID(
        queue_t q0,
        const uint keys_global_work_size,
        const uint keys_local_work_size,
        global CL_TYPE* keys_input,
        global int* indices_output,
        global int* enqueue_kernel_output
        ) {

        // reset
        for (int index = 0; index < keys_global_work_size; index++) {
            indices_output[index] = KEY_NOT_EXIST;
        }

        clk_event_t evt0;

        enqueue_kernel_output[0] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(keys_global_work_size, keys_local_work_size),
            0,
            NULL,
            &evt0,
            ^{
               check_duplicates_dict_key_input__DICT_ID_def_2(
                    keys_global_work_size,
                    keys_input,
                    indices_output
               );
            }
        );

        enqueue_kernel_output[1] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(keys_global_work_size, keys_local_work_size),
            1,
            &evt0,
            NULL,
            ^{
               confirm_dict_remove__DICT_ID(
                    keys_input,
                    indices_output
               );
            }
        );

        release_event(evt0);

    }
    "#;

impl<T: ClTypeTrait> DictSrc<T> {
    pub fn generate_dict_program_source_v1(&self) -> String {
        if self.is_empty() {
            return String::new();
        }

        let mut globals = String::new();

        let mut base_functions = String::new();
        let mut dict_functions = String::new();
        let mut base_kernels = String::new();

        let mut verify_and_write_kernels = String::new();
        let mut verify_and_remove_kernels = String::new();

        for config in self.get_configs() {
            let template = common_replace(GLOBALS, config);
            globals.push_str(&template);

            let template = common_replace(BASE_FUNCTIONS, config);
            base_functions.push_str(&template);

            let template = common_replace(DICT_FUNCTIONS, config);
            dict_functions.push_str(&template);

            let template = common_replace(BASE_KERNELS, config);
            base_kernels.push_str(&template);

            let template = common_replace(VERIFY_AND_WRITE_KERNEL, config);
            verify_and_write_kernels.push_str(&template);

            let template = common_replace(VERIFY_AND_REMOVE_KERNEL, config);
            verify_and_remove_kernels.push_str(&template);
        }

        format!(
            "
    /// *** DICT SRC *** ///

    /// constants

    const int KEYS_NOT_AVAILABLE = -1;
    const int KEY_NOT_EXIST = -2;
    const int DUPLICATE_KEY = -3;

    /// globals
    {globals}

    /// kernels
    {base_functions}
    {dict_functions}

    {base_kernels}

    {verify_and_write_kernels}
    {verify_and_remove_kernels}

    /// *** DICT SRC *** ///
    "
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a() {
        let mut dict_src: DictSrc<i32> = DictSrc::new();
        dict_src.add(8, 8, 8);

        let program_source = dict_src.generate_dict_program_source_v1();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut dict_src: DictSrc<i16> = DictSrc::new();
        dict_src.add(8, 16, 32);
        dict_src.add(32, 64, 64);
        dict_src.add(16, 32, 32);

        let program_source = dict_src.generate_dict_program_source_v1();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
