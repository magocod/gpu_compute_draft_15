use crate::config::ClTypeTrait;
use crate::map::config::MapSrc;
use crate::map::kernel::common_replace;

const COPY_FN: &str = r#"
    int map_copy_value(
        uint map_id,
        // from map block
        int from_map_value_len,
        int from_entry_index,
        int from_last_index,
        // to map block
        int to_map_value_len,
        int to_entry_index,
        int to_start_index
        ) {
        int r = -1;

        switch (from_map_value_len) {
            SWITCH_CASE
        }

        return r;
    }
    "#;

const SWITCH_CASE: &str = r#"
            // BLOCK_NAME
            case MAP_VALUE_LEN:

                if (to_map_value_len == MAP_VALUE_LEN) {
                    for (int index = 0; index < from_last_index; index++) {
                        map_keys__BLOCK_NAME[map_id][to_entry_index][index + to_start_index] = map_keys__BLOCK_NAME[map_id][from_entry_index][index];
                    }
                    r = 0;
                }
                TO_BLOCK_CASE

                break;
     "#;

const TO_BLOCK_CASE: &str = r#"
                // TO_BLOCK_NAME
                else if (to_map_value_len == MAP_VALUE_LEN) {
                    for (int index = 0; index < from_last_index; index++) {
                        map_values__TO_BLOCK_NAME[map_id][to_entry_index][index + to_start_index] = map_values__FROM_BLOCK_NAME[map_id][from_entry_index][index];
                    }
                    r = 0;
                }
     "#;

const COPY_KERNELS: &str = r#"
    kernel void map_copy_value_for__BLOCK_NAME(
        const uint map_id,
        global int* copy_params_input,
        global int* copy_output
        ) {
        int i = get_global_id(0);
        int param_index = i * 5;

        copy_output[i] = map_copy_value(
            map_id,
            // from
            MAP_VALUE_LEN,
            copy_params_input[param_index + FROM_ENTRY_INDEX],
            copy_params_input[param_index + FROM_LAST_INDEX],            
            // to
            copy_params_input[param_index + TO_VALUE_LEN],
            copy_params_input[param_index + TO_ENTRY_INDEX],
            copy_params_input[param_index + TO_START_INDEX]
        );
    }
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_copy_program_src(&self) -> String {
        let map_blocks = self.get_configs();

        let mut copy_kernels = String::new();
        let mut switch_cases = String::new();

        for (i, config) in map_blocks.iter().enumerate() {
            let template = common_replace(COPY_KERNELS, config);
            copy_kernels.push_str(&template);

            let case = common_replace(SWITCH_CASE, config).replace("BLOCK_INDEX", &i.to_string());
            let mut to_block_case = String::new();

            for to_config in map_blocks.iter() {
                if config.value_len >= to_config.value_len {
                    continue;
                }

                let body = TO_BLOCK_CASE
                    .replace("TO_BLOCK_NAME", &to_config.name)
                    .replace("FROM_BLOCK_NAME", &config.name)
                    .replace("MAP_VALUE_LEN", &to_config.value_len.to_string());

                to_block_case.push_str(&body)
            }

            let c = case.replace("TO_BLOCK_CASE", &to_block_case);
            switch_cases.push_str(&c);
        }

        let copy_fn = COPY_FN.replace("SWITCH_CASE", &switch_cases);

        format!(
            "
    /// - MAP_COPY START ///

    /// constants
    // ...

    /// globals

    // from
    const int FROM_ENTRY_INDEX = 0;
    const int FROM_LAST_INDEX = 1;

    // to
    const int TO_VALUE_LEN = 2;
    const int TO_ENTRY_INDEX = 3;
    const int TO_START_INDEX = 4;

    /// kernels
    {copy_fn}
    {copy_kernels}

    /// - MAP_COPY END ///
        "
        )
    }

    pub fn add_map_copy_program_src(&mut self) -> &mut Self {
        let src = self.generate_map_copy_program_src();
        self.optional_sources.push(src);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(256, 16);

        let program_source = map_src.generate_map_copy_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(256, 8);
        map_src.add(512, 16);

        let program_source = map_src.generate_map_copy_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut map_src: MapSrc<i32> = MapSrc::default();
        map_src.add(256, 8);
        map_src.add(512, 32);
        map_src.add(1024, 16);

        let program_source = map_src.generate_map_copy_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
