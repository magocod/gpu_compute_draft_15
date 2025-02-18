use crate::config::ClTypeTrait;
use crate::map::config::{check_local_work_size, MapSrc};
use crate::map::kernel::common_replace;

const MAP_GET_SUMMARY_KERNEL: &str = r#"
    kernel void map_get_summary(
        queue_t q0,
        const uint map_id,
        global int* output,
        global int* enqueue_kernel_output
        ) {
        clk_event_t evt0;

        enqueue_kernel_output[CMQ_CHECK_MAP_KEYS] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(1, 1),
            0,
            NULL,
            &evt0,
            ^{
               check_map_keys(
                    q0,
                    map_id,
                    output,
                    enqueue_kernel_output
               );
            }
        );

        enqueue_kernel_output[CMQ_CONFIRM_SUMMARY] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(TOTAL_BLOCKS + 1, TOTAL_BLOCKS_DEVICE_LOCAL_WORK_SIZE + 1),
            1,
            &evt0,
            NULL,
            ^{
               confirm_summary(
                    map_id,
                    output
               );
            }
        );

        release_event(evt0);
    }
    "#;

const CHECK_MAP_KEYS_KERNEL: &str = r#"
    kernel void check_map_keys(
        queue_t q0,
        const uint map_id,
        global int* output,
        global int* enqueue_kernel_output
        ) {

        KERNEL_BODY

    }
    "#;

const CHECK_MAP_KEYS_ENQUEUE_KERNEL: &str = r#"
        // BLOCK_NAME
        enqueue_kernel_output[CMQ_CHECK_KEY__BLOCK_NAME] = enqueue_kernel(
            q0,
            CLK_ENQUEUE_FLAGS_NO_WAIT,
            ndrange_1D(MAP_CAPACITY, CAPACITY_DEVICE_LOCAL_WORK_SIZE),
            ^{
               check_map_keys__BLOCK_NAME(
                    map_id,
                    output
               );
            }
        );
     "#;

const CHECK_MAP_KEYS_KERNEL_FOR_BLOCK: &str = r#"
    kernel void check_map_keys__BLOCK_NAME(
        const uint map_id,
        global int* output
        ) {

        int i = get_global_id(0);

        tmp_for_map_get_summary[map_id][i + TMP_INDEX__BLOCK_NAME] = get_map_key_size__BLOCK_NAME(
            map_id,
            i
        );

        // It would be better to use a barrier and count the general result from here ?
        // barrier(CLK_GLOBAL_MEM_FENCE);
        //
        // if (i == 0) {
        //    ...
        // }
    }
    "#;

const CONFIRM_SUMMARY_KERNEL: &str = r#"
    kernel void confirm_summary(
        const uint map_id,
        global int* output
        ) {

        int i = get_global_id(0);

        // general result
        if (i >= TOTAL_BLOCKS) {

            int general_summary = 0;

            for (int index = 0; index < TMP_LEN; index++) {
                if (tmp_for_map_get_summary[map_id][index] > 0) {
                    general_summary += 1;
                }
            }

            output[SUMMARY_INDEX__GENERAL] = general_summary;

        } else {
            // result per block
            int block_summary = 0;

            switch (i) {
                SWITCH_CASES
            }

        }
    }
    "#;

const SWITCH_SUMMARY: &str = r#"
                // BLOCK_NAME
                case BLOCK_INDEX:

                    for (int index = TMP_INDEX__BLOCK_NAME; index < TMP_INDEX__BLOCK_NAME + MAP_CAPACITY; index++) {
                        if (tmp_for_map_get_summary[map_id][index] > 0) {
                            block_summary += 1;
                        }
                    }

                    output[SUMMARY_INDEX__BLOCK_NAME] = block_summary;

                    break;
    "#;

// TODO explain
const GLOBAL_TMP_ARRAY: &str = r#"
    global int tmp_for_map_get_summary[TOTAL_MAPS][TMP_LEN];

    kernel void get_tmp_for_map_get_summary(
        const uint map_id,
        global int* output
        ) {
        int i = get_global_id(0);
        output[i] = tmp_for_map_get_summary[map_id][i];
    }

    // TODO reset kernel
    "#;

const CMQ_CONST_DEF: &str = r#"
    const int CMQ_CHECK_KEY__BLOCK_NAME = BLOCK_INDEX;
    "#;

const TMP_CONST_DEF: &str = r#"
    const int TMP_INDEX__BLOCK_NAME = BLOCK_INDEX;
    "#;

const SUMMARY_CONST_DEF: &str = r#"
    const int SUMMARY_INDEX__BLOCK_NAME = BLOCK_INDEX;
    "#;

impl<T: ClTypeTrait> MapSrc<T> {
    pub fn generate_map_get_summary_program_src(&self) -> String {
        // const definitions
        let mut cmq_const_def = String::new();
        let mut tmp_const_def = String::new();
        let mut summary_const_def = String::new();

        // kernels
        let mut check_map_keys_enqueue_kernel = String::new();
        let mut check_map_key_kernels = String::new();

        let mut get_summary_switch_case = String::new();

        let map_blocks = self.get_configs();

        let total_blocks = map_blocks.len();

        let cmq_check_map_keys_index = total_blocks;
        let cmq_confirm_summary_index = total_blocks + 1;

        let general_result_index = total_blocks;

        for (i, config) in map_blocks.iter().enumerate() {
            let tmp_index: usize = map_blocks[0..i].iter().map(|x| x.capacity).sum();

            let template = common_replace(TMP_CONST_DEF, config)
                .replace("BLOCK_INDEX", &tmp_index.to_string());
            tmp_const_def.push_str(&template);

            let template =
                common_replace(CMQ_CONST_DEF, config).replace("BLOCK_INDEX", &i.to_string());
            cmq_const_def.push_str(&template);

            let template =
                common_replace(SUMMARY_CONST_DEF, config).replace("BLOCK_INDEX", &i.to_string());
            summary_const_def.push_str(&template);

            let template = common_replace(CHECK_MAP_KEYS_KERNEL_FOR_BLOCK, config);
            check_map_key_kernels.push_str(&template);

            let template = common_replace(CHECK_MAP_KEYS_ENQUEUE_KERNEL, config)
                .replace("BLOCK_INDEX", &i.to_string());
            check_map_keys_enqueue_kernel.push_str(&template);

            let template =
                common_replace(SWITCH_SUMMARY, config).replace("BLOCK_INDEX", &i.to_string());
            get_summary_switch_case.push_str(&template);
        }

        let tmp_len = self.get_maximum_assignable_keys();

        let global_tmp_array = GLOBAL_TMP_ARRAY
            .replace("TMP_LEN", &tmp_len.to_string())
            .replace("TOTAL_MAPS", &self.get_total_maps().to_string());

        let check_map_keys_kernel =
            CHECK_MAP_KEYS_KERNEL.replace("KERNEL_BODY", &check_map_keys_enqueue_kernel);

        let get_summary_kernel = CONFIRM_SUMMARY_KERNEL
            .replace("TOTAL_BLOCKS", &total_blocks.to_string())
            .replace("TMP_LEN", &tmp_len.to_string())
            .replace("SWITCH_CASES", &get_summary_switch_case);

        let total_blocks_device_local_work_size = check_local_work_size(total_blocks).to_string();

        let get_map_summary_kernel = MAP_GET_SUMMARY_KERNEL
            .replace(
                "TOTAL_BLOCKS_DEVICE_LOCAL_WORK_SIZE",
                &total_blocks_device_local_work_size,
            )
            .replace("TOTAL_BLOCKS", &total_blocks.to_string());

        format!(
            "
    /// - MAP_GET_SUMMARY START ///

    /// constants
    {tmp_const_def}

    {summary_const_def}
    const int SUMMARY_INDEX__GENERAL = {general_result_index};

    {cmq_const_def}
    const int CMQ_CHECK_MAP_KEYS = {cmq_check_map_keys_index};
    const int CMQ_CONFIRM_SUMMARY = {cmq_confirm_summary_index};

    /// globals
    {global_tmp_array}

    /// kernels
    {check_map_key_kernels}

    {check_map_keys_kernel}
    {get_summary_kernel}

    {get_map_summary_kernel}

    /// - MAP_GET_SUMMARY END ///
        "
        )
    }

    pub fn add_map_get_summary_program_src(&mut self) -> &mut Self {
        let src = self.generate_map_get_summary_program_src();
        self.optional_sources.push(src);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::KB;

    #[test]
    fn test_a() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(KB, 16);

        let program_source = map_src.generate_map_get_summary_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_b() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 16);

        let program_source = map_src.generate_map_get_summary_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }

    #[test]
    fn test_c() {
        let mut map_src: MapSrc<i32> = MapSrc::new(2);
        map_src.add(256, 8);
        map_src.add(512, 32);
        map_src.add(1024, 16);

        let program_source = map_src.generate_map_get_summary_program_src();
        println!("{program_source}");
        assert!(!program_source.is_empty());
    }
}
