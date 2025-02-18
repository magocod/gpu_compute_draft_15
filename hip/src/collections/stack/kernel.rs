pub const STACK_KERNEL_SOURCE: &str = r#"
    const int STACK_CAPACITY = CURRENT_CAPACITY;
    const int STACK_MAX_CAPACITY = STACK_CAPACITY - 1;

    __device__ int stack[STACK_CAPACITY];
    __device__ int stack_top = -1;
    
    __device__ int stack_push(int* v) {
        int rear_i = -1;

        if (stack_top >= -1 && stack_top <= STACK_MAX_CAPACITY) {
       
            // rear_i = stack_top + 1;
            // stack_top++;
            
            rear_i = (atomicAdd(&stack_top, 1) + 1);

            if (STACK_MAX_CAPACITY >= rear_i) {

                stack[rear_i] = *v;

            } else {

                rear_i = - 1;
                // FIXME stack overflow
                stack_top = STACK_MAX_CAPACITY;

            }
        }

        return rear_i;
    }

    __device__ int stack_pop(int* v) {
        int front_i = -1;

        if (stack_top >= 0) {
        
            // front_i = stack_top;
            // stack_top--;
        
            front_i = atomicSub(&stack_top, 1);

            if (front_i >= 0) {
                *v = stack[front_i];
            }
        }

        // FIXME stack overflow
        if (stack_top < -1) {
            stack_top = -1;
        }

        return front_i;
    }
    
    extern "C" __global__ void stack_debug(
        int* items_output,
        int* meta_output
        ) {
        int i = threadIdx.x;

        items_output[i] = stack[i];

        if (i == 0) {
            meta_output[0] = stack_top;
        }
    }
    
    extern "C" __global__ void write_to_stack(
        int* input,
        int* output
        ) {
        int i = threadIdx.x;
        
        output[i] = stack_push(&input[i]);
    }

    extern "C" __global__ void read_on_stack(
        int* output
        ) {
        int i = threadIdx.x;
    
        int v = -1;
        stack_pop(&v);
    
        output[i] = v;
    }
    
    "#;

pub fn generate_stack_program_source(capacity: usize) -> String {
    STACK_KERNEL_SOURCE.replace("CURRENT_CAPACITY", &capacity.to_string())
}
