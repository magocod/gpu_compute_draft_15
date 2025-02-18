# Opencl Collections

* At the time of making these examples, I was a complete newbie to opencl, I chose to write the cl code in text,
  at first it seemed like a good idea (It was not for advanced cases)

Implementations that seek to provide containers the same (similar) to those provided by rust std and c++ containers

Inspired by:
* https://github.com/stotko/stdgpu
* https://github.com/NVIDIA/cuCollections
* https://github.com/ROCm/hipCollections

## Rust std collections

https://doc.rust-lang.org/std/collections/

* Sequences:
  - [ ] Vec
  - [ ] VecDeque
  - [ ] LinkedList

* Maps:
  - [x] HashMap -> OpenclCollection.Map (without hashing) or OpenclCollection.Dictionary (without hashing)
  - [ ] BTreeMap

* Sets:
  - [x] HashSet -> OpenclCollection.ArraySet (without hashing)
  - [ ] BTreeSet

* Misc:
  - [ ] BinaryHeap

    
## C++ Containers: 

https://cplusplus.com/reference/stl/

* Sequence containers:
  - [ ] array
  - [ ] vector
  - [ ] deque
  - [ ] forward_list
  - [ ] list

* Container adaptors:
  - [x] stack	LIFO stack -> OpenclCollection.Stack
  - [x] queue	FIFO queue -> OpenclCollection.LinearQueue or OpenclCollection.CircularQueue
  - [ ] priority_queue

* Associative containers:
  - [ ] set
  - [ ] multiset
  - [ ] map
  - [ ] multimap

* Unordered associative containers:
  - [x] unordered_set -> OpenclCollection.ArraySet (without hashing)
  - [ ] unordered_multiset
  - [x] unordered_map -> OpenclCollection.Map (without hashing) or OpenclCollection.Dictionary (without hashing)
  - [ ] unordered_multimap

## Collections

* Stack

* Set

* Dictionary

* Queue
  * LinearQueue
  * PriorityQueue
  * CircularQueue

* Cache

* Map


test
```bash
cargo test -- --test-threads=1
```

test - errors
```bash
cargo test
```

TODO

* fix array_set_v1 errors
* fix mini_lru errors
* fix lru errors


map_append error during relocation
