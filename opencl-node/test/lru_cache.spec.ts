import { LRUCache } from "lru-cache";
import {Device, Payload} from "../index";
import {unreachable} from "./helpers";

const expectedLruKeys1 = [
    'key_1', 'key_4',
    'key_3', 'key_7',
    'key_6', 'key_5',
    'key_2', 'key_0'
];

const expectedMiniLruKeys1 = [
    1, 4,
    3, 7,
    6, 5,
    2, 0
];

const expectedLruKeys2 = [
    'key_8',  'key_7',
    'key_14', 'key_10',
    'key_3',  'key_9',
    'key_6',  'key_5'
];

const expectedMiniLruKeys2 = [
    8,  7,
    14, 10,
    3,  9,
    6,  5
];

describe("lru cache", () => {
    describe("lru-cache js", function () {
        it("example 1", () => {
            const cache = new LRUCache<string, string>({
                max: 8,
                maxSize: 5000,
                sizeCalculation: (_value: any, _key: any) => {
                    return 1
                },
            })

            cache.set('key_0', 'value')
            cache.get('key_0')
            console.log("get('key_0'): ", cache.get("key_0"))

            console.log("cache.keys()", Array.from(cache.keys()));

            for (let i = 1; i <= 7; i++) {
                let k = `key_${i}`;
                cache.set(k, i.toString())
            }

            cache.get('key_3')
            cache.get('key_4')

            cache.set('key_1', "update")

            let keys = Array.from(cache.keys());

            console.log("get('key_1'): ", cache.get("key_1"))

            expect(keys).toEqual(expectedLruKeys1)
        });

        it("example 2", () => {
            const cache = new LRUCache<string, string>({
                max: 8,
                maxSize: 5000,
                sizeCalculation: (_value: any, _key: any) => {
                    return 1
                },
            })

            for (let i = 1; i <= 10; i++) {
                let k = `key_${i}`;
                cache.set(k, i.toString())
            }

            cache.set('key_3', "updated")
            cache.get('key_10')
            cache.set('key_14', "new")
            cache.get('key_7')
            cache.set('key_8', "updated")

            let keys = Array.from(cache.keys())

            expect(keys).toEqual(expectedLruKeys2)
        });
    })

    describe("lru_cache cl", function () {
        let device: Device;

        beforeAll(function () {
            device = new Device({
                cache: {
                    lru: [
                        // 256 byte
                        { capacity: 16, valueLen: 256 },
                        { capacity: 8, valueLen: 256 },
                        { capacity: 32, valueLen: 256 },
                    ]
                }
            });
        })

        it("example 1", () => {
            const cache = device.createLruCache(0);

            cache.set('key_0', 'value')
            cache.get('key_0')
            console.log("get('key_0'):", cache.get("key_0"))

            // cache.debug();
            console.log("cache.keys()", cache.keys())

            for (let i = 1; i <= 7; i++) {
                let k = `key_${i}`;
                cache.set(k, i.toString())
            }

            cache.get('key_3')
            cache.get('key_4')

            cache.set('key_1', "update")

            // cache.debug();
            let keys = cache.keys();
            console.log("cache.keys()", keys)

            console.log("get('key_1'): ", cache.get("key_1"))

            expect(keys).toEqual(expectedLruKeys1)
        })

        it("example 2", () => {
            const cache = device.createLruCache(1);

            for (let i = 1; i <= 10; i++) {
                let k = `key_${i}`;
                cache.set(k, i.toString())
            }

            cache.set('key_3', "updated")
            cache.get('key_10')
            cache.set('key_14', "new")
            cache.get('key_7')
            cache.set('key_8', "updated")

            let keys = cache.keys();
            console.log("cache.keys()", keys)

            expect(keys).toEqual(expectedLruKeys2)
        });

        it("example 1 v2", () => {
            const cache = device.createLruCache(0);

            cache.set('key_0', 'value')
            cache.get('key_0')
            console.log("get('key_0'):", cache.get("key_0"))

            // cache.debug();
            console.log("cache.keys()", cache.keys())

            let payloads: Payload[] = [];
            for (let i = 1; i <= 7; i++) {
                payloads.push({
                    key:  `key_${i}`,
                    value: i.toString()
                })
            }
            cache.setMany(payloads)

            cache.get('key_3')
            cache.get('key_4')

            cache.set('key_1', "update")

            // getKeys(cache)
            // cache.debug();
            let keys = cache.keys();
            console.log("cache.keys()", keys)

            console.log("get('key_1'): ", cache.get("key_1"))

            expect(keys).toEqual(expectedLruKeys1)

        })

        it("cache config empty", () => {
            const invalid_device = new Device({})

            try {
                invalid_device.createLruCache(0)
                unreachable()
            } catch (e) {
                console.log(e)
                let error = e as Error;
                expect(error.message).toEqual("cache src empty")
            }
        });

        it("invalid lru cache id", () => {
            const invalid_device = new Device({
                cache: {
                    lru: [
                        {
                            valueLen: 256,
                            capacity: 32
                        }
                    ]
                }
            })
            // invalid_device.printCacheSrc();

            try {
                invalid_device.createLruCache(10)
                unreachable()
            } catch (e) {
                console.log(e)
                let error = e as Error;
                expect(error.message).toEqual("opencl collection error code: -700")
            }
        });
    })

    describe("mini_lru_cache cl", function () {
        let device: Device;

        beforeAll(function () {
            device = new Device({
                cache: {
                    miniLru: [
                        { capacity: 16 },
                        { capacity: 8 },
                        { capacity: 32 },
                    ]
                }
            });
        })

        it("example 1", () => {
            const cache = device.createMiniLruCache(0)

            cache.set(0, 1000)
            cache.get(0)
            console.log("get('key'): ", cache.get(0))

            // getKeys(cache)
            // cache.debug();
            console.log("cache.keys()", cache.keys())

            for (let i = 1; i <= 7; i++) {
                cache.set(i, i + 10)
            }

            cache.get(3)
            cache.get(4)

            cache.set(1, 111)

            // getKeys(cache)
            // cache.debug();
            let keys = cache.keys();
            console.log("cache.keys()", keys)

            console.log("get('key_1'): ", cache.get(1))

            expect(keys).toEqual(expectedMiniLruKeys1)
        });

        it("example 2", () => {
            const cache = device.createMiniLruCache(1);

            for (let i = 1; i <= 10; i++) {
                cache.set(i, i)
            }

            cache.set(3, 30)
            cache.get(10)
            cache.set(14, 14)
            cache.get(7)
            cache.set(8, 80)

            // getKeys(cache)
            // cache.debug();
            let keys = cache.keys();
            console.log("cache.keys()", keys)

            expect(keys).toEqual(expectedMiniLruKeys2)
        });

        it("cache config empty", () => {
            const invalid_device = new Device({})

            try {
                invalid_device.createMiniLruCache(0)
                unreachable()
            } catch (e) {
                console.log(e)
                let error = e as Error;
                expect(error.message).toEqual("cache src empty")
            }
        });

        it("invalid lru cache id", () => {
            const invalid_device = new Device({
                cache: {
                    miniLru: [
                        {
                            capacity: 32
                        }
                    ]
                }
            })
            invalid_device.printCacheSrc();

            try {
                invalid_device.createMiniLruCache(10)
                unreachable()
            } catch (e) {
                console.log(e)
                let error = e as Error;
                expect(error.message).toEqual("opencl collection error code: -600")
            }
        });
    })
})