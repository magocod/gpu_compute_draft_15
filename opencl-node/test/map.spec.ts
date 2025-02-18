import {Device} from "../index";
import {unreachable} from "./helpers";

const expectedMapKeys1 = ['a', 'b', 'c', 'd', 'e', 'f'];

const expectedMapKeys2 = ['a', 'b', 'c', 'd', 'e'];

const expectedMapKeys3 = ['a', 'b', 'c']

describe("map", () => {
    describe("map js", function () {
        it("set, get", () => {
            const m = new Map();
            console.log("set")
            console.log(m.set("a", 10));
            console.log(m.set("b", false));
            console.log(m.set("c", [20]));
            console.log(m.set("d", "hello"));
            console.log(m.set("e", { message: "hello" }));
            console.log(m.set("f", null));

            console.log("get")
            console.log(m.get("a"));
            console.log(m.get("b"));
            console.log(m.get("c"));
            console.log(m.get("d"));
            console.log(m.get("e"));
            console.log(m.get("f"));

            let keys = Array.from(m.keys());
            console.log(keys)
            expect(keys).toEqual(expectedMapKeys1)
            // TODO assert values
        });

        it("not typing", () => {
            const m = new Map();
            m.set("a", 10);
            m.set("b", [20]);
            m.set("c", false);
            m.set("d", "hello");
            m.set("e", { propertyA: "a", propertyB: "b" });

            console.log("a", m.get("a"))
            console.log("b", m.get("b"))
            console.log("c", m.get("c"))
            console.log("d", m.get("d"))
            console.log("e", m.get("e"))
            console.log("not_exist", m.get("not_exist"))

            let keys = Array.from(m.keys());
            console.log(keys)
            expect(keys).toEqual(expectedMapKeys2)
            // TODO assert values
        });

        it("typing", () => {
            const m: Map<string, number[]> = new Map();
            m.set("a", [2]);
            m.set("b", [3, 4]);
            m.set("c", []);
            // error
            // m.set(1, "2");
            // m.set("d", ["1"]);

            let keys = Array.from(m.keys());
            console.log(keys)
            expect(keys).toEqual(expectedMapKeys3)
            // TODO assert values
        });
    })

    describe("map cl", function () {
        let device: Device;

        beforeAll(function () {
            device = new Device({
                map: {
                    totalMaps: 2,
                    blocks: [
                        // 1 kb
                        { valueLen: 1024, capacity: 8 },
                        // 2 kb
                        { valueLen: 1024 * 2, capacity: 32 },
                    ]
                },
            });
            // device.printMapSrc();
        })

        // afterAll(function () {
        //     // console.log("row 0", device.summary());
        //     // console.log("row 1", device.summary({ matrixRowIndex: 1 }));
        // })

        it("set, get", () => {
            const m = device.createMap({ mapId: 0 });
            console.log("set")
            console.log(m.set("a", 10));
            console.log(m.set("b", false));
            console.log(m.set("c", [20]));
            console.log(m.set("d", "hello"));
            // console.log(m.set("e", { message: "hello" }));
            console.log(m.set("f", null));

            console.log("get")
            console.log(m.get("a"));
            console.log(m.get("b"));
            console.log(m.get("c"));
            console.log(m.get("d"));
            // console.log(m.get("e"));
            console.log(m.get("f"));

            let keys = m.keys();
            console.log(keys)
            expect(keys).toEqual(expectedMapKeys1)
            // TODO assert values
        });

        it("not typing", () => {
            const m = device.createMap({ mapId: 1 });
            m.set("a", 10);
            m.set("b", [20]);
            m.set("c", false);
            m.set("d", "hello");
            // m.set("e", { propertyA: "a", propertyB: "b" });

            console.log("a", m.get("a"))
            console.log("b", m.get("b"))
            console.log("c", m.get("c"))
            console.log("d", m.get("d"))
            // console.log("e", m.get("e"))
            console.log("not_exist", m.get("not_exist"))

            let keys = m.keys();
            console.log(keys)
            expect(keys).toEqual(expectedMapKeys1)
            // TODO assert values
        });

        // TODO typing

        it("map config empty", () => {
            const invalid_device = new Device({})

            try {
                invalid_device.createMap({mapId: 0})
                unreachable()
            } catch (e) {
                console.log(e)
                let error = e as Error;
                expect(error.message).toEqual("map src empty")
            }
        });

        // Error: Memory access fault by GPU node ...
        // it("invalid map id - fatal error", () => {
        //     const invalid_device = new Device({
        //         map: {
        //             blocks: [
        //                 {
        //                     valueLen: 256,
        //                     capacity: 32
        //                 }
        //             ]
        //         }
        //     })
        //     // invalid_device.printMapSrc();
        //
        //     let map = invalid_device.createMap({mapId: 10})
        //
        //     // fatal error
        //     map.set("error fatal", 10);
        // });

    })
});