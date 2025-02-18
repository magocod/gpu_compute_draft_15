import {sum} from "../index";
import TTLCache from "@isaacs/ttlcache";

function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

describe("ttl_cache", () => {
    describe("js" , function () {
        it("example 1", async () => {
            const cache = new TTLCache({ max: 10000, ttl: 1000 })

            // set some value
            cache.set(1, 2)

            // 999 ms later
            const h = cache.has(1) // returns true
            const v = cache.get(1);  // returns 2

            console.log(h, v)

            expect(h).toEqual(true);
            expect(v).toEqual(2);

            await sleep(1000);

            // 1000 ms later
            let h2 = cache.has(1) // returns false
            let v2 = cache.get(1) // returns undefined

            console.log(h2, v2)

            expect(h2).toEqual(false);
            expect(v2).toEqual(undefined);
        });
    })
});