import { assert } from "chai";
import { mul } from "../src";

describe("index", () => {
    it("mul", () => {
        const result = mul(2, 3);
        assert.equal(result, 6);
    });
});