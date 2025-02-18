import {sum} from "../index";

describe("index", () => {
    it("sum", () => {
        expect(sum(1, 2)).toEqual(3);
    });
});