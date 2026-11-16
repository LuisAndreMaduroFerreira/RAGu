const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

async function main() {
  const input = JSON.parse(fs.readFileSync(0, "utf8"));
  const state = input.state;
  const output = { state, nextSet: false, nextRequest: null, tests: [], error: null };
  const get = key => Object.hasOwn(state.local, key) ? state.local[key] : state.collection[key];
  const replaceIn = text => text.replace(/\{\{([^}]+)\}\}/g, (_, key) => {
    if (key.startsWith("vault:")) return state.vault[key.slice(6)] ?? "";
    if (key === "$timestamp") return String(Math.floor(Date.now() / 1000));
    const value = get(key);
    if (value === undefined) throw new Error(`Unknown collection variable: ${key}`);
    return String(value);
  });
  const pm = {
    variables: { get, set: (key, value) => { state.local[key] = value; }, replaceIn },
    collectionVariables: {
      get: key => state.collection[key],
      set: (key, value) => { state.collection[key] = value; }
    },
    vault: {
      get: async key => state.vault[key],
      set: async (key, value) => { state.vault[key] = value; }
    },
    info: { requestName: input.item.name },
    execution: {
      setNextRequest: value => { output.nextSet = true; output.nextRequest = value; }
    },
    response: input.response && {
      code: input.response.code,
      json: () => input.response.body
    },
    test: (name, fn) => { fn(); output.tests.push(name); },
    expect: actual => ({
      to: {
        equal: expected => assert.deepEqual(actual, expected),
        include: expected => assert.ok(actual.includes(expected)),
        be: {
          within: (low, high) => assert.ok(actual >= low && actual <= high),
          at: { least: minimum => assert.ok(actual >= minimum) }
        }
      }
    })
  };
  try {
    for (const owner of [input.collection, input.item]) {
      for (const event of owner.event ?? []) {
        if (event.listen !== input.stage) continue;
        const code = `(async () => {\n${event.script.exec.join("\n")}\n})()`;
        await new vm.Script(code).runInNewContext({ pm, Date, JSON, Number, String, Error },
                                                { timeout: 1000 });
      }
    }
    if (input.stage === "prerequest") {
      const request = input.item.request;
      output.request = {
        method: request.method,
        url: replaceIn(request.url),
        headers: Object.fromEntries((request.header ?? []).map(h => [h.key, replaceIn(h.value)]))
      };
      if (request.body) output.request.content = replaceIn(request.body.raw);
      if (request.auth?.type !== "noauth") {
        output.request.headers.Authorization =
          `Bearer ${replaceIn(input.collection.auth.bearer[0].value)}`;
      }
    }
  } catch (error) {
    output.error = error.message;
  }
  process.stdout.write(JSON.stringify(output));
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
