#!/usr/bin/env node
/**
 * Lightweight STDIO→HTTPS bridge for Databricks MCP over StreamableHTTP.
 * Replaces mcp-remote to avoid OAuth discovery overhead and stdin-EOF shutdown race.
 *
 * Usage: node databricks-mcp-proxy.js
 * Env:   DATABRICKS_TOKEN  (required)
 *        DATABRICKS_HOST   (optional, defaults to dbc-8dbfaaf3-7252.cloud.databricks.com)
 *        DATABRICKS_PATH   (optional, defaults to /api/2.0/mcp/functions/main/default)
 */

const https = require('https');
const URL = require('url').URL;

const HOST   = process.env.DATABRICKS_HOST  || 'dbc-8dbfaaf3-7252.cloud.databricks.com';
const PATH   = process.env.DATABRICKS_PATH  || '/api/2.0/mcp/functions/main/default';
const TOKEN  = process.env.DATABRICKS_TOKEN;

if (!TOKEN) {
  process.stderr.write('Error: DATABRICKS_TOKEN environment variable is required\n');
  process.exit(1);
}

function postToServer(message) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(message);
    const req = https.request({
      hostname: HOST,
      port: 443,
      path: PATH,
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${TOKEN}`,
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream',
        'Content-Length': Buffer.byteLength(body),
      },
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        // 202 = notification accepted, no body to parse
        if (res.statusCode === 202) { resolve(null); return; }
        try { resolve(JSON.parse(data)); }
        catch (e) { reject(new Error(`JSON parse error (status ${res.statusCode}): ${e.message}\nBody: ${data}`)); }
      });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

let buffer = '';
process.stdin.setEncoding('utf8');

process.stdin.on('data', (chunk) => {
  buffer += chunk;
  const lines = buffer.split('\n');
  buffer = lines.pop(); // last (possibly incomplete) line

  for (const line of lines) {
    if (!line.trim()) continue;
    let message;
    try { message = JSON.parse(line); }
    catch (e) { process.stderr.write(`[databricks-mcp-proxy] JSON parse error: ${e.message}\n`); continue; }

    postToServer(message)
      .then((response) => {
        if (response !== null) {
          process.stdout.write(JSON.stringify(response) + '\n');
        }
      })
      .catch((err) => {
        process.stderr.write(`[databricks-mcp-proxy] Request error: ${err.message}\n`);
        // Send JSON-RPC error back to client if we know the request id
        if (message && message.id !== undefined) {
          const errResp = { jsonrpc: '2.0', id: message.id, error: { code: -32603, message: err.message } };
          process.stdout.write(JSON.stringify(errResp) + '\n');
        }
      });
  }
});

process.stdin.on('end', () => { process.exit(0); });
process.stdin.on('error', () => { process.exit(1); });
