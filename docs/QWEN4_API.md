# Local API

Start the resident loopback service:

```bash
./qwen4.sh --server 8080 --no-thinking
```

Use `http://127.0.0.1:8080/v1` as the OpenAI base URL and
`qwen3.8-flash-next-in-c` as the model. No API key is required; clients that
require one may use any non-empty placeholder.

The resident service exposes:

- `GET /health` for readiness checks;
- `GET /v1/models` for client discovery;
- `POST /v1/chat/completions` for chat and function tools;
- `OPTIONS` with CORS headers for browser clients.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.8-flash-next-in-c",
    "messages":[{"role":"user","content":"Where are the boundaries of technology?"}],
    "stream":true
  }'
```

The server emits UTF-8-safe content chunks, a final finish reason and usage,
then `data: [DONE]`. During long prefill or buffered tool transactions it sends
SSE keep-alive comments every ten seconds.

Function tools use OpenAI Chat Completions fields. The model may return several
calls in one response. Execute them in the application, then send back the
assistant `tool_calls` message and one matching `tool` message per result. The
runtime proposes calls but never executes external functions itself.

When a later request starts with the same tokenized conversation as the
previous request, the server restores the exact recurrent and attention state
at that boundary and evaluates only the appended messages. The cache is
accepted only after an exact token-prefix comparison. A miss resets and
evaluates the complete prompt, so unrelated requests cannot inherit state.

Current limits:

- one request at a time;
- loopback-only HTTP, without authentication or TLS;
- text messages and function tools;
- `tool_choice` supports `auto` and `none`;
- context defaults to 8,192 and may be configured up to the model's 262,144
  token limit when enough memory is available;
- request bodies are bounded by the shared native HTTP parser.

The normal launcher uses passive OpenMP waiting for resident services, keeping
idle CPU use low. The reproducible benchmark script explicitly selects active
waiting during its measured request; active waiting should not be left on for
an idle laptop service.
