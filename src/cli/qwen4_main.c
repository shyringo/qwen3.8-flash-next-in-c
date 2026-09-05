#define _POSIX_C_SOURCE 200809L

#include "qwen4_model.h"
#include "qwen38_http.h"
#include "qwen38_sampler.h"
#include "qwen38_tokenizer.h"
#include "qwen38_tool.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <sys/socket.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include <unistd.h>

typedef struct {
    const char *model;
    const char *prompt;
    const char *system;
    uint32_t context;
    uint32_t memory_gib;
    uint32_t max_tokens;
    uint64_t seed;
    float temperature;
    int thinking;
    uint32_t server_port;
} Options;

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} Buffer;

typedef struct {
    Q4ModelState *state;
    uint32_t *tokens;
    uint32_t count;
    uint32_t capacity;
} PrefixCache;

static double now_seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

static int append(Buffer *buffer, const char *text, size_t length)
{
    if (length > SIZE_MAX - buffer->length - 1u) return 0;
    const size_t needed = buffer->length + length + 1u;
    if (needed > buffer->capacity) {
        size_t capacity = buffer->capacity ? buffer->capacity : 1024u;
        while (capacity < needed) capacity *= 2u;
        char *larger = (char *)realloc(buffer->data, capacity);
        if (!larger) return 0;
        buffer->data = larger;
        buffer->capacity = capacity;
    }
    memcpy(buffer->data + buffer->length, text, length);
    buffer->length += length;
    buffer->data[buffer->length] = '\0';
    return 1;
}

static int append_text(Buffer *buffer, const char *text)
{
    return append(buffer, text, strlen(text));
}

static int parse_u32(const char *text, uint32_t *value)
{
    char *end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(text, &end, 10);
    if (errno || !end || *end || parsed > UINT32_MAX) return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static int parse_options(int argc, char **argv, Options *options)
{
    memset(options, 0, sizeof(*options));
    options->context = 8192u;
    options->max_tokens = 256u;
    options->seed = 1u;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            options->model = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc)
            options->prompt = argv[++i];
        else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc)
            options->system = argv[++i];
        else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &options->context)) return 0;
        } else if (strcmp(argv[i], "--memory-gib") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &options->memory_gib) ||
                !options->memory_gib) return 0;
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &options->max_tokens)) return 0;
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            char *end = NULL;
            options->temperature = strtof(argv[++i], &end);
            if (!end || *end || options->temperature < 0.0f) return 0;
        } else if (strcmp(argv[i], "--thinking") == 0) options->thinking = 1;
        else if (strcmp(argv[i], "--no-thinking") == 0) options->thinking = 0;
        else if (strcmp(argv[i], "--server") == 0 && i + 1 < argc) {
            if (!parse_u32(argv[++i], &options->server_port) ||
                options->server_port > 65535u) return 0;
        }
        else if (strcmp(argv[i], "--help") == 0) return 0;
        else return 0;
    }
    return options->model && options->context >= 64u && options->max_tokens;
}

static void usage(const char *program)
{
    fprintf(stderr,
        "usage: %s --model MODEL-00001-of-N.gguf [--prompt TEXT] [options]\n"
        "  --prompt TEXT       run one request; omit for interactive chat\n"
        "  --system TEXT       system instruction\n"
        "  --context N         context capacity (default: 8192)\n"
        "  --memory-gib N      available RAM budget (auto-detected)\n"
        "  --max-tokens N      output limit (default: 256)\n"
        "  --temperature N     0 for greedy (default: 0)\n"
        "  --thinking          enable visible reasoning\n"
        "  --no-thinking       answer directly (default)\n"
        "  --server PORT       loopback OpenAI-compatible API\n", program);
}

static uint32_t physical_memory_gib(void)
{
#if defined(__APPLE__)
    uint64_t bytes = 0u;
    size_t size = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &size, NULL, 0) != 0 ||
        size != sizeof(bytes)) return 0u;
#elif defined(_SC_PHYS_PAGES)
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0) return 0u;
    const uint64_t bytes = (uint64_t)pages * (uint64_t)page_size;
#else
    return 0u;
#endif
    const uint64_t gib = UINT64_C(1) << 30;
    return bytes > UINT32_MAX * gib
         ? UINT32_MAX : (uint32_t)((bytes + gib - 1u) / gib);
}

static void configure_memory_policy(const Options *options)
{
    const uint32_t budget = options->memory_gib
                          ? options->memory_gib : physical_memory_gib();
    if (budget && budget <= 10u && !getenv("Q4_LOW_MEMORY")) {
        (void)setenv("Q4_LOW_MEMORY", "1", 0);
        fprintf(stderr,
                "qwen4: low-memory weight streaming enabled (%u GiB budget)\n",
                budget);
    }
}

static int render_prompt(Buffer *prompt, const Buffer *history,
                         const char *user, const Options *options)
{
    prompt->length = 0u;
    if (prompt->data) prompt->data[0] = '\0';
    if (options->system && *options->system) {
        if (!append_text(prompt, "<|im_start|>system\n") ||
            !append_text(prompt, options->system) ||
            !append_text(prompt, "<|im_end|>\n")) return 0;
    }
    if (history->length && !append(prompt, history->data, history->length)) return 0;
    return append_text(prompt, "<|im_start|>user\n") &&
           append_text(prompt, user) &&
           append_text(prompt, "<|im_end|>\n<|im_start|>assistant\n") &&
           append_text(prompt, options->thinking
                ? "<think>\n" : "<think>\n\n</think>\n\n");
}

static int generate(Q4Model *model, Q38Tokenizer *tokenizer,
                    const Options *options, const char *prompt,
                    Buffer *visible, int echo, PrefixCache *cache,
                    int *prompt_tokens, uint32_t *output_tokens,
                    int (*emit)(void *, const char *, size_t), void *emit_context)
{
    uint32_t *tokens = (uint32_t *)malloc(
        (size_t)options->context * sizeof(*tokens));
    uint8_t *presence = (uint8_t *)calloc(q4_model_vocab_size(model), 1u);
    if (!tokens || !presence) { free(tokens); free(presence); return 0; }
    const int count = q38_tokenizer_encode(tokenizer, prompt, strlen(prompt),
                                            tokens, options->context);
    if (count <= 0 || (uint32_t)count >= options->context) {
        fprintf(stderr, "qwen4: prompt is too long for --context\n");
        free(tokens); free(presence); return 0;
    }
    Q38Sampler sampler;
    q38_sampler_init(&sampler, options->seed);
    sampler.temperature = options->temperature;
    sampler.top_k = 20u;
    sampler.top_p = 0.95f;
    sampler.presence = presence;
    sampler.presence_size = q4_model_vocab_size(model);
    for (int i = 0; i < count; ++i) q38_sampler_observe(&sampler, tokens[i]);
    const double started = now_seconds();
    const float *logits = NULL;
    uint32_t cached = 0u;
    if (cache && cache->count && cache->count <= (uint32_t)count &&
        memcmp(cache->tokens, tokens,
               (size_t)cache->count * sizeof(*tokens)) == 0 &&
        q4_model_state_restore(model, cache->state, &logits))
        cached = cache->count;
    else
        q4_model_reset(model);
    if (cached < (uint32_t)count &&
        !q4_model_prefill(model, tokens + cached,
                          (uint32_t)count - cached, &logits)) {
        free(tokens); free(presence); return 0;
    }
    if (cache) {
        if (q4_model_state_save(model, cache->state)) {
            memcpy(cache->tokens, tokens, (size_t)count * sizeof(*tokens));
            cache->count = (uint32_t)count;
        } else {
            cache->count = 0u;
        }
    }
    const double first_ready = now_seconds();
    uint32_t generated = 0;
    const uint32_t eos = q38_tokenizer_eos(tokenizer);
    const int im_end = q38_tokenizer_find(tokenizer, "<|im_end|>");
    visible->length = 0u;
    if (visible->data) visible->data[0] = '\0';
    while (generated < options->max_tokens &&
           q4_model_position(model) < q4_model_context_length(model)) {
        uint32_t token = 0;
        if (!q38_sample(&sampler, logits, q4_model_vocab_size(model), &token)) break;
        if (token == eos || (im_end >= 0 && token == (uint32_t)im_end)) break;
        q38_sampler_observe(&sampler, token);
        char piece[1024];
        const int length = q38_tokenizer_decode_token(
            tokenizer, token, piece, sizeof(piece));
        if (length < 0) break;
        if (!q38_tokenizer_is_special(tokenizer, token)) {
            if (echo) {
                fwrite(piece, 1, (size_t)length, stdout);
                fflush(stdout);
            }
            if (!append(visible, piece, (size_t)length)) break;
            if (emit && !emit(emit_context, piece, (size_t)length)) break;
        }
        ++generated;
        if (generated >= options->max_tokens) break;
        if (!q4_model_forward_token(model, token, &logits)) break;
    }
    if (echo) putchar('\n');
    const double ended = now_seconds();
    fprintf(stderr, "[prompt=%d tokens, cached=%u, TTFT=%.3fs", count,
            cached, first_ready - started);
    if (generated > 1u)
        fprintf(stderr, ", output=%u tokens, TPOT=%.3fs", generated,
                (ended - first_ready) / (generated - 1u));
    fprintf(stderr, "]\n");
    if (prompt_tokens) *prompt_tokens = count;
    if (output_tokens) *output_tokens = generated;
    free(tokens); free(presence);
    return 1;
}

static int send_all(int socket, const char *data, size_t length)
{
    while (length) {
        const ssize_t sent = send(socket, data, length, 0);
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) return 0;
        data += sent;
        length -= (size_t)sent;
    }
    return 1;
}

static int http_response(int client, int status, const char *reason,
                         const char *body, size_t body_length)
{
    char header[512];
    const int length = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\nContent-Type: application/json; charset=utf-8\r\n"
        "Content-Length: %zu\r\nAccess-Control-Allow-Origin: *\r\n"
        "Connection: close\r\n\r\n", status, reason, body_length);
    return length > 0 && (size_t)length < sizeof(header) &&
           send_all(client, header, (size_t)length) &&
           send_all(client, body, body_length);
}

static int error_response(int client, int status, const char *message)
{
    Q38Buffer body;
    q38_buffer_init(&body);
    int ok = q38_buffer_append_string(&body, "{\"error\":{\"message\":") &&
             q38_buffer_append_json_string(&body, message, strlen(message)) &&
             q38_buffer_append_string(&body, ",\"type\":\"invalid_request_error\"}}");
    const char *reason = status == 400 ? "Bad Request" :
                         status == 404 ? "Not Found" :
                         "Internal Server Error";
    if (ok) ok = http_response(client, status, reason,
                               body.data, body.length);
    q38_buffer_free(&body);
    return ok;
}

typedef struct {
    int client;
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int stop;
    int failed;
} Heartbeat;

static void *heartbeat_main(void *context)
{
    Heartbeat *heartbeat = (Heartbeat *)context;
    static const char event[] = ": keep-alive\n\n";
    pthread_mutex_lock(&heartbeat->mutex);
    while (!heartbeat->stop) {
        struct timespec deadline;
        clock_gettime(CLOCK_REALTIME, &deadline);
        deadline.tv_sec += 10;
        const int waited = pthread_cond_timedwait(
            &heartbeat->condition, &heartbeat->mutex, &deadline);
        if (heartbeat->stop) break;
        if (waited != ETIMEDOUT) continue;
        const int ok = send_all(heartbeat->client, event, sizeof(event) - 1u);
        if (!ok) { heartbeat->failed = 1; heartbeat->stop = 1; }
    }
    pthread_mutex_unlock(&heartbeat->mutex);
    return NULL;
}

static int heartbeat_start(Heartbeat *heartbeat, int client)
{
    memset(heartbeat, 0, sizeof(*heartbeat));
    heartbeat->client = client;
    if (pthread_mutex_init(&heartbeat->mutex, NULL) != 0) return 0;
    if (pthread_cond_init(&heartbeat->condition, NULL) != 0) {
        pthread_mutex_destroy(&heartbeat->mutex); return 0;
    }
    if (pthread_create(&heartbeat->thread, NULL,
                       heartbeat_main, heartbeat) != 0) {
        pthread_cond_destroy(&heartbeat->condition);
        pthread_mutex_destroy(&heartbeat->mutex); return 0;
    }
    return 1;
}

static int heartbeat_finish(Heartbeat *heartbeat)
{
    pthread_mutex_lock(&heartbeat->mutex);
    heartbeat->stop = 1;
    pthread_cond_signal(&heartbeat->condition);
    pthread_mutex_unlock(&heartbeat->mutex);
    pthread_join(heartbeat->thread, NULL);
    pthread_cond_destroy(&heartbeat->condition);
    pthread_mutex_destroy(&heartbeat->mutex);
    return !heartbeat->failed;
}

static int stream_headers(int client)
{
    static const char headers[] =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
        "data: {\"object\":\"chat.completion.chunk\",\"model\":"
        "\"qwen3.8-flash-next-in-c\",\"choices\":[{\"index\":0,"
        "\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
        "\"finish_reason\":null}]}\n\n";
    return send_all(client, headers, sizeof(headers) - 1u);
}

static int stream_event(int client, const char *json)
{
    return send_all(client, "data: ", 6u) &&
           send_all(client, json, strlen(json)) && send_all(client, "\n\n", 2u);
}

static int stream_content_emit(void *context, const char *text, size_t length)
{
    Heartbeat *heartbeat = (Heartbeat *)context;
    Q38Buffer event;
    q38_buffer_init(&event);
    int ok = q38_buffer_append_string(&event,
        "{\"object\":\"chat.completion.chunk\",\"model\":"
        "\"qwen3.8-flash-next-in-c\",\"choices\":[{\"index\":0,"
        "\"delta\":{\"content\":") &&
        q38_buffer_append_json_string(&event, text, length) &&
        q38_buffer_append_string(&event, "},\"finish_reason\":null}]}");
    pthread_mutex_lock(&heartbeat->mutex);
    if (ok && !heartbeat->failed) ok = stream_event(heartbeat->client, event.data);
    if (!ok) heartbeat->failed = 1;
    pthread_mutex_unlock(&heartbeat->mutex);
    q38_buffer_free(&event);
    return ok;
}

static int read_request(int client, char **body, size_t *body_length,
                        char *method, size_t method_capacity,
                        char *path, size_t path_capacity)
{
    char header[65537];
    size_t used = 0u;
    char *end = NULL;
    while (!end && used < sizeof(header) - 1u) {
        const ssize_t got = recv(client, header + used,
                                 sizeof(header) - 1u - used, 0);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return 0;
        used += (size_t)got;
        header[used] = '\0';
        end = strstr(header, "\r\n\r\n");
    }
    if (!end) return 0;
    if (method_capacity < 16u || path_capacity < 256u ||
        sscanf(header, "%15s %255s", method, path) != 2) return 0;
    size_t content_length = 0u;
    for (char *line = strstr(header, "\r\n"); line && line < end; ) {
        line += 2;
        if (strncasecmp(line, "Content-Length:", 15u) == 0)
            content_length = (size_t)strtoull(line + 15u, NULL, 10);
        line = strstr(line, "\r\n");
    }
    if (content_length > Q38_HTTP_MAX_BODY) return 0;
    if (!content_length) {
        *body = NULL;
        *body_length = 0u;
        return strcmp(method, "GET") == 0 || strcmp(method, "OPTIONS") == 0;
    }
    *body = (char *)malloc(content_length + 1u);
    if (!*body) return 0;
    const size_t offset = (size_t)(end + 4 - header);
    size_t copied = used > offset ? used - offset : 0u;
    if (copied > content_length) copied = content_length;
    memcpy(*body, header + offset, copied);
    while (copied < content_length) {
        const ssize_t got = recv(client, *body + copied,
                                 content_length - copied, 0);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) { free(*body); *body = NULL; return 0; }
        copied += (size_t)got;
    }
    (*body)[content_length] = '\0';
    *body_length = content_length;
    return 1;
}

static int options_response(int client)
{
    static const char response[] =
        "HTTP/1.1 204 No Content\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
        "Access-Control-Max-Age: 86400\r\n"
        "Content-Length: 0\r\nConnection: close\r\n\r\n";
    return send_all(client, response, sizeof(response) - 1u);
}

static int append_tool_calls(Q38Buffer *response, const Q38ToolResult *parsed,
                             const char *id)
{
    if (!parsed->call_count) return 1;
    if (!q38_buffer_append_string(response, ",\"tool_calls\":[")) return 0;
    for (size_t i = 0; i < parsed->call_count; ++i) {
        char prefix[192];
        const int length = snprintf(prefix, sizeof(prefix),
            "%s{\"id\":\"call-%s-%zu\",\"type\":\"function\","
            "\"function\":{\"name\":", i ? "," : "", id, i);
        if (length <= 0 || (size_t)length >= sizeof(prefix) ||
            !q38_buffer_append(response, prefix, (size_t)length) ||
            !q38_buffer_append_json_string(response, parsed->calls[i].name,
                                            strlen(parsed->calls[i].name)) ||
            !q38_buffer_append_string(response, ",\"arguments\":") ||
            !q38_buffer_append_json_string(response, parsed->calls[i].arguments,
                                            strlen(parsed->calls[i].arguments)) ||
            !q38_buffer_append_string(response, "}}")) return 0;
    }
    return q38_buffer_append_string(response, "]");
}

static int stream_finish_response(int client, const char *content,
                                  const Q38ToolResult *parsed,
                                  int prompt_tokens, uint32_t output_tokens)
{
    Q38Buffer event;
    q38_buffer_init(&event);
    int ok = 1;
    if (content && *content) {
        ok = q38_buffer_append_string(&event,
            "{\"object\":\"chat.completion.chunk\",\"model\":"
            "\"qwen3.8-flash-next-in-c\",\"choices\":[{\"index\":0,"
            "\"delta\":{\"content\":") &&
             q38_buffer_append_json_string(&event, content, strlen(content)) &&
             q38_buffer_append_string(&event, "},\"finish_reason\":null}]}") &&
             stream_event(client, event.data);
        q38_buffer_free(&event);
    }
    for (size_t i = 0; ok && parsed && i < parsed->call_count; ++i) {
        q38_buffer_init(&event);
        char prefix[256];
        const int n = snprintf(prefix, sizeof(prefix),
            "{\"object\":\"chat.completion.chunk\",\"model\":"
            "\"qwen3.8-flash-next-in-c\",\"choices\":[{\"index\":0,"
            "\"delta\":{\"tool_calls\":[{\"index\":%zu,"
            "\"id\":\"call-local-%zu\",\"type\":\"function\","
            "\"function\":{\"name\":", i, i);
        ok = n > 0 && (size_t)n < sizeof(prefix) &&
             q38_buffer_append(&event, prefix, (size_t)n) &&
             q38_buffer_append_json_string(&event, parsed->calls[i].name,
                                             strlen(parsed->calls[i].name)) &&
             q38_buffer_append_string(&event, ",\"arguments\":") &&
             q38_buffer_append_json_string(&event, parsed->calls[i].arguments,
                                             strlen(parsed->calls[i].arguments)) &&
             q38_buffer_append_string(&event,
                "}}]},\"finish_reason\":null}]}") &&
             stream_event(client, event.data);
        q38_buffer_free(&event);
    }
    if (!ok) return 0;
    char final[512];
    const int n = snprintf(final, sizeof(final),
        "{\"object\":\"chat.completion.chunk\",\"model\":"
        "\"qwen3.8-flash-next-in-c\",\"choices\":[{\"index\":0,"
        "\"delta\":{},\"finish_reason\":\"%s\"}],\"usage\":{"
        "\"prompt_tokens\":%d,\"completion_tokens\":%u,\"total_tokens\":%u}}",
        parsed && parsed->call_count ? "tool_calls" : "stop", prompt_tokens,
        output_tokens, (unsigned)(prompt_tokens + (int)output_tokens));
    return n > 0 && (size_t)n < sizeof(final) && stream_event(client, final) &&
           send_all(client, "data: [DONE]\n\n", 14u);
}

static int handle_chat(int client, Q4Model *model, Q38Tokenizer *tokenizer,
                       PrefixCache *cache, const Options *base,
                       const char *json, size_t length)
{
    Q38HttpChatRequest request;
    char error[256] = {0};
    if (!q38_http_parse_chat_request(json, length, &request,
                                     error, sizeof(error)))
        return error_response(client, 400, error);
    const int streaming = request.stream;
    if (strcmp(request.model, "qwen3.8-flash-next-in-c") != 0 &&
        strcmp(request.model, "qwen3.8-flash-next") != 0) {
        q38_http_chat_request_free(&request);
        return error_response(client, 400, "requested model is not available");
    }
    Options options = *base;
    if (request.has_max_tokens) options.max_tokens = request.max_tokens;
    if (request.has_temperature) options.temperature = request.temperature;
    if (request.has_seed) options.seed = request.seed;
    char *prompt = q38_http_render_messages(&request, options.system,
                                             options.thinking,
                                             error, sizeof(error));
    if (!prompt) {
        q38_http_chat_request_free(&request);
        return error_response(client, 400, error);
    }
    Heartbeat heartbeat;
    if (streaming && (!stream_headers(client) ||
                      !heartbeat_start(&heartbeat, client))) {
        free(prompt); q38_http_chat_request_free(&request); return 0;
    }
    Buffer answer = {0};
    int prompt_tokens = 0;
    uint32_t output_tokens = 0;
    const int tools = request.tool_count && !request.tool_choice_none;
    const int generated = generate(model, tokenizer, &options, prompt,
                                   &answer, 0, cache,
                                   &prompt_tokens, &output_tokens,
                                   streaming && !tools ? stream_content_emit : NULL,
                                   streaming && !tools ? &heartbeat : NULL);
    free(prompt);
    const int heartbeat_ok = !streaming || heartbeat_finish(&heartbeat);
    if (!generated) {
        q38_http_chat_request_free(&request);
        free(answer.data);
        return streaming ? 0 : error_response(client, 500, "model generation failed");
    }
    char id[96];
    snprintf(id, sizeof(id), "local-%lld", (long long)time(NULL));
    Q38ToolResult parsed;
    if (tools && !q38_tool_parse(answer.data ? answer.data : "", answer.length,
                                 request.tools, request.tool_count, &parsed,
                                 error, sizeof(error))) {
        q38_http_chat_request_free(&request);
        free(answer.data);
        return streaming ? 0 : error_response(client, 500, error);
    }
    if (streaming) {
        const char *content = tools ? parsed.content : NULL;
        const int sent = heartbeat_ok && stream_finish_response(
            client, content, tools ? &parsed : NULL,
            prompt_tokens, output_tokens);
        if (tools) q38_tool_result_free(&parsed);
        q38_http_chat_request_free(&request);
        free(answer.data);
        return sent;
    }
    Q38Buffer response;
    q38_buffer_init(&response);
    int ok = q38_buffer_append_string(&response,
        "{\"id\":\"chatcmpl-") && q38_buffer_append_string(&response, id) &&
        q38_buffer_append_string(&response,
        "\",\"object\":\"chat.completion\",\"model\":\"qwen3.8-flash-next-in-c\","
        "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    const char *content = tools ? parsed.content : (answer.data ? answer.data : "");
    if (ok && content && *content)
        ok = q38_buffer_append_json_string(&response, content, strlen(content));
    else if (ok) ok = q38_buffer_append_string(&response, "null");
    if (ok && tools) ok = append_tool_calls(&response, &parsed, id);
    char suffix[256];
    const int suffix_length = snprintf(suffix, sizeof(suffix),
        "},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,"
        "\"completion_tokens\":%u,\"total_tokens\":%u}}",
        tools && parsed.call_count ? "tool_calls" : "stop", prompt_tokens,
        output_tokens, (unsigned)(prompt_tokens + (int)output_tokens));
    if (ok) ok = suffix_length > 0 && (size_t)suffix_length < sizeof(suffix) &&
                 q38_buffer_append(&response, suffix, (size_t)suffix_length);
    if (ok) ok = http_response(client, 200, "OK", response.data, response.length);
    q38_buffer_free(&response);
    if (tools) q38_tool_result_free(&parsed);
    q38_http_chat_request_free(&request);
    free(answer.data);
    return ok;
}

static int serve(Q4Model *model, Q38Tokenizer *tokenizer,
                 PrefixCache *cache, const Options *options)
{
    signal(SIGPIPE, SIG_IGN);
    const int server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) return 0;
    int reuse = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    if (inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) != 1) {
        close(server);
        return 0;
    }
    address.sin_port = htons((uint16_t)options->server_port);
    if (bind(server, (struct sockaddr *)&address, sizeof(address)) != 0 ||
        listen(server, 8) != 0) { close(server); return 0; }
    fprintf(stderr, "qwen4: listening on http://127.0.0.1:%u\n",
            options->server_port);
    for (;;) {
        const int client = accept(server, NULL, NULL);
        if (client < 0 && errno == EINTR) continue;
        if (client < 0) break;
        char *body = NULL;
        size_t body_length = 0;
        char method[16] = {0};
        char path[256] = {0};
        if (!read_request(client, &body, &body_length,
                          method, sizeof(method), path, sizeof(path)))
            (void)error_response(client, 400, "invalid HTTP request");
        else if (strcmp(method, "OPTIONS") == 0)
            (void)options_response(client);
        else if (strcmp(method, "GET") == 0 &&
                 strcmp(path, "/health") == 0)
            (void)http_response(client, 200, "OK",
                "{\"status\":\"ok\"}", sizeof("{\"status\":\"ok\"}") - 1u);
        else if (strcmp(method, "GET") == 0 &&
                 strcmp(path, "/v1/models") == 0) {
            static const char models[] =
                "{\"object\":\"list\",\"data\":[{\"id\":"
                "\"qwen3.8-flash-next-in-c\",\"object\":\"model\","
                "\"owned_by\":\"local\"}]}";
            (void)http_response(client, 200, "OK", models,
                                sizeof(models) - 1u);
        } else if (strcmp(method, "POST") == 0 &&
                   strcmp(path, "/v1/chat/completions") == 0)
            (void)handle_chat(client, model, tokenizer, cache, options,
                              body, body_length);
        else
            (void)error_response(client, 404, "unknown endpoint");
        free(body);
        close(client);
    }
    close(server);
    return 1;
}

int main(int argc, char **argv)
{
    Options options;
    if (!parse_options(argc, argv, &options)) {
        usage(argv[0]);
        return argc > 1 && strcmp(argv[1], "--help") == 0 ? 0 : 2;
    }
    configure_memory_policy(&options);
    Q4Model *model = q4_model_open_gguf(options.model, options.context);
    Q38Tokenizer *tokenizer = q38_tokenizer_open_gguf(options.model);
    if (!model || !tokenizer) {
        q4_model_close(model); q38_tokenizer_close(tokenizer); return 1;
    }
    Buffer prompt = {0}, history = {0}, answer = {0};
    PrefixCache cache = {0};
    if (!options.prompt) {
        cache.state = q4_model_state_create();
        cache.tokens = (uint32_t *)malloc(
            (size_t)options.context * sizeof(*cache.tokens));
        cache.capacity = options.context;
        if (!cache.state || !cache.tokens) {
            q4_model_state_destroy(cache.state);
            free(cache.tokens);
            memset(&cache, 0, sizeof(cache));
            fprintf(stderr, "qwen4: prefix cache disabled (not enough memory)\n");
        }
    }
    int ok = 1;
    if (options.server_port) {
        ok = serve(model, tokenizer, &cache, &options);
    } else if (options.prompt) {
        ok = render_prompt(&prompt, &history, options.prompt, &options) &&
             generate(model, tokenizer, &options, prompt.data, &answer, 1, NULL,
                      NULL, NULL, NULL, NULL);
    } else {
        char line[65536];
        for (;;) {
            fputs("you> ", stdout); fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) break;
            line[strcspn(line, "\r\n")] = '\0';
            if (strcmp(line, "/exit") == 0) break;
            if (strcmp(line, "/reset") == 0) {
                history.length = 0u; if (history.data) history.data[0] = '\0';
                continue;
            }
            if (!*line) continue;
            fputs("assistant> ", stdout); fflush(stdout);
            if (!render_prompt(&prompt, &history, line, &options) ||
                !generate(model, tokenizer, &options, prompt.data, &answer, 1,
                          &cache,
                          NULL, NULL, NULL, NULL)) {
                ok = 0; break;
            }
            ok = append_text(&history, "<|im_start|>user\n") &&
                 append_text(&history, line) &&
                 append_text(&history, "<|im_end|>\n<|im_start|>assistant\n") &&
                 append(&history, answer.data ? answer.data : "", answer.length) &&
                 append_text(&history, "<|im_end|>\n");
            if (!ok) break;
        }
    }
    q4_model_state_destroy(cache.state);
    free(cache.tokens);
    free(prompt.data); free(history.data); free(answer.data);
    q38_tokenizer_close(tokenizer); q4_model_close(model);
    return ok ? 0 : 1;
}
