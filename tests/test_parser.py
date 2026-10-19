"""Tests for the response parser and request classifier."""

from __future__ import annotations

import json

from ollama_log_proxy.parser import (
    classify_request,
    extract_model_from_body,
    extract_usage_from_response,
)


class TestClassifyRequest:
    def test_inference_endpoints(self):
        assert classify_request("/api/generate") == "inference"
        assert classify_request("/api/chat") == "inference"
        assert classify_request("/api/embeddings") == "inference"
        assert classify_request("/api/embed") == "inference"
        assert classify_request("/v1/chat/completions") == "inference"
        assert classify_request("/v1/completions") == "inference"
        assert classify_request("/v1/embeddings") == "inference"
        assert classify_request("/v1/responses") == "inference"

    def test_inference_endpoint_trailing_slash(self):
        assert classify_request("/v1/chat/completions/") == "inference"
        assert classify_request("/api/chat/") == "inference"

    def test_poll_endpoints(self):
        assert classify_request("/api/tags") == "poll"
        assert classify_request("/api/ps") == "poll"
        assert classify_request("/api/version") == "poll"
        assert classify_request("/api/show") == "poll"

    def test_other_endpoints(self):
        assert classify_request("/api/pull") == "other"
        assert classify_request("/api/push") == "other"
        assert classify_request("/") == "other"
        assert classify_request("/health") == "other"


class TestExtractModel:
    def test_extracts_model(self):
        body = json.dumps({"model": "llama3.2", "prompt": "hello"}).encode()
        assert extract_model_from_body(body) == "llama3.2"

    def test_missing_model(self):
        body = json.dumps({"prompt": "hello"}).encode()
        assert extract_model_from_body(body) == ""

    def test_empty_body(self):
        assert extract_model_from_body(b"") == ""

    def test_invalid_json(self):
        assert extract_model_from_body(b"not json") == ""


class TestExtractUsage:
    def test_generate_response(self, generate_response_bytes):
        prompt, completion, total = extract_usage_from_response(
            generate_response_bytes, "/api/generate"
        )
        assert prompt == 26
        assert completion == 42
        assert total == 68

    def test_chat_response(self, chat_response_bytes):
        prompt, completion, total = extract_usage_from_response(chat_response_bytes, "/api/chat")
        assert prompt == 18
        assert completion == 35
        assert total == 53

    def test_streaming_response(self, streaming_response_bytes):
        prompt, completion, total = extract_usage_from_response(
            streaming_response_bytes, "/api/chat"
        )
        assert prompt == 12
        assert completion == 28
        assert total == 40

    def test_embeddings_response(self, embeddings_response_bytes):
        prompt, completion, total = extract_usage_from_response(
            embeddings_response_bytes, "/api/embeddings"
        )
        assert prompt == 8
        assert completion == 0
        assert total == 8

    def test_empty_response(self):
        prompt, completion, total = extract_usage_from_response(b"", "/api/chat")
        assert prompt == 0
        assert completion == 0
        assert total == 0

    def test_invalid_json_lines(self):
        data = b"not json\nalso not json\n"
        prompt, completion, total = extract_usage_from_response(data, "/api/chat")
        assert prompt == 0
        assert completion == 0
        assert total == 0

    def test_usage_field_override(self):
        data = json.dumps(
            {
                "prompt_eval_count": 10,
                "eval_count": 20,
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 25,
                    "total_tokens": 40,
                },
            }
        ).encode()
        prompt, completion, total = extract_usage_from_response(data, "/api/chat")
        assert prompt == 15
        assert completion == 25
        assert total == 40

    def test_openai_sse_usage(self):
        data = (
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":15,"completion_tokens":25,"total_tokens":40}}\n\n'
            b"data: [DONE]\n\n"
        )
        prompt, completion, total = extract_usage_from_response(
            data, "/v1/chat/completions"
        )
        assert prompt == 15
        assert completion == 25
        assert total == 40

    def test_openai_sse_without_usage(self):
        data = (
            b'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        prompt, completion, total = extract_usage_from_response(
            data, "/v1/chat/completions"
        )
        assert prompt == 0
        assert completion == 0
        assert total == 0

    def test_openai_sse_computes_missing_total(self):
        data = (
            b'data: {"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n'
            b"data: [DONE]\n\n"
        )
        prompt, completion, total = extract_usage_from_response(
            data, "/v1/chat/completions"
        )
        assert prompt == 7
        assert completion == 3
        assert total == 10

    def test_openai_responses_usage(self):
        data = json.dumps(
            {
                "object": "response",
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 22,
                    "total_tokens": 33,
                },
            }
        ).encode()
        prompt, completion, total = extract_usage_from_response(data, "/v1/responses")
        assert prompt == 11
        assert completion == 22
        assert total == 33
