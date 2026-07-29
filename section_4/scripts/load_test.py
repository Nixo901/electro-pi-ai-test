import asyncio
import time
import argparse
import sys
import json
import httpx
from typing import List, Dict, Any

# Windows console encoding fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEST_PROMPT = "Explain the difference between a process and a thread in one paragraph."

async def send_single_request(
    client: httpx.AsyncClient, 
    url: str, 
    prompt: str, 
    max_tokens: int,
    req_id: int
) -> Dict[str, Any]:
    """Sends a single POST generation request and measures client-side metrics."""
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False
    }
    
    t_start = time.perf_counter()
    try:
        response = await client.post(f"{url}/generate", json=payload)
        t_end = time.perf_counter()
        
        client_latency_ms = (t_end - t_start) * 1000.0
        
        if response.status_code == 200:
            data = response.json()
            return {
                "id": req_id,
                "status": "success",
                "client_latency_ms": client_latency_ms,
                "server_total_latency_ms": data.get("total_latency_ms", 0.0),
                "server_ttft_ms": data.get("time_to_first_token_ms", 0.0),
                "tokens_generated": data.get("tokens_generated", 0),
                "tokens_per_sec": data.get("tokens_per_sec", 0.0)
            }
        else:
            return {
                "id": req_id,
                "status": f"error_status_{response.status_code}",
                "client_latency_ms": client_latency_ms,
                "error": response.text
            }
    except Exception as e:
        t_end = time.perf_counter()
        return {
            "id": req_id,
            "status": "exception",
            "client_latency_ms": (t_end - t_start) * 1000.0,
            "error": str(e)
        }

async def test_streaming_endpoint(client: httpx.AsyncClient, url: str) -> None:
    """Verifies that streaming works and measures time to first token client-side."""
    print("\n" + "=" * 60)
    print("Testing Streaming Response Endpoint (/generate/stream)...")
    print("=" * 60)
    
    payload = {
        "prompt": "Tell a very short 1-sentence joke.",
        "max_new_tokens": 50,
        "temperature": 0.7,
        "stream": True
    }
    
    t_start = time.perf_counter()
    first_token_time = None
    tokens_received = 0
    full_text = []
    
    try:
        async with client.stream("POST", f"{url}/generate/stream", json=payload) as response:
            if response.status_code != 200:
                print(f"Error: Streaming endpoint returned status code {response.status_code}")
                return
                
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        token = data.get("token", "")
                        if token:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            tokens_received += 1
                            full_text.append(token)
                            print(token, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        
        t_end = time.perf_counter()
        total_time_ms = (t_end - t_start) * 1000.0
        
        if first_token_time:
            ttft_ms = (first_token_time - t_start) * 1000.0
            print(f"\n\nStreaming Metrics:")
            print(f"  - Client-perceived TTFT:    {ttft_ms:.2f} ms")
            print(f"  - Client-perceived Latency: {total_time_ms:.2f} ms")
            print(f"  - Chunks/Tokens Streamed:   {tokens_received}")
        else:
            print(f"\nNo tokens streamed successfully.")
            
    except Exception as e:
        print(f"\nStreaming failed with exception: {e}")

def calculate_percentile(sorted_list: List[float], percentile: float) -> float:
    """Calculates percentile from a sorted list of numbers."""
    if not sorted_list:
        return 0.0
    index = (len(sorted_list) - 1) * percentile
    lower = int(index)
    upper = lower + 1
    weight = index - lower
    if upper < len(sorted_list):
        return sorted_list[lower] * (1 - weight) + sorted_list[upper] * weight
    return sorted_list[lower]

def print_results_table(results: List[Dict[str, Any]], total_wall_time: float) -> None:
    """Prints a structured table summarizing the concurrent load test results."""
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] != "success"]
    
    print("\n" + "=" * 60)
    print(f"Load Test Summary (Concurrency: {len(results)})")
    print("=" * 60)
    print(f"Total Requests: {len(results)}")
    print(f"Successes:      {len(successes)}")
    print(f"Failures:       {len(failures)}")
    print(f"Total Wall Time:{total_wall_time:.2f} seconds\n")
    
    if not successes:
        print("No successful requests to analyze.")
        return
        
    client_latencies = sorted([r["client_latency_ms"] for r in successes])
    server_ttfts = sorted([r["server_ttft_ms"] for r in successes])
    server_latencies = sorted([r["server_total_latency_ms"] for r in successes])
    total_tokens = sum([r["tokens_generated"] for r in successes])
    
    avg_client = sum(client_latencies) / len(client_latencies)
    avg_ttft = sum(server_ttfts) / len(server_ttfts)
    avg_server = sum(server_latencies) / len(server_latencies)
    
    p50_client = calculate_percentile(client_latencies, 0.50)
    p95_client = calculate_percentile(client_latencies, 0.95)
    p99_client = calculate_percentile(client_latencies, 0.99)
    
    p50_ttft = calculate_percentile(server_ttfts, 0.50)
    p95_ttft = calculate_percentile(server_ttfts, 0.95)
    p99_ttft = calculate_percentile(server_ttfts, 0.99)
    
    p50_server = calculate_percentile(server_latencies, 0.50)
    p95_server = calculate_percentile(server_latencies, 0.95)
    p99_server = calculate_percentile(server_latencies, 0.99)


    print(f"Throughput & Token Metrics:")
    print(f"  - Total generated tokens:   {total_tokens} tokens")
    print(f"  - Overall request rate:     {len(results)/total_wall_time:.2f} req/sec")
    print(f"  - Combined generation rate: {total_tokens/total_wall_time:.2f} tok/sec")
    print()
    
    print(f"Latency Percentiles:")
    print(f"  | Metric (ms)              | Average  | Min      | p50 (Med)| p95      | p99      | Max      |")
    print(f"  |--------------------------|----------|----------|----------|----------|----------|----------|")
    print(f"  | Client-Side Latency      | {avg_client:8.1f} | {client_latencies[0]:8.1f} | {p50_client:8.1f} | {p95_client:8.1f} | {p99_client:8.1f} | {client_latencies[-1]:8.1f} |")
    print(f"  | Server-Side TTFT (First) | {avg_ttft:8.1f} | {server_ttfts[0]:8.1f} | {p50_ttft:8.1f} | {p95_ttft:8.1f} | {p99_ttft:8.1f} | {server_ttfts[-1]:8.1f} |")
    print(f"  | Server-Side Gen Latency  | {avg_server:8.1f} | {server_latencies[0]:8.1f} | {p50_server:8.1f} | {p95_server:8.1f} | {p99_server:8.1f} | {server_latencies[-1]:8.1f} |")
    print()
    
    print("Individual Requests detail:")
    print("  | Req ID | Status  | Client Latency (ms) | Server TTFT (ms) | Gen Tokens | Rate (tok/s) |")
    print("  |--------|---------|---------------------|------------------|------------|--------------|")
    for r in sorted(results, key=lambda x: x["id"]):
        if r["status"] == "success":
            print(f"  | {r['id']:6d} | SUCCESS | {r['client_latency_ms']:19.1f} | {r['server_ttft_ms']:16.1f} | {r['tokens_generated']:10d} | {r['tokens_per_sec']:12.1f} |")
        else:
            print(f"  | {r['id']:6d} | FAILED  | {r['client_latency_ms']:19.1f} | {'N/A':16} | {'N/A':10} | {'N/A':12} |")
    
    print("\nNote: Because the server processes requests sequentially on a single GPU queue without batching,")
    print("consecutive requests wait behind prior requests, causing higher Client Latency & Server TTFT for later runs.")

async def main():
    parser = argparse.ArgumentParser(description="Load and latency benchmarking client for LLM API.")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base URL of the running API server")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests to trigger")
    parser.add_argument("--tokens", type=int, default=100, help="Max tokens to generate per request")
    args = parser.parse_args()
    
    print(f"Starting load test on server: {args.url}")
    print(f"Concurrency level: {args.concurrency}")
    print(f"Tokens limit per request: {args.tokens}")
    print(f"Prompt text: '{TEST_PROMPT}'")
    
    # Establish connection
    limits = httpx.Limits(max_keepalive_connections=args.concurrency, max_connections=args.concurrency * 2)
    async with httpx.AsyncClient(limits=limits, timeout=180.0) as client:
        # Check API health
        try:
            health_check = await client.get(f"{args.url}/health")
            if health_check.status_code == 200:
                print(f"Health check passed: {health_check.json()}")
            else:
                print(f"Warning: Health check returned status {health_check.status_code}")
        except Exception as e:
            print(f"Failed to connect to server at {args.url}: {e}")
            sys.exit(1)
        
        # Run streaming test first to verify correctness
        await test_streaming_endpoint(client, args.url)
        
        # Run concurrent load test
        print(f"\nLaunching {args.concurrency} concurrent requests now...")
        t_start = time.perf_counter()
        tasks = [
            send_single_request(client, args.url, TEST_PROMPT, args.tokens, i + 1)
            for i in range(args.concurrency)
        ]
        results = await asyncio.gather(*tasks)
        t_end = time.perf_counter()
        
        print_results_table(results, t_end - t_start)

if __name__ == "__main__":
    asyncio.run(main())
