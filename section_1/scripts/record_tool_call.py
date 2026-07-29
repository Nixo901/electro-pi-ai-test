import asyncio
import os
import json
import sys
from pathlib import Path
from groq import AsyncGroq

# Windows console encoding fix — must happen before any print()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Adjust python path
sys.path.append(str(Path(__file__).parent.parent / "src"))


from arabic_voice_agent.config import Settings
from arabic_voice_agent.services.order_service import OrderService
from arabic_voice_agent.prompts import SYSTEM_PROMPT

async def main():
    print("Loading settings...", flush=True)
    settings = Settings.from_env()
    
    # We will try the settings LLM model first, but fallback to llama-3.3-70b-versatile
    # since openai/gpt-oss-120b might be a custom or deprecated endpoint.
    models_to_try = [
        "llama-3.3-70b-versatile",
        settings.llm_model,
        "llama3-8b-8192"
    ]
    
    client = AsyncGroq(api_key=settings.groq_api_key)
    orders = OrderService()
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_order_status",
                "description": "Look up an order by its numeric order identifier.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The numeric order ID (e.g. 1002)."
                        }
                    },
                    "required": ["order_id"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "ما حالة طلبي رقم 1002؟"}
    ]
    
    print("\nSending request to Groq with tool definition...", flush=True)
    
    response = None
    selected_model = None
    for model in models_to_try:
        try:
            print(f"Trying model: {model}...", flush=True)
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0 # deterministic
            )
            selected_model = model
            print(f"Success with model: {model}", flush=True)
            break
        except Exception as e:
            print(f"Failed with model {model}: {e}", flush=True)
            
    if response is None:
        print("Error: All LLM models failed to execute.", file=sys.stderr, flush=True)
        sys.exit(1)
        
    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None)
    
    log_artifact = {
        "timestamp_utc": "", # will set below
        "model_used": selected_model,
        "user_utterance": "ما حالة طلبي رقم 1002؟",
        "llm_tool_call": None,
        "tool_result": None,
        "expected_agent_reply": None
    }
    
    import datetime
    log_artifact["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    if tool_calls:
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        print(f"\n[LLM Decision] Invoke tool call: '{tool_name}' with args: {tool_args}", flush=True)
        
        log_artifact["llm_tool_call"] = {
            "name": tool_name,
            "arguments": tool_args
        }
        
        # Execute the python tool
        if tool_name == "get_order_status":
            order_id = tool_args.get("order_id")
            print(f"Executing Python Tool call: get_order_status(order_id='{order_id}')...", flush=True)
            tool_output = await orders.get_status(order_id)
            print(f"Tool Result: {tool_output}", flush=True)
            
            log_artifact["tool_result"] = tool_output
            
            # Send result back to LLM
            messages.append(message) # Add assistant message with tool calls
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": tool_output
            })
            
            print("Sending tool result back to LLM to formulate final reply...", flush=True)
            final_response = await client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=0.3
            )
            final_reply = final_response.choices[0].message.content
            print(f"Final LLM Reply: {final_reply}", flush=True)
            
            log_artifact["expected_agent_reply"] = final_reply
    else:
        print("\nWarning: The LLM did not choose to invoke any tool call.", flush=True)
        print(f"LLM output was: {message.content}", flush=True)
        
    # Save the log to the artifacts folder
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    artifact_path = artifacts_dir / "tool_call_demo.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(log_artifact, f, indent=2, ensure_ascii=False)
        
    print(f"\nReal LLM tool call log saved to: {artifact_path.resolve()}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
