# Section 1 write-up

## Design and trade-offs

The agent uses the official LiveKit Agents `AgentSession` pipeline with Groq Whisper Large V3 Turbo for Arabic transcription, Groq GPT-OSS-120B for reasoning and tool selection, and Groq Orpheus Arabic Saudi with the Abdullah voice for speech output. LiveKit handles room/media orchestration, voice activity detection, turn-taking, publication, and interruption mechanics; the application does not reimplement WebRTC. The modular standalone provider interfaces support isolated testing and a vendor swap, while the production room pipeline uses the maintained LiveKit Groq adapters.

`get_order_status` is deliberately narrow: it accepts only a numeric order id, returns only a limited mocked state, and has no access to customer data. A production implementation would authenticate the participant, bind the lookup to that authenticated identity rather than trusting a spoken number, enforce a JSON-schema-compatible input contract, set a timeout, log an audit event without sensitive data, and map upstream failures to a brief safe response. A second tool (for example, `cancel_order`) would require explicit confirmation, idempotency keys, authorization checks, and human escalation for exceptions.

## Barge-in / interruption handling

`allow_interruptions=True` enables the framework to interrupt ongoing assistant playback when new user speech is detected. In production I would tune endpointing thresholds per language and test noisy mobile networks, reject extremely short VAD activations, cancel the pending LLM/TTS task on a confirmed interruption, and preserve the partial conversation safely. I would measure interruption success rate, time-to-first-audio, and false-interrupt rate, then adjust VAD and turn-detection settings against recorded consented test sessions.

## Bonus: provider swap

The app isolates its offline STT/LLM/TTS dependencies behind `STTProvider`, `LLMProvider`, and `TTSProvider`. To switch the LiveKit STT component, replace only `groq.STT(...)` in `build_session` with a supported alternative such as `deepgram.STT(model="nova-3", language="multi")`, add its package/key, and leave `FoodDeliveryAgent`, its tool, and the rest of the flow unchanged. This is intentionally a small composition-root change rather than a rewrite.
