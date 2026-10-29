"""Step 2 · The AI language model from Hugging Face.

`LLM(...)` downloads Qwen3.5 (about 19 GB the first time) and loads it in 4-bit so it fits on a T4 GPU.
`llm.ask(prompt, schema)` is what the rest of the project uses:
  - with a pydantic `schema`, the reply must be one JSON object of that shape; small slips are repaired
    (a `null` in place of an empty field, a LaTeX backslash such as \\Omega);
  - an unusable reply raises ValueError that says why, and the full reply is added to llm_bad_replies.txt;
  - every good reply is cached in llm_cache.json, so re-running never repeats the same work.
Warnings that `causal_conv1d` or `flash-linear-attention` "falls back to its reference PyTorch implementation"
are normal on a T4; they only mean the model runs slower.
"""
import hashlib
import json
import re
import time
from pathlib import Path

from pydantic import ValidationError


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def drop_nulls(x):
    """Remove null values so pydantic uses the defaults (models often write null for an empty field)."""
    if isinstance(x, dict):
        return {k: drop_nulls(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [drop_nulls(v) for v in x if v is not None]
    return x


def read_json(reply, schema):
    """The JSON object in a reply that has every field the schema needs.
    It skips words or ``` around the JSON, and smaller objects inside it (for example one part of a question)."""
    text = re.sub(r'(\\\\)|\\(?![/"bfnrtu])', lambda m: m[1] or r"\\", reply)   # LaTeX \Omega -> \\Omega, valid JSON
    needed = {name for name, field in schema.model_fields.items() if field.is_required()}
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj = decoder.raw_decode(text, m.start())[0]
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and needed <= obj.keys():
            return drop_nulls(obj)
    if "{" not in text:
        raise ValueError("the reply has no JSON object")
    raise ValueError(f"no complete JSON object with the fields {sorted(needed)} - the reply may be cut off")


class LLM:
    """The language model, loaded once, with a cache of good replies on disk."""

    def __init__(self, model_name, work_dir):
        import torch
        from transformers import AutoProcessor

        assert torch.cuda.is_available(), "No GPU: choose Runtime > Change runtime type > T4 GPU, then run again."
        self.torch, self.model_name = torch, model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        fast_type = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16   # a T4 is fastest in float16
        self.model = self._load(fast_type)
        start = time.time()
        reply = self.generate("In one sentence, what is a photon?", max_new_tokens=60)
        if fast_type == torch.float16 and not re.search(r"[A-Za-z]{4}", reply):
            print("float16 gave broken output, so the model is reloaded in bfloat16 (slower but safe).")
            del self.model
            torch.cuda.empty_cache()
            self.model = self._load(torch.bfloat16)
            reply = self.generate("In one sentence, what is a photon?", max_new_tokens=60)
        print(reply, f"({time.time() - start:.0f} s)")

        self.cache_file = Path(work_dir) / "llm_cache.json"
        self.bad_replies = Path(work_dir) / "llm_bad_replies.txt"      # unusable replies are saved here
        self.cache = json.loads(self.cache_file.read_text()) if self.cache_file.exists() else {}

    def _load(self, dtype):
        """Load the model with 4-bit weights (about 7 GB of GPU memory for the 9B model), all on the GPU."""
        from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        return Qwen3_5ForConditionalGeneration.from_pretrained(self.model_name, quantization_config=quant,
                                                               dtype=dtype, device_map={"": 0}).eval()

    def generate(self, prompt, temperature=0.0, max_new_tokens=1500):
        """Send one prompt to the model and return its reply as text."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        sampling = dict(do_sample=True, temperature=temperature, top_p=0.9) if temperature else dict(do_sample=False)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.processor.tokenizer.eos_token_id, **sampling)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        if len(new_tokens) >= max_new_tokens:
            print(f"  (the reply was cut off at {max_new_tokens} tokens)")
        reply = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return re.sub(r"<think>.*?</think>", "", reply, flags=re.S).strip()

    def ask(self, prompt, schema=None, temperature=0.0, attempt=0, max_new_tokens=1500):
        """Ask the model. With a pydantic `schema`, the reply must be JSON of that shape and is returned parsed."""
        if schema:                                             # the prompt itself shows the JSON format to use
            prompt += "\n\nReply with ONLY one JSON object, written compactly, with no other text."
        key = hashlib.sha256(f"{self.model_name}|{temperature}|{attempt}|{prompt}".encode()).hexdigest()
        reply = self.cache.get(key) or self.generate(prompt, temperature, max_new_tokens)
        result = reply
        if schema:
            try:
                result = schema.model_validate(read_json(reply, schema))
            except (ValueError, ValidationError) as e:         # unusable reply: explain why, save it, do NOT cache it
                reason = ("; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:3])
                          if isinstance(e, ValidationError) else str(e))
                with open(self.bad_replies, "a", encoding="utf-8") as f:
                    f.write(f"\n\n===== {schema.__name__}: {reason}\n{reply}")
                raise ValueError(f"{reason}. The reply starts: {reply[:150]!r}") from None
        if key not in self.cache:
            self.cache[key] = reply
            save_json(self.cache_file, self.cache)
        return result
