//! Chat-template rendering — model-specific message → token-id pipelines.
//!
//! Right now this only handles Gemma 4's chat template. The template
//! ships in the GGUF as a Jinja blob, but for the basic chat case
//! (no tool calls, no reasoning channel) the layout is small enough
//! to render directly with the runtime's existing SPM tokenizer plus
//! the three special-token ids that delimit a turn.

use crate::tokenizer::{GemmaTokenizer, Tokenizer};

/// Best-effort identification of a model's chat template by matching
/// signature strings in the GGUF `tokenizer.chat_template` jinja blob.
/// Returns a stable label or `Unknown(template_first_120_chars)`. Used
/// at serve-load to log "you loaded a {family} model; serve will apply
/// the {qwen3,gemma4} template" — if the family doesn't match, the
/// operator sees the mismatch immediately instead of after the first
/// garbled generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatTemplateFamily {
    /// Qwen 3.5/3.6 `<|im_start|>role\n…<|im_end|>\n`. Server applies.
    Qwen3,
    /// Gemma 4 `<|turn>role\n…<turn|>\n`. Server applies.
    Gemma4,
    /// Mistral `[INST] … [/INST]`. Not currently applied by serve.
    Mistral,
    /// Llama 3 `<|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>`.
    /// Not currently applied by serve.
    Llama3,
    /// DeepSeek chat (v2.5+). Not currently applied by serve.
    DeepSeek,
    /// ChatML generic (`<|im_start|>…<|im_end|>`) but vocab not Qwen-3.
    ChatML,
    /// We don't have a signature match. Holds the template prefix for
    /// operator inspection.
    Unknown(String),
}

impl ChatTemplateFamily {
    pub fn label(&self) -> &str {
        match self {
            Self::Qwen3      => "qwen3",
            Self::Gemma4     => "gemma4",
            Self::Mistral    => "mistral",
            Self::Llama3     => "llama3",
            Self::DeepSeek   => "deepseek",
            Self::ChatML     => "chatml",
            Self::Unknown(_) => "unknown",
        }
    }

    /// Whether the serve worker can apply this template natively (i.e.,
    /// our `format_*` helpers cover it). Currently only Qwen3 + Gemma4.
    pub fn supported_by_serve(&self) -> bool {
        matches!(self, Self::Qwen3 | Self::Gemma4)
    }
}

/// Detect the chat template family from a raw GGUF
/// `tokenizer.chat_template` string. Heuristic match on stable
/// signatures — works on real-world templates because each family's
/// template is essentially a unique string.
pub fn detect_chat_template(jinja: &str) -> ChatTemplateFamily {
    // Order matters: more-specific matches come first.
    // Gemma 4: `<|turn>` is the unique delimiter.
    if jinja.contains("<|turn>") || jinja.contains("<turn|>") {
        return ChatTemplateFamily::Gemma4;
    }
    // Llama 3: <|start_header_id|>role<|end_header_id|>.
    if jinja.contains("start_header_id") || jinja.contains("eot_id") {
        return ChatTemplateFamily::Llama3;
    }
    // Mistral / Mixtral: [INST] wrapping.
    if jinja.contains("[INST]") || jinja.contains("[/INST]") {
        return ChatTemplateFamily::Mistral;
    }
    // DeepSeek: `User:` / `Assistant:` literal + `<|begin_of_sentence|>`.
    if jinja.contains("<|begin_of_sentence|>") && jinja.contains("Assistant:") {
        return ChatTemplateFamily::DeepSeek;
    }
    // Qwen 3.5/3.6: <|im_start|>…<|im_end|> with role variable. The
    // Qwen-3 jinja also contains `image_count` / `video_count` setup
    // for multi-modal turns; older Qwen-2 ChatML doesn't.
    if jinja.contains("<|im_start|>") {
        if jinja.contains("image_count") || jinja.contains("248045")
            || jinja.contains("Qwen3") || jinja.contains("qwen3") {
            return ChatTemplateFamily::Qwen3;
        }
        return ChatTemplateFamily::ChatML;
    }
    // Fallback — first ~120 chars of the template so an operator can
    // grep for it.
    let prefix: String = jinja.chars().take(120).collect();
    ChatTemplateFamily::Unknown(prefix)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    /// The literal role string the Gemma 4 chat template writes after
    /// `<|turn>` — `assistant` is rendered as `model` to match the
    /// vocab entry the model was trained on.
    pub fn template_name(&self) -> &'static str {
        match self {
            Role::System    => "system",
            Role::User      => "user",
            Role::Assistant => "model",
        }
    }
}

pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

// Gemma 4 chat-template delimiter ids — verified against the
// `google/gemma-4-*-it` vocab by round-tripping
// `[2, 105, 2364, 107, …, 106, 107]` through generate-text and
// confirming the decode produced `<bos><|turn>user\n…<turn|>\n`.
// `TURN_CLOSE` is also the model's EOS id.
const TURN_OPEN:  u32 = 105;   // <|turn>
const TURN_CLOSE: u32 = 106;   // <turn|>
const NEWLINE:    u32 = 107;   // \n

/// Render `messages` into a Gemma 4 chat-template token sequence,
/// matching `google/gemma-4-*-it/chat_template.jinja` for the basic
/// chat case.
///
/// Always prepends BOS. If `add_generation_prompt` is true, ends with
/// `<|turn>model\n` so the model decodes the assistant turn directly.
pub fn format_gemma4(tok: &GemmaTokenizer, messages: &[ChatMessage],
                     add_generation_prompt: bool) -> Result<Vec<u32>, String>
{
    let approx = 8 + messages.iter().map(|m| 8 + m.content.len() / 2).sum::<usize>();
    let mut out = Vec::with_capacity(approx);
    out.push(tok.bos_id);
    for m in messages {
        let role_id = tok.token_id(m.role.template_name()).ok_or_else(|| format!(
            "gemma4 chat: role '{}' not in vocab", m.role.template_name()))?;
        out.push(TURN_OPEN);
        out.push(role_id);
        out.push(NEWLINE);
        out.extend(tok.encode(&m.content));
        out.push(TURN_CLOSE);
        out.push(NEWLINE);
    }
    if add_generation_prompt {
        let model_id = tok.token_id("model").ok_or("gemma4 chat: 'model' not in vocab")?;
        out.push(TURN_OPEN);
        out.push(model_id);
        out.push(NEWLINE);
    }
    Ok(out)
}

/// Per-turn extension for an already-rendered Gemma 4 prefix: emits
/// the tokens for `<|turn>user\n{content}<turn|>\n<|turn>model\n`. Use
/// when a prior conversation prefix is already in the KV cache and
/// you want to append a new user turn + generation prompt.
pub fn format_gemma4_user_turn(tok: &GemmaTokenizer, content: &str)
    -> Result<Vec<u32>, String>
{
    let user_id  = tok.token_id("user").ok_or("gemma4 chat: 'user' not in vocab")?;
    let model_id = tok.token_id("model").ok_or("gemma4 chat: 'model' not in vocab")?;
    let mut out = Vec::with_capacity(8 + content.len() / 2);
    out.push(TURN_OPEN);
    out.push(user_id);
    out.push(NEWLINE);
    out.extend(tok.encode(content));
    out.push(TURN_CLOSE);
    out.push(NEWLINE);
    out.push(TURN_OPEN);
    out.push(model_id);
    out.push(NEWLINE);
    Ok(out)
}

// Qwen 3.5/3.6 chat-template delimiter ids — verified against the
// `Qwen3.6-*` vocab. (The ids changed vs Qwen 2.x — these are not the
// 151644/151645 you may see in older Qwen docs.)
const QWEN_IM_START: u32 = 248045;   // <|im_start|>
const QWEN_IM_END:   u32 = 248046;   // <|im_end|>
const QWEN_NEWLINE:  u32 = 198;      // \n

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn detects_qwen3() {
        let q = r#"{%- set image_count = namespace(value=0) %}\n<|im_start|>"#;
        assert_eq!(detect_chat_template(q), ChatTemplateFamily::Qwen3);
    }
    #[test]
    fn detects_gemma4() {
        let g = "{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n{% endfor %}";
        assert_eq!(detect_chat_template(g), ChatTemplateFamily::Gemma4);
    }
    #[test]
    fn detects_mistral() {
        let m = "{% for m in messages %}[INST] {{ m.content }} [/INST]{% endfor %}";
        assert_eq!(detect_chat_template(m), ChatTemplateFamily::Mistral);
    }
    #[test]
    fn detects_llama3() {
        let l = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n";
        assert_eq!(detect_chat_template(l), ChatTemplateFamily::Llama3);
    }
    #[test]
    fn detects_chatml_when_no_qwen_signature() {
        let c = "<|im_start|>system\n{{ system }}<|im_end|>";
        assert_eq!(detect_chat_template(c), ChatTemplateFamily::ChatML);
    }
    #[test]
    fn unknown_keeps_prefix() {
        let other = "{% completely unfamiliar template {{ stuff }} %}";
        match detect_chat_template(other) {
            ChatTemplateFamily::Unknown(s) => assert!(!s.is_empty()),
            o => panic!("expected Unknown, got {o:?}"),
        }
    }
    #[test]
    fn think_spec_reads_each_family_default() {
        let q38 = "{%- if enable_thinking is undefined or enable_thinking is true %}\n\
                   {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}";
        assert_eq!(Qwen3ThinkSpec::from_template(q38),
                   Qwen3ThinkSpec { default_on: true, reasoning_effort: true });
        let q35 = "{%- if enable_thinking is defined and enable_thinking is true %}\n\
                   {{- '<think>\n' }}";
        assert_eq!(Qwen3ThinkSpec::from_template(q35), Qwen3ThinkSpec::QWEN35);
    }
    #[test]
    fn reasoning_instruction_levels() {
        assert!(qwen3_reasoning_instruction("xhigh").unwrap().is_some());
        assert_eq!(qwen3_reasoning_instruction("high"), qwen3_reasoning_instruction("xhigh"));
        assert!(qwen3_reasoning_instruction("medium").unwrap().is_none());
        assert!(qwen3_reasoning_instruction("low").unwrap().is_some());
        assert!(qwen3_reasoning_instruction("max").is_err());
    }
    #[test]
    fn supported_by_serve_is_correct() {
        assert!(ChatTemplateFamily::Qwen3.supported_by_serve());
        assert!(ChatTemplateFamily::Gemma4.supported_by_serve());
        assert!(!ChatTemplateFamily::Llama3.supported_by_serve());
        assert!(!ChatTemplateFamily::Mistral.supported_by_serve());
    }
}

/// What a Qwen 3.x chat template does about thinking, read off the
/// GGUF's jinja blob. The family changed its default between releases:
///
/// * Qwen 3.5 / 3.6: `enable_thinking is defined and … is true` — the
///   generation prompt ends `<think>\n\n</think>\n\n` (thinking OFF)
///   unless the caller asks for it.
/// * Qwen 3.8: `enable_thinking is undefined or … is true` — thinking
///   ON by default, the generation prompt ends with an opened
///   `<think>\n`, and a `reasoning_effort` (`xhigh` default, `medium`,
///   `low`; `high` maps to `xhigh`) prepends an instruction sentence to
///   the system turn.
///
/// `format_qwen3` renders whichever the template calls for, so a model
/// gets the same prompt it would from `apply_chat_template`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen3ThinkSpec {
    /// Thinking on when the request does not say.
    pub default_on: bool,
    /// The template understands `reasoning_effort` (Qwen 3.8+).
    pub reasoning_effort: bool,
}

impl Qwen3ThinkSpec {
    /// Qwen 3.5-style: no thinking unless asked, no effort levels.
    pub const QWEN35: Self = Self { default_on: false, reasoning_effort: false };

    pub fn from_template(template: &str) -> Self {
        let default_on = template.contains("enable_thinking is undefined or enable_thinking is true");
        let reasoning_effort = template.contains("reasoning_effort");
        Self { default_on, reasoning_effort }
    }
}

/// Per-request thinking controls for `format_qwen3`. `None` means "the
/// template's default".
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Qwen3ThinkOpts {
    pub enable_thinking: Option<bool>,
    /// `xhigh` | `high` | `medium` | `low` (Qwen 3.8+; ignored elsewhere).
    pub reasoning_effort: Option<String>,
}

/// The instruction sentence Qwen 3.8's template prepends to the system
/// turn for a reasoning effort, if any (`medium` adds none). Verbatim
/// from the template.
fn qwen3_reasoning_instruction(effort: &str) -> Result<Option<&'static str>, String> {
    Ok(match effort {
        "xhigh" | "high" => Some("Reasoning effort is set to xhigh. Please think carefully \
through the task, validate key assumptions, consider plausible alternatives, and \
prioritize correctness, consistency, and clarity in the final answer."),
        "low" => Some("Reasoning effort is set to low. Keep your thinking brief and \
focused, moving directly to the conclusion without unnecessary elaboration."),
        "medium" => None,
        other => return Err(format!(
            "qwen3 chat: unexpected reasoning_effort '{other}' (xhigh, high, medium, low)")),
    })
}

/// Render `messages` into a Qwen 3.x chat-template token sequence,
/// matching the model's `chat_template.jinja` for the basic chat case
/// (system / user / assistant; no tool calls, no reasoning history).
///
/// Qwen's chat template does **not** prepend BOS — the model expects to
/// start with `<|im_start|>` directly. If `add_generation_prompt` is
/// true, ends with `<|im_start|>assistant\n` plus the thinking
/// scaffold `spec` and `opts` resolve to: an opened `<think>\n` when
/// thinking is on, a closed empty `<think>\n\n</think>\n\n` when off.
pub fn format_qwen3(tok: &Tokenizer, messages: &[ChatMessage], add_generation_prompt: bool,
                    spec: Qwen3ThinkSpec, opts: &Qwen3ThinkOpts) -> Result<Vec<u32>, String>
{
    let thinking = opts.enable_thinking.unwrap_or(spec.default_on);
    // Reasoning-effort instruction (3.8+): part of the system turn,
    // creating one if the conversation has none.
    let instruction = if thinking && spec.reasoning_effort {
        qwen3_reasoning_instruction(opts.reasoning_effort.as_deref().unwrap_or("xhigh"))?
    } else { None };

    let approx = 8 + messages.iter().map(|m| 8 + m.content.len() / 2).sum::<usize>();
    let mut out = Vec::with_capacity(approx);
    let role = |name: &str| tok.token_id(name)
        .ok_or_else(|| format!("qwen3 chat: role '{name}' not in vocab"));
    let has_system = messages.first().map(|m| m.role == Role::System).unwrap_or(false);
    if let (Some(instr), false) = (instruction, has_system) {
        out.push(QWEN_IM_START);
        out.push(role("system")?);
        out.push(QWEN_NEWLINE);
        out.extend(tok.encode(instr));
        out.push(QWEN_IM_END);
        out.push(QWEN_NEWLINE);
    }
    for (i, m) in messages.iter().enumerate() {
        let role_str = match m.role {
            Role::System    => "system",
            Role::User      => "user",
            Role::Assistant => "assistant",       // Qwen keeps "assistant" verbatim
        };
        out.push(QWEN_IM_START);
        out.push(role(role_str)?);
        out.push(QWEN_NEWLINE);
        if i == 0 && has_system {
            if let Some(instr) = instruction {
                out.extend(tok.encode(&format!("{instr}\n\n{}", m.content)));
            } else {
                out.extend(tok.encode(&m.content));
            }
        } else {
            out.extend(tok.encode(&m.content));
        }
        out.push(QWEN_IM_END);
        out.push(QWEN_NEWLINE);
    }
    if add_generation_prompt {
        out.push(QWEN_IM_START);
        out.push(role("assistant")?);
        out.push(QWEN_NEWLINE);
        let think = tok.token_id("<think>").ok_or("qwen3 chat: '<think>' not in vocab")?;
        out.push(think);
        if thinking {
            out.push(QWEN_NEWLINE);
        } else {
            let end_think = tok.token_id("</think>").ok_or("qwen3 chat: '</think>' not in vocab")?;
            out.extend(tok.encode("\n\n"));
            out.push(end_think);
            out.extend(tok.encode("\n\n"));
        }
    }
    Ok(out)
}
