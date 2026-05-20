//! Chat-template rendering — model-specific message → token-id pipelines.
//!
//! Right now this only handles Gemma 4's chat template. The template
//! ships in the GGUF as a Jinja blob, but for the basic chat case
//! (no tool calls, no reasoning channel) the layout is small enough
//! to render directly with the runtime's existing SPM tokenizer plus
//! the three special-token ids that delimit a turn.

use crate::tokenizer::{GemmaTokenizer, Tokenizer};

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

// Qwen 3.5/3.6 chat-template delimiter ids — verified against the
// `Qwen3.6-*` vocab. (The ids changed vs Qwen 2.x — these are not the
// 151644/151645 you may see in older Qwen docs.)
const QWEN_IM_START: u32 = 248045;   // <|im_start|>
const QWEN_IM_END:   u32 = 248046;   // <|im_end|>
const QWEN_NEWLINE:  u32 = 198;      // \n

/// Render `messages` into a Qwen 3.5/3.6 chat-template token sequence,
/// matching `Qwen3.5-*/chat_template.jinja` for the basic chat case
/// (system / user / assistant; no tool calls, no `<think>` channel).
///
/// Qwen's chat template does **not** prepend BOS — the model expects to
/// start with `<|im_start|>` directly. If `add_generation_prompt` is
/// true, ends with `<|im_start|>assistant\n` so the model decodes the
/// assistant turn directly.
pub fn format_qwen3(tok: &Tokenizer, messages: &[ChatMessage],
                    add_generation_prompt: bool) -> Result<Vec<u32>, String>
{
    let approx = 8 + messages.iter().map(|m| 8 + m.content.len() / 2).sum::<usize>();
    let mut out = Vec::with_capacity(approx);
    for m in messages {
        let role_str = match m.role {
            Role::System    => "system",
            Role::User      => "user",
            Role::Assistant => "assistant",       // Qwen keeps "assistant" verbatim
        };
        let role_id = tok.token_id(role_str).ok_or_else(|| format!(
            "qwen3 chat: role '{role_str}' not in vocab"))?;
        out.push(QWEN_IM_START);
        out.push(role_id);
        out.push(QWEN_NEWLINE);
        out.extend(tok.encode(&m.content));
        out.push(QWEN_IM_END);
        out.push(QWEN_NEWLINE);
    }
    if add_generation_prompt {
        let role_id = tok.token_id("assistant").ok_or("qwen3 chat: 'assistant' not in vocab")?;
        out.push(QWEN_IM_START);
        out.push(role_id);
        out.push(QWEN_NEWLINE);
    }
    Ok(out)
}
